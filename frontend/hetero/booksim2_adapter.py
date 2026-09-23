"""ctypes owner for the ATLAS-patched BookSim2 cycle adapter."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Mapping


class BookSim2AdapterError(RuntimeError):
    """Raised when the real BookSim2 adapter rejects or loses a packet."""


class _CompletionV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flow_id", ctypes.c_uint64),
        ("source_node", ctypes.c_uint32),
        ("destination_node", ctypes.c_uint32),
        ("flit_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("accepted_cycle", ctypes.c_uint64),
        ("completion_cycle", ctypes.c_uint64),
    ]


class _StatsV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("cycles", ctypes.c_uint64),
        ("accepted_packets", ctypes.c_uint64),
        ("injected_packets", ctypes.c_uint64),
        ("delivered_packets", ctypes.c_uint64),
        ("accepted_flits", ctypes.c_uint64),
        ("delivered_flits", ctypes.c_uint64),
        ("outstanding_credits", ctypes.c_uint64),
        ("queued_packets", ctypes.c_uint64),
        ("completed_unpolled_packets", ctypes.c_uint64),
        ("network_drained", ctypes.c_uint32),
        ("zero_in_flight", ctypes.c_uint32),
    ]


class BookSim2CycleAdapter:
    ABI_VERSION = 1
    ABI_NAME = "gpu-atlas-booksim2-adapter/v1"

    def __init__(self, library: Path, network_config: Path, node_count: int) -> None:
        if not library.is_file():
            raise BookSim2AdapterError(f"BookSim2 adapter library is missing: {library}")
        if not network_config.is_file():
            raise BookSim2AdapterError(
                f"BookSim2 network configuration is missing: {network_config}"
            )
        if node_count < 2:
            raise BookSim2AdapterError("BookSim2 requires at least two nodes")
        self._library = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self._bind()
        abi = self._library.heterosim_booksim2_abi().decode("utf-8")
        if abi != self.ABI_NAME:
            raise BookSim2AdapterError(f"unexpected BookSim2 adapter ABI: {abi}")
        self._handle = self._library.heterosim_booksim2_create(
            os.fsencode(network_config), node_count
        )
        if not self._handle:
            raise BookSim2AdapterError(self._last_error("BookSim2 create failed"))
        self.node_count = node_count
        self._closed = False
        self._accepted: set[int] = set()
        self._completed: set[int] = set()

    def _bind(self) -> None:
        library = self._library
        library.heterosim_booksim2_abi.argtypes = []
        library.heterosim_booksim2_abi.restype = ctypes.c_char_p
        library.heterosim_booksim2_last_error.argtypes = []
        library.heterosim_booksim2_last_error.restype = ctypes.c_char_p
        library.heterosim_booksim2_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        library.heterosim_booksim2_create.restype = ctypes.c_void_p
        library.heterosim_booksim2_destroy.argtypes = [ctypes.c_void_p]
        library.heterosim_booksim2_destroy.restype = None
        library.heterosim_booksim2_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.heterosim_booksim2_submit.restype = ctypes.c_int
        library.heterosim_booksim2_step.argtypes = [ctypes.c_void_p]
        library.heterosim_booksim2_step.restype = ctypes.c_int
        library.heterosim_booksim2_pop_completion.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_CompletionV1),
        ]
        library.heterosim_booksim2_pop_completion.restype = ctypes.c_int
        library.heterosim_booksim2_get_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_StatsV1),
        ]
        library.heterosim_booksim2_get_stats.restype = ctypes.c_int

    def _last_error(self, fallback: str) -> str:
        raw = self._library.heterosim_booksim2_last_error()
        return raw.decode("utf-8", errors="replace") if raw else fallback

    def _require_open(self) -> None:
        if self._closed:
            raise BookSim2AdapterError("BookSim2 adapter is closed")

    def submit(
        self, flow_id: int, source_node: int, destination_node: int, flit_count: int
    ) -> None:
        self._require_open()
        if flow_id in self._accepted:
            raise BookSim2AdapterError(f"duplicate flow ID {flow_id}")
        result = int(
            self._library.heterosim_booksim2_submit(
                self._handle, flow_id, source_node, destination_node, flit_count
            )
        )
        if result != 1:
            raise BookSim2AdapterError(self._last_error("BookSim2 submit failed"))
        self._accepted.add(flow_id)

    def step(self) -> None:
        self._require_open()
        if int(self._library.heterosim_booksim2_step(self._handle)) != 1:
            raise BookSim2AdapterError(self._last_error("BookSim2 step failed"))

    def pop_completions(self) -> list[dict[str, int]]:
        self._require_open()
        result: list[dict[str, int]] = []
        for destination in range(self.node_count):
            while True:
                completion = _CompletionV1()
                found = int(
                    self._library.heterosim_booksim2_pop_completion(
                        self._handle, destination, ctypes.byref(completion)
                    )
                )
                if found == 0:
                    break
                if found != 1:
                    raise BookSim2AdapterError(
                        self._last_error("BookSim2 completion failed")
                    )
                if (
                    completion.abi_version != self.ABI_VERSION
                    or completion.struct_size != ctypes.sizeof(_CompletionV1)
                    or completion.destination_node != destination
                    or completion.flow_id not in self._accepted
                    or completion.flow_id in self._completed
                ):
                    raise BookSim2AdapterError("invalid or duplicate BookSim2 completion")
                self._completed.add(int(completion.flow_id))
                result.append(
                    {
                        "flow_id": int(completion.flow_id),
                        "source_node": int(completion.source_node),
                        "destination_node": int(completion.destination_node),
                        "flit_count": int(completion.flit_count),
                        "accepted_cycle": int(completion.accepted_cycle),
                        "completion_cycle": int(completion.completion_cycle),
                    }
                )
        return sorted(result, key=lambda item: (item["completion_cycle"], item["flow_id"]))

    def stats(self) -> dict[str, object]:
        self._require_open()
        stats = _StatsV1()
        if int(self._library.heterosim_booksim2_get_stats(self._handle, ctypes.byref(stats))) != 1:
            raise BookSim2AdapterError(self._last_error("BookSim2 stats failed"))
        if stats.abi_version != self.ABI_VERSION or stats.struct_size != ctypes.sizeof(
            _StatsV1
        ):
            raise BookSim2AdapterError("invalid BookSim2 stats ABI")
        return {
            "schema_version": "hetero-booksim2-cycle-stats/v1",
            "adapter_abi": self.ABI_NAME,
            "cycles": int(stats.cycles),
            "accepted_packets": int(stats.accepted_packets),
            "injected_packets": int(stats.injected_packets),
            "delivered_packets": int(stats.delivered_packets),
            "accepted_flits": int(stats.accepted_flits),
            "delivered_flits": int(stats.delivered_flits),
            "outstanding_credits": int(stats.outstanding_credits),
            "queued_packets": int(stats.queued_packets),
            "completed_unpolled_packets": int(stats.completed_unpolled_packets),
            "network_drained": bool(stats.network_drained),
            "zero_in_flight": bool(stats.zero_in_flight),
            "accepted_flow_ids": len(self._accepted),
            "observed_completion_ids": len(self._completed),
        }

    def close(self) -> dict[str, object]:
        stats = self.stats()
        if not stats["zero_in_flight"]:
            raise BookSim2AdapterError("BookSim2 close requires zero in-flight work")
        self._library.heterosim_booksim2_destroy(self._handle)
        self._closed = True
        return stats


def _probe_once(library: Path, network_config: Path) -> dict[str, object]:
    adapter = BookSim2CycleAdapter(library, network_config, 4)
    adapter.submit(1, 0, 3, 2)
    adapter.submit(2, 1, 2, 3)
    completions: list[dict[str, int]] = []
    for _ in range(10000):
        adapter.step()
        completions.extend(adapter.pop_completions())
        if len(completions) == 2 and bool(adapter.stats()["zero_in_flight"]):
            break
    if len(completions) != 2 or not bool(adapter.stats()["zero_in_flight"]):
        raise BookSim2AdapterError("BookSim2 deterministic probe timed out")
    stats = adapter.close()
    return {"completions": completions, "stats": stats}


def probe_booksim2_adapter(library: Path, network_config: Path) -> Mapping[str, object]:
    """Run a two-leg real-network activation probe for the P25 gate."""

    first = _probe_once(library, network_config)
    second = _probe_once(library, network_config)
    first_stats = dict(first["stats"])
    return {
        "adapter_abi": BookSim2CycleAdapter.ABI_NAME,
        "cycle_step_api": int(first_stats["cycles"]) > 0,
        "credit_accounting": (
            int(first_stats["outstanding_credits"]) == 0
            and bool(first_stats["zero_in_flight"])
        ),
        "deterministic_probe_passed": first == second,
        "packet_conservation": (
            first_stats["accepted_packets"] == first_stats["delivered_packets"] == 2
        ),
        "flit_conservation": (
            first_stats["accepted_flits"] == first_stats["delivered_flits"] == 5
        ),
        "probe_cycles": first_stats["cycles"],
    }
