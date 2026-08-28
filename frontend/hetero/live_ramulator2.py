"""ctypes owner for the one live P10b-B Ramulator2 bridge instance."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Mapping


class LiveRamulator2Error(RuntimeError):
    """Raised when the live Ramulator2 ABI rejects or loses a request."""


class _ParentRequestV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("parent_id", ctypes.c_uint64),
        ("global_address", ctypes.c_uint64),
        ("size_bytes", ctypes.c_uint32),
        ("partition_id", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("byte_mask", ctypes.c_uint64 * 2),
        ("byte_mask_word_count", ctypes.c_uint32),
        ("sector_mask", ctypes.c_uint32),
        ("ordering_domain", ctypes.c_uint64),
        ("sequence_number", ctypes.c_uint64),
        ("qos_class", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("payload", ctypes.c_void_p),
    ]


class _ParentCompletionV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("parent_id", ctypes.c_uint64),
        ("partition_id", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("total_children", ctypes.c_uint32),
        ("completed_children", ctypes.c_uint32),
        ("durable", ctypes.c_uint32),
        ("initiator", ctypes.c_uint32),
        ("payload", ctypes.c_void_p),
    ]


_ENV_FIELDS = {
    "transaction_bytes": "HETEROSIM_DRAM_TRANSACTION_BYTES",
    "gateway_ingress_queue_depth": "HETEROSIM_GATEWAY_INGRESS_QUEUE_DEPTH",
    "gateway_parent_table_entries": "HETEROSIM_GATEWAY_PARENT_TABLE_ENTRIES",
    "gateway_issue_width": "HETEROSIM_GATEWAY_ISSUE_WIDTH",
    "write_ack_policy": "HETEROSIM_GATEWAY_WRITE_ACK_POLICY",
    "gpu_clock_hz": "HETEROSIM_GPU_CLOCK_HZ",
    "link_clock_hz": "HETEROSIM_LINK_CLOCK_HZ",
    "gateway_clock_hz": "HETEROSIM_GATEWAY_CLOCK_HZ",
    "dram_clock_hz": "HETEROSIM_DRAM_CLOCK_HZ",
    "request_bandwidth_Bps": "HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS",
    "response_bandwidth_Bps": "HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS",
    "request_header_bytes": "HETEROSIM_LINK_REQUEST_HEADER_BYTES",
    "response_header_bytes": "HETEROSIM_LINK_RESPONSE_HEADER_BYTES",
    "flit_bytes": "HETEROSIM_LINK_FLIT_BYTES",
    "propagation_latency_fs": "HETEROSIM_LINK_PROPAGATION_LATENCY_FS",
    "external_queue_depth": "HETEROSIM_LINK_QUEUE_DEPTH",
    "external_credits": "HETEROSIM_LINK_CREDITS",
    "duplex_mode": "HETEROSIM_LINK_DUPLEX_MODE",
}


class LiveRamulator2Bridge:
    """Own exactly one bridge handle and expose deterministic event stepping."""

    ABI_VERSION = 2
    GPU_INITIATOR = 0
    ATLAS_INITIATOR = 1
    SEND_INVALID = -1
    SEND_RETRY = 0
    SEND_ACCEPTED = 1

    def __init__(
        self,
        project_root: Path,
        config: Mapping[str, object],
    ) -> None:
        library = self._resolve(project_root, config.get("bridge_library"))
        ramulator_config = self._resolve(project_root, config.get("config_ref"))
        if not library.is_file():
            raise LiveRamulator2Error(f"Ramulator2 bridge library is missing: {library}")
        if not ramulator_config.is_file():
            raise LiveRamulator2Error(
                f"Ramulator2 configuration is missing: {ramulator_config}"
            )
        previous: dict[str, str | None] = {}
        for field, variable in _ENV_FIELDS.items():
            if field not in config:
                continue
            previous[variable] = os.environ.get(variable)
            os.environ[variable] = str(config[field])
        try:
            self._library = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            self._bind()
            self._handle = self._library.heterosim_ramulator_create(
                os.fsencode(ramulator_config), 0, 1
            )
        finally:
            for variable, value in previous.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value
        if not self._handle:
            raise LiveRamulator2Error("heterosim_ramulator_create returned null")
        self._closed = False
        self._accepted: set[int] = set()
        self._completed: set[int] = set()

    @staticmethod
    def _resolve(project_root: Path, raw: object) -> Path:
        if not isinstance(raw, str) or not raw:
            raise LiveRamulator2Error("live Ramulator2 path is required")
        path = Path(raw)
        return path if path.is_absolute() else (project_root / path).resolve()

    def _bind(self) -> None:
        lib = self._library
        lib.heterosim_ramulator_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.heterosim_ramulator_create.restype = ctypes.c_void_p
        lib.heterosim_ramulator_destroy.argtypes = [ctypes.c_void_p]
        lib.heterosim_ramulator_destroy.restype = None
        for name in ("heterosim_ramulator_send_v2", "heterosim_ramulator_send_internal_v2"):
            function = getattr(lib, name)
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ParentRequestV2)]
            function.restype = ctypes.c_int
        lib.heterosim_ramulator_tick.argtypes = [ctypes.c_void_p]
        lib.heterosim_ramulator_tick.restype = None
        self._advance_until = getattr(
            lib, "heterosim_ramulator_advance_until_event", None
        )
        if self._advance_until is None:
            raise LiveRamulator2Error(
                "Ramulator2 bridge lacks heterosim_ramulator_advance_until_event; "
                "rebuild the P10b-B bridge"
            )
        self._advance_until.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._advance_until.restype = ctypes.c_uint64
        lib.heterosim_ramulator_pop_completed_for_initiator_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(_ParentCompletionV2),
        ]
        lib.heterosim_ramulator_pop_completed_for_initiator_v2.restype = ctypes.c_int
        lib.heterosim_ramulator_finish.argtypes = [ctypes.c_void_p]
        lib.heterosim_ramulator_finish.restype = None
        self._counter_names = (
            "clock",
            "reads",
            "writes",
            "completed",
            "rejected",
            "outstanding",
            "durable_completed",
            "children_sent",
            "children_completed",
            "logical_bytes",
            "internal_bytes",
            "request_payload_bytes",
            "response_payload_bytes",
            "request_wire_bytes",
            "response_wire_bytes",
            "link_cycles",
            "gpu_cycles",
            "gateway_cycles",
            "global_time_fs",
        )
        for counter in self._counter_names:
            function = getattr(lib, f"heterosim_ramulator_{counter}")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_uint64
        for counter in ("parents", "completed", "children"):
            function = getattr(lib, f"heterosim_ramulator_initiator_{counter}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            function.restype = ctypes.c_uint64

    @property
    def current_cycle(self) -> int:
        return int(self._library.heterosim_ramulator_gpu_cycles(self._handle))

    @property
    def global_time_fs(self) -> int:
        return int(self._library.heterosim_ramulator_global_time_fs(self._handle))

    def send(
        self,
        parent_id: int,
        global_address: int,
        size_bytes: int,
        operation: str,
        initiator: int,
        ordering_domain: int,
        sequence_number: int,
    ) -> int:
        if self._closed or parent_id <= 0 or parent_id in self._accepted:
            raise LiveRamulator2Error(f"invalid or duplicate parent {parent_id}")
        if operation not in {"read", "write"}:
            raise LiveRamulator2Error(f"invalid memory operation {operation}")
        request = _ParentRequestV2()
        request.abi_version = self.ABI_VERSION
        request.struct_size = ctypes.sizeof(_ParentRequestV2)
        request.parent_id = parent_id
        request.global_address = global_address
        request.size_bytes = size_bytes
        request.partition_id = 0
        request.operation = 1 if operation == "write" else 0
        request.flags = 0
        request.byte_mask_word_count = 0
        request.sector_mask = 0
        request.ordering_domain = ordering_domain
        request.sequence_number = sequence_number
        request.qos_class = 0
        request.reserved = 0
        request.payload = ctypes.c_void_p(parent_id)
        function = (
            self._library.heterosim_ramulator_send_v2
            if initiator == self.GPU_INITIATOR
            else self._library.heterosim_ramulator_send_internal_v2
            if initiator == self.ATLAS_INITIATOR
            else None
        )
        if function is None:
            raise LiveRamulator2Error(f"invalid initiator {initiator}")
        result = int(function(self._handle, ctypes.byref(request)))
        if result == self.SEND_ACCEPTED:
            self._accepted.add(parent_id)
        elif result == self.SEND_INVALID:
            raise LiveRamulator2Error(f"Ramulator2 rejected invalid parent {parent_id}")
        return result

    def advance_until_event(self, max_cycles: int) -> int:
        if self._closed or max_cycles < 0:
            raise LiveRamulator2Error("invalid Ramulator2 advance")
        if max_cycles == 0:
            return 0
        advanced = int(self._advance_until(self._handle, max_cycles))
        if advanced <= 0:
            raise LiveRamulator2Error(
                "Ramulator2 made no progress while no completion was visible"
            )
        return advanced

    def pop_completions(self) -> list[dict[str, int]]:
        result: list[dict[str, int]] = []
        for initiator in (self.GPU_INITIATOR, self.ATLAS_INITIATOR):
            while True:
                completion = _ParentCompletionV2()
                found = int(
                    self._library.heterosim_ramulator_pop_completed_for_initiator_v2(
                        self._handle, initiator, ctypes.byref(completion)
                    )
                )
                if not found:
                    break
                if (
                    completion.abi_version != self.ABI_VERSION
                    or completion.struct_size != ctypes.sizeof(_ParentCompletionV2)
                    or completion.initiator != initiator
                    or completion.completed_children != completion.total_children
                    or completion.durable != 1
                ):
                    raise LiveRamulator2Error("invalid durable parent completion")
                parent_id = int(completion.parent_id)
                if parent_id not in self._accepted or parent_id in self._completed:
                    raise LiveRamulator2Error(
                        f"unknown or duplicate completion {parent_id}"
                    )
                self._completed.add(parent_id)
                result.append(
                    {
                        "parent_id": parent_id,
                        "initiator": initiator,
                        # Keep the request trace's semantic read/write string;
                        # the ABI enum is reported separately for auditing.
                        "operation_code": int(completion.operation),
                        "total_children": int(completion.total_children),
                        "completion_cycle": self.current_cycle,
                        "completion_time_fs": self.global_time_fs,
                    }
                )
        result.sort(key=lambda item: (item["completion_cycle"], item["parent_id"]))
        return result

    def stats(self) -> dict[str, object]:
        counters = {
            name: int(
                getattr(self._library, f"heterosim_ramulator_{name}")(self._handle)
            )
            for name in self._counter_names
        }
        initiators = {}
        for initiator, label in (
            (self.GPU_INITIATOR, "gpu0"),
            (self.ATLAS_INITIATOR, "atlas0.compute"),
        ):
            initiators[label] = {
                counter: int(
                    getattr(
                        self._library,
                        f"heterosim_ramulator_initiator_{counter}",
                    )(self._handle, initiator)
                )
                for counter in ("parents", "completed", "children")
            }
        return {
            "schema_version": "hetero-live-ramulator2-statistics/v1",
            "instances": 1,
            **counters,
            "initiators": initiators,
            "accepted_parent_ids": len(self._accepted),
            "observed_completion_ids": len(self._completed),
        }

    def close(self) -> dict[str, object]:
        if self._closed:
            raise LiveRamulator2Error("Ramulator2 bridge already closed")
        self._library.heterosim_ramulator_finish(self._handle)
        stats = self.stats()
        self._library.heterosim_ramulator_destroy(self._handle)
        self._closed = True
        if (
            int(stats["outstanding"]) != 0
            or int(stats["accepted_parent_ids"])
            != int(stats["observed_completion_ids"])
        ):
            raise LiveRamulator2Error("Ramulator2 did not close with zero in-flight work")
        return stats
