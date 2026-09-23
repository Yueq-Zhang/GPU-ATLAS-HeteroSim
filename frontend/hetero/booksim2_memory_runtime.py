"""Cycle-stepped BookSim2 + live Ramulator2 request/response orchestration."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .booksim2_adapter import BookSim2CycleAdapter
from .live_ramulator2 import LiveRamulator2Bridge


class BookSim2MemoryRuntimeError(RuntimeError):
    """Raised when NoC, DRAM, QoS or request conservation fails."""


class _NoC(Protocol):
    def submit(self, flow_id: int, source_node: int, destination_node: int, flit_count: int) -> None: ...
    def step(self) -> None: ...
    def pop_completions(self) -> list[dict[str, int]]: ...
    def stats(self) -> dict[str, object]: ...
    def close(self) -> dict[str, object]: ...


class _DRAM(Protocol):
    GPU_INITIATOR: int
    ATLAS_INITIATOR: int
    SEND_RETRY: int
    SEND_ACCEPTED: int

    def send_from_noc(
        self,
        parent_id: int,
        global_address: int,
        size_bytes: int,
        operation: str,
        initiator: int,
        ordering_domain: int,
        sequence_number: int,
    ) -> int: ...
    def tick(self) -> None: ...
    def pop_completions(self) -> list[dict[str, int]]: ...
    def stats(self) -> dict[str, object]: ...
    def close(self) -> dict[str, object]: ...


NoCFactory = Callable[[Path, Path, int], _NoC]
DRAMFactory = Callable[[Path, Mapping[str, object]], _DRAM]


def _resolve(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise BookSim2MemoryRuntimeError("required runtime path is missing")
    path = Path(raw)
    return path if path.is_absolute() else (root / path).resolve()


def _flits(payload_bytes: int, header_bytes: int, flit_bytes: int) -> int:
    return max(1, math.ceil((payload_bytes + header_bytes) / flit_bytes))


def _validate(
    requests: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> tuple[dict[str, int], dict[str, int]]:
    nodes_raw = config.get("nodes")
    policy = config.get("qos_policy")
    if not isinstance(nodes_raw, Mapping) or not isinstance(policy, Mapping):
        raise BookSim2MemoryRuntimeError("nodes and qos_policy are required")
    nodes = {str(key): int(value) for key, value in nodes_raw.items()}
    if set(nodes) != {"gpu0", "atlas0.compute", "gateway0", "dram0"}:
        raise BookSim2MemoryRuntimeError("exact GPU/ATLAS/Gateway/DRAM nodes are required")
    if len(set(nodes.values())) != 4 or min(nodes.values()) < 0:
        raise BookSim2MemoryRuntimeError("BookSim2 node IDs must be unique and unsigned")
    weights_raw = policy.get("class_weights")
    if not isinstance(weights_raw, Mapping) or not weights_raw:
        raise BookSim2MemoryRuntimeError("positive QoS class weights are required")
    weights = {str(key): int(value) for key, value in weights_raw.items()}
    if any(value <= 0 for value in weights.values()):
        raise BookSim2MemoryRuntimeError("QoS weights must be positive")
    identifiers: set[int] = set()
    for request in requests:
        parent_id = int(request.get("parent_id", 0))
        initiator = str(request.get("initiator", ""))
        qos_class = str(request.get("qos_class", ""))
        operation = str(request.get("operation", ""))
        if parent_id <= 0 or parent_id in identifiers:
            raise BookSim2MemoryRuntimeError("parent IDs must be positive and unique")
        identifiers.add(parent_id)
        if initiator not in {"gpu0", "atlas0.compute"}:
            raise BookSim2MemoryRuntimeError("unknown memory initiator")
        if qos_class not in weights or operation not in {"read", "write"}:
            raise BookSim2MemoryRuntimeError("invalid QoS class or memory operation")
        if int(request.get("size_bytes", 0)) <= 0 or int(request.get("global_address", -1)) < 0:
            raise BookSim2MemoryRuntimeError("invalid memory request range")
    return nodes, weights


def run_booksim2_ramulator2_qos(
    project_root: Path,
    requests: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    noc_factory: NoCFactory = BookSim2CycleAdapter,
    dram_factory: DRAMFactory = LiveRamulator2Bridge,
) -> dict[str, object]:
    """Run real NoC packets around a single live Ramulator2 timing owner.

    Each parent traverses four BookSim2 packet segments:
    initiator->gateway, gateway->DRAM, DRAM->gateway and gateway->initiator.
    Ramulator2 accepts the parent only after the request reaches the DRAM node;
    the response is released only after the durable parent completion.
    """

    root = project_root.resolve()
    nodes, weights = _validate(requests, config)
    library = _resolve(root, config["adapter_library"])
    network = _resolve(root, config["network_config"])
    dram_config = config.get("ramulator2")
    if not isinstance(dram_config, Mapping):
        raise BookSim2MemoryRuntimeError("ramulator2 configuration is required")
    flit_bytes = int(config.get("flit_bytes", 32))
    request_header = int(config.get("request_header_bytes", 16))
    response_header = int(config.get("response_header_bytes", 16))
    max_cycles = int(config.get("max_cycles", 1_000_000))
    starvation_limit = int(
        dict(config["qos_policy"]).get("max_starvation_cycles", 64)  # type: ignore[arg-type]
    )
    issue_width = int(config.get("ramulator_issue_width", 1))
    if min(flit_bytes, max_cycles, starvation_limit, issue_width) <= 0:
        raise BookSim2MemoryRuntimeError("runtime limits and widths must be positive")

    noc = noc_factory(library, network, max(nodes.values()) + 1)
    dram = dram_factory(root, dram_config)
    request_by_parent = {int(item["parent_id"]): dict(item) for item in requests}
    state = {
        parent_id: {
            "state": "WAITING_ISSUE",
            "issue_cycle": int(request.get("issue_cycle", 0)),
            "durable_cycle": None,
            "completion_cycle": None,
        }
        for parent_id, request in request_by_parent.items()
    }
    pending_packets: list[dict[str, object]] = []
    ramulator_pending: list[int] = []
    packet_ledger: list[dict[str, object]] = []
    request_events: list[dict[str, object]] = []
    flow_records: dict[int, dict[str, object]] = {}
    next_flow_id = 1
    virtual_runtime: dict[int, float] = defaultdict(float)
    last_service: dict[int, int] = {}
    max_wait = 0

    def enqueue_segment(parent_id: int, segment: str, ready_cycle: int) -> None:
        nonlocal next_flow_id
        request = request_by_parent[parent_id]
        initiator = str(request["initiator"])
        size = int(request["size_bytes"])
        operation = str(request["operation"])
        if segment == "initiator_to_gateway":
            source, destination = nodes[initiator], nodes["gateway0"]
            payload = size if operation == "write" else 0
            header = request_header
        elif segment == "gateway_to_dram":
            source, destination = nodes["gateway0"], nodes["dram0"]
            payload = size if operation == "write" else 0
            header = request_header
        elif segment == "dram_to_gateway":
            source, destination = nodes["dram0"], nodes["gateway0"]
            payload = size if operation == "read" else 0
            header = response_header
        elif segment == "gateway_to_initiator":
            source, destination = nodes["gateway0"], nodes[initiator]
            payload = size if operation == "read" else 0
            header = response_header
        else:
            raise BookSim2MemoryRuntimeError("unknown packet segment")
        packet = {
            "flow_id": next_flow_id,
            "parent_id": parent_id,
            "segment": segment,
            "source_node": source,
            "destination_node": destination,
            "flit_count": _flits(payload, header, flit_bytes),
            "qos_class": str(request["qos_class"]),
            "ready_cycle": ready_cycle,
            "injected_cycle": None,
        }
        pending_packets.append(packet)
        flow_records[next_flow_id] = packet
        next_flow_id += 1

    for parent_id, request in sorted(request_by_parent.items()):
        enqueue_segment(parent_id, "initiator_to_gateway", int(request.get("issue_cycle", 0)))
        request_events.append(
            {"cycle": int(request.get("issue_cycle", 0)), "parent_id": parent_id, "event": "ISSUE_READY"}
        )

    completed_parents: set[int] = set()
    for cycle in range(max_cycles):
        ready_by_source: dict[int, list[dict[str, object]]] = defaultdict(list)
        for packet in pending_packets:
            if int(packet["ready_cycle"]) <= cycle:
                ready_by_source[int(packet["source_node"])].append(packet)
        for source in sorted(ready_by_source):
            candidates = ready_by_source[source]
            for packet in candidates:
                flow_id = int(packet["flow_id"])
                wait = cycle - int(packet["ready_cycle"])
                if flow_id in last_service:
                    wait = cycle - last_service[flow_id] - 1
                max_wait = max(max_wait, wait)
            starving = [
                packet
                for packet in candidates
                if cycle - int(packet["ready_cycle"]) >= starvation_limit
            ]
            pool = starving or candidates
            selected = min(
                pool,
                key=lambda packet: (
                    virtual_runtime[int(packet["parent_id"])]
                    + 1.0 / weights[str(packet["qos_class"])],
                    int(packet["ready_cycle"]),
                    int(packet["flow_id"]),
                ),
            )
            flow_id = int(selected["flow_id"])
            parent_id = int(selected["parent_id"])
            noc.submit(
                flow_id,
                int(selected["source_node"]),
                int(selected["destination_node"]),
                int(selected["flit_count"]),
            )
            selected["injected_cycle"] = cycle
            pending_packets.remove(selected)
            last_service[flow_id] = cycle
            virtual_runtime[parent_id] += 1.0 / weights[str(selected["qos_class"])]
            packet_ledger.append(dict(selected))

        noc.step()
        for completion in noc.pop_completions():
            flow_id = int(completion["flow_id"])
            packet = flow_records[flow_id]
            packet["completion_cycle"] = int(completion["completion_cycle"])
            parent_id = int(packet["parent_id"])
            segment = str(packet["segment"])
            request_events.append(
                {
                    "cycle": int(completion["completion_cycle"]),
                    "parent_id": parent_id,
                    "event": "NOC_SEGMENT_COMPLETE",
                    "segment": segment,
                    "flow_id": flow_id,
                }
            )
            if segment == "initiator_to_gateway":
                enqueue_segment(parent_id, "gateway_to_dram", cycle + 1)
                state[parent_id]["state"] = "AT_GATEWAY"
            elif segment == "gateway_to_dram":
                ramulator_pending.append(parent_id)
                state[parent_id]["state"] = "WAITING_DRAM_ACCEPT"
            elif segment == "dram_to_gateway":
                enqueue_segment(parent_id, "gateway_to_initiator", cycle + 1)
                state[parent_id]["state"] = "DURABLE_AT_GATEWAY"
            elif segment == "gateway_to_initiator":
                state[parent_id]["state"] = "COMPLETED"
                state[parent_id]["completion_cycle"] = cycle
                completed_parents.add(parent_id)

        accepted_now: list[int] = []
        for parent_id in list(ramulator_pending)[:issue_width]:
            request = request_by_parent[parent_id]
            initiator = (
                dram.GPU_INITIATOR
                if request["initiator"] == "gpu0"
                else dram.ATLAS_INITIATOR
            )
            send_result = dram.send_from_noc(
                parent_id,
                int(request["global_address"]),
                int(request["size_bytes"]),
                str(request["operation"]),
                initiator,
                int(request.get("ordering_domain", initiator)),
                int(request.get("sequence_number", parent_id)),
            )
            if send_result == dram.SEND_ACCEPTED:
                accepted_now.append(parent_id)
                state[parent_id]["state"] = "IN_DRAM"
                request_events.append(
                    {"cycle": cycle, "parent_id": parent_id, "event": "RAMULATOR_ACCEPT"}
                )
            elif send_result != dram.SEND_RETRY:
                raise BookSim2MemoryRuntimeError("Ramulator2 returned an invalid send code")
        for parent_id in accepted_now:
            ramulator_pending.remove(parent_id)

        dram.tick()
        for completion in dram.pop_completions():
            parent_id = int(completion["parent_id"])
            if state[parent_id]["state"] != "IN_DRAM":
                raise BookSim2MemoryRuntimeError("unexpected durable DRAM completion")
            state[parent_id]["state"] = "DURABLE_IN_DRAM"
            state[parent_id]["durable_cycle"] = cycle
            request_events.append(
                {"cycle": cycle, "parent_id": parent_id, "event": "RAMULATOR_DURABLE"}
            )
            enqueue_segment(parent_id, "dram_to_gateway", cycle + 1)

        noc_stats = noc.stats()
        dram_stats = dram.stats()
        if (
            len(completed_parents) == len(requests)
            and not pending_packets
            and not ramulator_pending
            and bool(noc_stats["zero_in_flight"])
            and int(dram_stats["outstanding"]) == 0
        ):
            break
    else:
        raise BookSim2MemoryRuntimeError("BookSim2/Ramulator2 QoS runtime timed out")

    final_noc = noc.close()
    final_dram = dram.close()
    if (
        int(final_noc["accepted_packets"]) != len(requests) * 4
        or final_noc["accepted_packets"] != final_noc["delivered_packets"]
        or final_noc["accepted_flits"] != final_noc["delivered_flits"]
        or int(final_noc["outstanding_credits"]) != 0
        or int(final_dram["accepted_parent_ids"]) != len(requests)
        or final_dram["accepted_parent_ids"] != final_dram["observed_completion_ids"]
        or int(final_dram["outstanding"]) != 0
    ):
        raise BookSim2MemoryRuntimeError("packet, flit, credit or request conservation failed")
    return {
        "schema_version": "hetero-booksim2-ramulator2-qos-runtime/v1",
        "timing_owner": "global_cycle_booksim2_plus_single_ramulator2",
        "performance_claim_allowed": False,
        "nodes": nodes,
        "requests": [
            {"parent_id": parent_id, **state[parent_id], **request_by_parent[parent_id]}
            for parent_id in sorted(request_by_parent)
        ],
        "packet_ledger": sorted(packet_ledger, key=lambda item: int(item["flow_id"])),
        "request_events": sorted(
            request_events,
            key=lambda item: (int(item["cycle"]), int(item["parent_id"]), str(item["event"])),
        ),
        "booksim2": final_noc,
        "ramulator2": final_dram,
        "qos": {
            "class_weights": weights,
            "max_starvation_cycles": starvation_limit,
            "max_observed_packet_wait_cycles": max_wait,
            "virtual_runtime": {str(key): value for key, value in sorted(virtual_runtime.items())},
        },
        "conservation": {
            "parent_requests": len(requests),
            "durable_parent_completions": len(completed_parents),
            "packets": int(final_noc["accepted_packets"]),
            "packet_completions": int(final_noc["delivered_packets"]),
            "flits": int(final_noc["accepted_flits"]),
            "flit_completions": int(final_noc["delivered_flits"]),
            "outstanding_credits": int(final_noc["outstanding_credits"]),
            "zero_in_flight": True,
        },
        "qualification_boundary": (
            "BookSim2 packet/flit/credit and one live Ramulator2 request path are active. "
            "This is a functional cycle-coupling qualification; clock-domain and hardware "
            "performance calibration remain closed."
        ),
    }
