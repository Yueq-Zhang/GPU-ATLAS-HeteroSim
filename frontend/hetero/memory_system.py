"""Address, residency and request-cycle reference models.

This module keeps stable data identity separate from physical placement.  It is
the control-plane implementation of the frozen address and explicit
non-coherent residency contracts; timing is delegated to the C++ services.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .runtime_bridge import (
    plan_memory_lifecycle,
    run_task_dag,
    simulate_bounded_link,
    simulate_shared_3d_memory,
)


@dataclass(frozen=True, slots=True)
class CanonicalRange:
    value_id: str
    version: int
    offset_bytes: int
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.value_id or self.version < 0 or self.offset_bytes < 0:
            raise ValueError("invalid canonical range")
        if self.size_bytes <= 0:
            raise ValueError("canonical range size must be positive")


@dataclass(frozen=True, slots=True)
class ResidencyRecord:
    value_id: str
    version: int
    memory_space_id: str
    owner_device: str
    state: str
    allocation_epoch: int


class ResidencyManager:
    """Range-independent first-version residency state machine.

    Byte-range synchronization is carried in the event log.  Overlapping range
    conflict detection belongs to the runtime memory planner and is not hidden
    in this state machine.
    """

    def __init__(self) -> None:
        self._records: dict[str, ResidencyRecord] = {}
        self.events: list[dict[str, object]] = []

    def register(
        self,
        value_id: str,
        memory_space_id: str,
        owner_device: str,
        allocation_epoch: int,
        version: int = 0,
    ) -> None:
        if value_id in self._records:
            raise ValueError(f"residency already registered: {value_id}")
        self._records[value_id] = ResidencyRecord(
            value_id,
            version,
            memory_space_id,
            owner_device,
            "exclusive_clean",
            allocation_epoch,
        )

    def transition(
        self,
        value_range: CanonicalRange,
        destination_space: str,
        destination_device: str,
        action: str,
        time_fs: int,
    ) -> ResidencyRecord:
        if action not in {"copy", "migrate", "remote", "synchronize", "write"}:
            raise ValueError(f"unsupported residency action: {action}")
        previous = self._records.get(value_range.value_id)
        if previous is None:
            raise ValueError(f"unknown resident value: {value_range.value_id}")
        if value_range.version < previous.version:
            raise ValueError(f"stale value version for {value_range.value_id}")
        version = previous.version + 1 if action == "write" else value_range.version
        if action in {"remote", "synchronize"}:
            next_space = previous.memory_space_id
            next_owner = previous.owner_device
            next_state = "shared_clean"
        else:
            next_space = destination_space
            next_owner = destination_device
            next_state = "exclusive_dirty" if action == "write" else "exclusive_clean"
        record = ResidencyRecord(
            value_range.value_id,
            version,
            next_space,
            next_owner,
            next_state,
            previous.allocation_epoch + (1 if action == "migrate" else 0),
        )
        self._records[value_range.value_id] = record
        self.events.append(
            {
                "time_fs": time_fs,
                "action": action,
                "range": asdict(value_range),
                "from": asdict(previous),
                "to": asdict(record),
            }
        )
        return record

    def snapshot(self) -> list[dict[str, object]]:
        return [asdict(self._records[key]) for key in sorted(self._records)]


def kv_allocation_size(
    request: Mapping[str, object],
    model: Mapping[str, object],
    address: Mapping[str, object],
) -> int:
    final_tokens = (
        int(request.get("initial_kv_length", 0)) + 1
        if request.get("execution_scope", "full_request") == "decode_step"
        else int(request["prompt_length"]) + int(request["output_length"]) - 1
    )
    page_tokens = int(address["page_tokens"])
    pages = (final_tokens + page_tokens - 1) // page_tokens
    bytes_per_block = (
        page_tokens
        * int(model["num_kv_heads"])
        * int(model["head_dim"])
        * int(model["bytes_per_element"])
    )
    return pages * int(model["num_layers"]) * 2 * bytes_per_block


def build_dynamic_kv_lifecycle(
    requests: Sequence[Mapping[str, object]],
    scheduler_result: Mapping[str, object],
    model: Mapping[str, object],
    address: Mapping[str, object],
    memory_space_id: str,
) -> dict[str, object]:
    """Allocate on first admission, release at retirement, and permit reuse."""

    first_use: dict[str, int] = {}
    for epoch in scheduler_result["epochs"]:  # type: ignore[index]
        boundary = int(epoch["boundary_time_fs"])
        for selection in epoch["selections"]:
            first_use.setdefault(str(selection["request_id"]), boundary)
    finishes = {
        str(item["request_id"]): int(item["finish_time_fs"])
        for item in scheduler_result["requests"]  # type: ignore[index]
    }
    events: list[dict[str, object]] = []
    for request in requests:
        request_id = str(request["request_id"])
        events.append(
            {
                "time_fs": first_use[request_id],
                "operation": "allocate",
                "allocation_id": f"{request_id}.kv",
                "memory_space_id": memory_space_id,
                "size_bytes": kv_allocation_size(request, model, address),
                "alignment_bytes": int(address.get("allocation_alignment_bytes", 64)),
                "lifetime": "request",
            }
        )
        events.append(
            {
                "time_fs": finishes[request_id],
                "operation": "release",
                "allocation_id": f"{request_id}.kv",
            }
        )
    events.sort(
        key=lambda item: (
            int(item["time_fs"]),
            0 if item["operation"] == "release" else 1,
            str(item["allocation_id"]),
        )
    )
    spaces = [
        {
            "memory_space_id": memory_space_id,
            "capacity_bytes": int(address["kv_capacity_bytes"]),
            "base_alignment_bytes": int(address.get("allocation_alignment_bytes", 64)),
        }
    ]
    return plan_memory_lifecycle(spaces, events)


def run_link_transactions(
    route_records: Sequence[Mapping[str, object]],
    timings: Mapping[str, Mapping[str, object]],
    links: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate each physical route with a finite transaction/credit queue."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for index, route in enumerate(route_records):
        route_id = str(route["route_id"])
        if route_id not in links:
            continue
        timing = timings[str(route["task_id"])]
        grouped.setdefault(route_id, []).append(
            {
                "transaction_id": index + 1,
                "parent_task_id": index + 1,
                "source_id": str(route["producer_device"]),
                "destination_id": str(route["consumer_device"]),
                "payload_bytes": int(route.get("payload_bytes", 0)),
                "header_bytes": int(dict(links[route_id]).get("header_bytes", 0)),
                "issue_time_fs": int(timing["start_time_fs"]),
            }
        )
    results: dict[str, object] = {}
    for route_id, transactions in grouped.items():
        link = dict(links[route_id])
        config = {
            "route_id": route_id,
            "wire_bandwidth_Bps": int(link["wire_bandwidth_Bps"]),
            "latency_fs": int(link.get("latency_fs", 0)),
            "queue_depth_transactions": int(
                link.get("queue_depth_transactions", 64)
            ),
            "credits": int(link.get("credits", link.get("queue_depth_transactions", 64))),
            "full_duplex": bool(link.get("full_duplex", True)),
        }
        results[route_id] = simulate_bounded_link(config, transactions)
    return {
        "schema_version": "hetero-link-bundle/v1",
        "routes": results,
        "submitted_payload_bytes": sum(
            int(value["payload_bytes"]) for value in results.values()  # type: ignore[union-attr]
        ),
        "completed_payload_bytes": sum(
            int(value["payload_bytes"]) for value in results.values()  # type: ignore[union-attr]
        ),
    }


def run_shared_memory_reference(
    task_records: Sequence[Mapping[str, object]],
    timings: Mapping[str, Mapping[str, object]],
    memory_config: Mapping[str, object],
    memory_space_id: str,
) -> dict[str, object]:
    """Convert bulk task traffic into shared memory transactions.

    This is the built-in request-cycle reference path.  It exercises address
    decode, arbitration, queue limits and conservation, but remains distinct
    from the externally qualified Accel-Sim/Ramulator2 callback bridge.
    """

    access_mode = str(memory_config.get("access_mode", "shared_gpu_atlas"))
    if access_mode not in {"gpu_only", "shared_gpu_atlas"}:
        raise ValueError(f"unsupported shared 3D memory access mode {access_mode!r}")
    allowed_initiators = list(
        memory_config.get("initiator_order", ["gpu0", "atlas0.compute"])
    )
    if access_mode == "gpu_only" and allowed_initiators != ["gpu0"]:
        raise ValueError(
            "gpu_only shared 3D memory requires initiator_order=['gpu0']"
        )

    requests: list[dict[str, object]] = []
    next_offset = 0
    request_id = 1
    for task_index, task in enumerate(task_records, 1):
        cost = task.get("analytical_cost")
        if not isinstance(cost, Mapping):
            continue
        timing = timings[str(task["task_id"])]
        device_id = str(task["device_id"])
        if device_id not in {"gpu0", "atlas0.compute"}:
            raise ValueError(f"unsupported shared 3D memory initiator {device_id!r}")
        initiator = device_id
        if initiator not in allowed_initiators:
            raise ValueError(
                f"shared 3D memory access_mode={access_mode} rejects initiator "
                f"{initiator!r}"
            )
        for operation, field in (("read", "read_bytes"), ("write", "write_bytes")):
            size = int(cost.get(field, 0))
            if size <= 0:
                continue
            requests.append(
                {
                    "request_id": request_id,
                    "parent_task_id": task_index,
                    "initiator_id": initiator,
                    "offset_bytes": next_offset,
                    "allocation_epoch": 1,
                    "value_id": str(task["template_node_id"]),
                    "value_version": 0,
                    "size_bytes": size,
                    "operation": operation,
                    "issue_time_fs": int(timing["start_time_fs"]),
                    "ordering_domain": task_index,
                    "sequence_number": request_id,
                    "qos_class": 0,
                }
            )
            request_id += 1
            next_offset += size
    config = dict(memory_config)
    config["memory_space_id"] = memory_space_id
    config.setdefault("initiator_order", ["gpu0", "atlas0.compute"])
    result = simulate_shared_3d_memory(config, requests)
    requests_by_initiator = result.get("requests_by_initiator", {})
    if not isinstance(requests_by_initiator, Mapping):
        raise RuntimeError("shared memory result lacks requests_by_initiator")
    logic_die_requests = int(requests_by_initiator.get("atlas0.compute", 0))
    if access_mode == "gpu_only" and logic_die_requests != 0:
        raise RuntimeError("gpu_only shared memory completed Logic Die requests")
    result["access_mode"] = access_mode
    result["gpu_logic_die_competition"] = {
        "enabled": access_mode == "shared_gpu_atlas",
        "gpu_requests": int(requests_by_initiator.get("gpu0", 0)),
        "logic_die_requests": logic_die_requests,
    }
    return result


def run_reference_coupled_dag(
    runtime_tasks: Sequence[Mapping[str, object]],
    execution_tasks: Sequence[Mapping[str, object]],
    route_records: Sequence[Mapping[str, object]],
    links: Mapping[str, object],
    memory_config: Mapping[str, object] | None,
    memory_space_id: str,
    max_iterations: int = 64,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Close the reference timing loop between tasks, links and memory.

    Each iteration runs the C++ DAG scheduler, submits transactions at the
    resulting task start times, and extends parent task durations to the latest
    response.  Durations only grow, so convergence is deterministic and cannot
    hide a response that should stall a dependent task.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    current = [dict(task) for task in runtime_tasks]
    base_duration = {
        str(task["task_id"]): int(task["duration_fs"]) for task in current
    }
    route_task_by_parent = {
        index: str(route["task_id"])
        for index, route in enumerate(route_records, 1)
    }
    device_task_by_parent = {
        index: str(task["task_id"])
        for index, task in enumerate(execution_tasks, 1)
    }
    final_runtime: dict[str, object] | None = None
    final_links: dict[str, object] | None = None
    final_memory: dict[str, object] | None = None
    for iteration in range(1, max_iterations + 1):
        runtime = run_task_dag(current)
        timings = {
            str(item["task_id"]): item
            for item in runtime["tasks"]  # type: ignore[index]
        }
        link_statistics = run_link_transactions(route_records, timings, links)
        if memory_config is not None:
            memory_statistics = run_shared_memory_reference(
                execution_tasks, timings, memory_config, memory_space_id
            )
        else:
            memory_statistics = {
                "schema_version": "hetero-memory-statistics/v1",
                "status": "external_or_separate_memory_service_not_exercised",
                "memory_space_id": memory_space_id,
                "parent_responses": [],
            }

        response_completion: dict[str, int] = {}
        for result in link_statistics["routes"].values():  # type: ignore[union-attr]
            for response in result["responses"]:
                task_id = route_task_by_parent[int(response["parent_task_id"])]
                response_completion[task_id] = max(
                    response_completion.get(task_id, 0),
                    int(response["completion_time_fs"]),
                )
        for response in memory_statistics.get("parent_responses", []):
            task_id = device_task_by_parent[int(response["parent_task_id"])]
            response_completion[task_id] = max(
                response_completion.get(task_id, 0),
                int(response["completion_time_fs"]),
            )

        changed = False
        for task in current:
            task_id = str(task["task_id"])
            timing = timings[task_id]
            completion = response_completion.get(task_id)
            requested_duration = base_duration[task_id]
            if completion is not None:
                requested_duration = max(
                    requested_duration,
                    completion - int(timing["start_time_fs"]),
                )
            next_duration = max(int(task["duration_fs"]), requested_duration)
            if next_duration != int(task["duration_fs"]):
                task["duration_fs"] = next_duration
                changed = True

        final_runtime = runtime
        final_links = link_statistics
        final_memory = memory_statistics
        if not changed:
            final_runtime["coupling_iterations"] = iteration
            final_links["coupling_iterations"] = iteration
            final_memory["coupling_iterations"] = iteration
            return final_runtime, final_links, final_memory
    raise RuntimeError(
        f"reference task/link/memory coupling did not converge in {max_iterations} iterations"
    )
