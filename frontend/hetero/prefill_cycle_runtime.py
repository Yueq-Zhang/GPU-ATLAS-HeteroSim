"""P10b-B live request-cycle runtime for strict Prefill execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .global_memory_map import (
    GlobalAllocation,
    sampled_requests_for_values,
)
from .live_ramulator2 import LiveRamulator2Bridge, LiveRamulator2Error
from .online_operator_runtime import OnlineDispatchSpec
from .prefill_cycle_artifact import (
    CycleTaskPlan,
    PrefillCycleDispatcher,
)


class PrefillCycleRuntimeError(RuntimeError):
    """Raised when the live plan violates dependency, version or memory rules."""


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise PrefillCycleRuntimeError("time conversion operands are invalid")
    return (numerator + denominator - 1) // denominator


@dataclass(slots=True)
class _Active:
    task_id: str
    record: dict[str, object]
    kind: str
    resource_id: str
    start_cycle: int
    start_time_fs: int
    compute_cycles: int
    compute_done_cycle: int
    plan: CycleTaskPlan | None
    pending_reads: list[dict[str, object]] = field(default_factory=list)
    pending_writes: list[dict[str, object]] = field(default_factory=list)
    outstanding: set[int] = field(default_factory=set)
    phase: str = "reading"
    writes_released: bool = False
    accepted_parents: int = 0
    completed_parents: int = 0
    represented_read_bytes: int = 0
    represented_write_bytes: int = 0
    simulated_request_bytes: int = 0


def _records(execution_graph: Mapping[str, object]) -> list[dict[str, object]]:
    tasks = execution_graph.get("tasks")
    routes = execution_graph.get("routes")
    if not isinstance(tasks, list) or not isinstance(routes, list):
        raise PrefillCycleRuntimeError("execution graph requires task and route arrays")
    records: list[dict[str, object]] = []
    for raw in [*tasks, *routes]:
        if not isinstance(raw, dict):
            raise PrefillCycleRuntimeError("runtime records must be mutable objects")
        records.append(raw)
    return records


def run_prefill_cycle_dag(
    execution_graph: Mapping[str, object],
    dispatch_specs: Mapping[str, OnlineDispatchSpec],
    dispatcher: PrefillCycleDispatcher,
    bridge: LiveRamulator2Bridge,
    allocations: Mapping[str, GlobalAllocation],
    *,
    global_clock_hz: int,
    transaction_bytes: int,
    max_samples_per_value: int,
) -> dict[str, object]:
    """Run device tasks, routes and one live Ramulator2 on one cycle timeline."""

    records = _records(execution_graph)
    by_id: dict[str, dict[str, object]] = {}
    device_task_ids: set[str] = set()
    for record in records:
        task_id = str(record.get("task_id", ""))
        resource_id = str(record.get("resource_id", ""))
        if not task_id or not resource_id or task_id in by_id:
            raise PrefillCycleRuntimeError(f"invalid runtime identity {task_id!r}")
        by_id[task_id] = record
        if record.get("task_kind") == "device":
            device_task_ids.add(task_id)
    if device_task_ids != set(dispatch_specs):
        raise PrefillCycleRuntimeError(
            "dispatch specs must equal device tasks: "
            f"missing={sorted(device_task_ids - set(dispatch_specs))}, "
            f"extra={sorted(set(dispatch_specs) - device_task_ids)}"
        )

    remaining: dict[str, int] = {}
    dependents: dict[str, list[str]] = {task_id: [] for task_id in by_id}
    dependency_ready_cycle: dict[str, int] = {}
    for task_id, record in by_id.items():
        dependencies = record.get("dependencies", [])
        if not isinstance(dependencies, Sequence) or isinstance(
            dependencies, (str, bytes)
        ):
            raise PrefillCycleRuntimeError(f"dependencies must be an array: {task_id}")
        dependency_ids = [str(value) for value in dependencies]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise PrefillCycleRuntimeError(f"duplicate dependency for {task_id}")
        for dependency in dependency_ids:
            if dependency not in by_id or dependency == task_id:
                raise PrefillCycleRuntimeError(
                    f"invalid dependency {dependency} for {task_id}"
                )
            dependents[dependency].append(task_id)
        remaining[task_id] = len(dependency_ids)
        release_fs = int(record.get("release_time_fs", 0))
        dependency_ready_cycle[task_id] = _ceil_div(
            release_fs * global_clock_hz, 10**15
        )

    residency_plan = execution_graph.get("residency_plan")
    if not isinstance(residency_plan, Mapping) or not isinstance(
        residency_plan.get("events"), list
    ):
        raise PrefillCycleRuntimeError("residency plan events are required")
    external_by_task: dict[str, list[Mapping[str, object]]] = {}
    for raw in residency_plan["events"]:
        if not isinstance(raw, Mapping):
            raise PrefillCycleRuntimeError("residency event must be an object")
        if raw.get("event") == "register_external_input":
            external_by_task.setdefault(str(raw["trigger_task_id"]), []).append(raw)

    latest_version: dict[str, int] = {}
    available_version: dict[tuple[str, str], int] = {}
    version_checks = 0

    def register_external_inputs(task_id: str) -> None:
        for event in external_by_task.get(task_id, []):
            value_id = str(event["value_id"])
            version = int(event["version"])
            device_id = str(event["device_id"])
            existing = latest_version.get(value_id)
            if existing is not None and existing != version:
                raise PrefillCycleRuntimeError(
                    f"external input {value_id} conflicts with v{existing}"
                )
            latest_version[value_id] = version
            available_version[(value_id, device_id)] = version

    def require_value(value_id: str, version: int, device_id: str, task_id: str) -> None:
        nonlocal version_checks
        version_checks += 1
        latest = latest_version.get(value_id)
        available = available_version.get((value_id, device_id))
        if latest != version or available != version:
            raise PrefillCycleRuntimeError(
                f"stale/unavailable input for {task_id}: {value_id}@v{version} "
                f"on {device_id}, latest={latest}, available={available}"
            )

    ready: set[str] = {
        task_id for task_id, count in remaining.items() if count == 0
    }
    active: dict[str, _Active] = {}
    resource_owner: dict[str, str] = {}
    completed: set[str] = set()
    timing: dict[str, dict[str, object]] = {}
    parent_owner: dict[int, str] = {}
    parent_metadata: dict[int, dict[str, object]] = {}
    memory_trace: list[dict[str, object]] = []
    launch_log: list[dict[str, object]] = []
    next_parent_id = 1
    backend_dispatch_count = 0

    def make_device_active(task_id: str, current_cycle: int) -> _Active:
        nonlocal backend_dispatch_count
        record = by_id[task_id]
        register_external_inputs(task_id)
        device_id = str(record["device_id"])
        inputs = record.get("input_values", [])
        outputs = record.get("output_values", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise PrefillCycleRuntimeError("device values must be arrays")
        checked: list[dict[str, object]] = []
        for value in inputs:
            if not isinstance(value, Mapping):
                raise PrefillCycleRuntimeError("input value must be an object")
            value_id = str(value["value_id"])
            version = int(value["version"])
            require_value(value_id, version, device_id, task_id)
            checked.append({"value_id": value_id, "version": version})
        plan = dispatcher.dispatch(dispatch_specs[task_id])
        reads = sampled_requests_for_values(
            task_id,
            device_id,
            inputs,
            allocations,
            "read",
            transaction_bytes,
            max_samples_per_value,
        )
        writes = sampled_requests_for_values(
            task_id,
            device_id,
            outputs,
            allocations,
            "write",
            transaction_bytes,
            max_samples_per_value,
        )
        record.update(
            {
                "backend_id": plan.backend_id,
                "backend_launch_cycle": current_cycle,
                "backend_launch_time_fs": bridge.global_time_fs,
                "native_compute_cycles": plan.native_compute_cycles,
                "global_compute_cycles": plan.global_compute_cycles,
                "cycle_formula": dict(plan.formula),
                "compiled_artifact": dict(plan.artifact),
                "fidelity": dict(plan.fidelity),
                "validated_input_versions": checked,
            }
        )
        backend_dispatch_count += 1
        launch_log.append(
            {
                "task_id": task_id,
                "kind": "backend",
                "device_id": device_id,
                "launch_cycle": current_cycle,
                "launch_time_fs": bridge.global_time_fs,
                "validated_input_versions": checked,
            }
        )
        return _Active(
            task_id=task_id,
            record=record,
            kind="device",
            resource_id=str(record["resource_id"]),
            start_cycle=current_cycle,
            start_time_fs=bridge.global_time_fs,
            compute_cycles=plan.global_compute_cycles,
            compute_done_cycle=(
                current_cycle + plan.global_compute_cycles if not reads else -1
            ),
            plan=plan,
            pending_reads=reads,
            pending_writes=writes,
            phase="computing" if not reads else "reading",
            represented_read_bytes=sum(int(item["represented_bytes"]) for item in reads),
            represented_write_bytes=sum(int(item["represented_bytes"]) for item in writes),
        )

    def make_route_active(task_id: str, current_cycle: int) -> _Active:
        record = by_id[task_id]
        value_id = str(record["value_id"])
        version = int(record["value_version"])
        producer = str(record["producer_device"])
        consumer = str(record["consumer_device"])
        require_value(value_id, version, producer, task_id)
        duration_fs = int(record.get("duration_fs", 1))
        route_cycles = max(1, _ceil_div(duration_fs * global_clock_hz, 10**15))
        allocation = allocations.get(value_id)
        if allocation is None:
            raise PrefillCycleRuntimeError(f"route has no allocation for {value_id}")
        # A Model-3 explicit-noncoherent route waits for producer durability,
        # then issues one consumer-side acquire probe through the live port.
        probe_size = min(transaction_bytes, allocation.size_bytes)
        probe = {
            "task_id": task_id,
            "device_id": consumer,
            "value_id": value_id,
            "version": version,
            "operation": "read",
            "global_address": allocation.base_address,
            "size_bytes": probe_size,
            "represented_bytes": 0,
            "sample_index": 0,
            "sample_count": 1,
            "logical_value_bytes": int(record["payload_bytes"]),
            "coherence_probe": True,
        }
        launch_log.append(
            {
                "task_id": task_id,
                "kind": "route",
                "producer_device": producer,
                "consumer_device": consumer,
                "launch_cycle": current_cycle,
                "launch_time_fs": bridge.global_time_fs,
                "value_id": value_id,
                "version": version,
                "actions": list(record.get("actions", [])),
            }
        )
        return _Active(
            task_id=task_id,
            record=record,
            kind="route",
            resource_id=str(record["resource_id"]),
            start_cycle=current_cycle,
            start_time_fs=bridge.global_time_fs,
            compute_cycles=route_cycles,
            compute_done_cycle=-1,
            plan=None,
            pending_reads=[probe],
        )

    def submit_pending(item: _Active) -> bool:
        nonlocal next_parent_id
        pending = (
            item.pending_reads
            if item.phase == "reading"
            else item.pending_writes
            if item.phase == "writing"
            else []
        )
        if not pending:
            return False
        sample = pending[0]
        device_id = str(sample["device_id"])
        initiator = (
            LiveRamulator2Bridge.GPU_INITIATOR
            if device_id == "gpu0"
            else LiveRamulator2Bridge.ATLAS_INITIATOR
            if device_id == "atlas0.compute"
            else -1
        )
        if initiator < 0:
            raise PrefillCycleRuntimeError(f"unsupported live initiator {device_id}")
        result = bridge.send(
            next_parent_id,
            int(sample["global_address"]),
            int(sample["size_bytes"]),
            str(sample["operation"]),
            initiator,
            list(by_id).index(item.task_id) + 1,
            next_parent_id,
        )
        if result == LiveRamulator2Bridge.SEND_RETRY:
            return False
        parent_id = next_parent_id
        next_parent_id += 1
        pending.pop(0)
        item.outstanding.add(parent_id)
        item.accepted_parents += 1
        item.simulated_request_bytes += int(sample["size_bytes"])
        parent_owner[parent_id] = item.task_id
        parent_metadata[parent_id] = dict(sample)
        memory_trace.append(
            {
                "parent_id": parent_id,
                "issue_cycle": bridge.current_cycle,
                "issue_time_fs": bridge.global_time_fs,
                **dict(sample),
            }
        )
        return True

    def commit_task(item: _Active, current_cycle: int) -> None:
        record = item.record
        if item.kind == "device":
            device_id = str(record["device_id"])
            outputs = record.get("output_values", [])
            assert isinstance(outputs, list)
            for value in outputs:
                assert isinstance(value, Mapping)
                value_id = str(value["value_id"])
                version = int(value["version"])
                previous = latest_version.get(value_id)
                expected = 1 if previous is None else previous + 1
                if version != expected:
                    raise PrefillCycleRuntimeError(
                        f"non-monotonic write {value_id}@v{version}, expected v{expected}"
                    )
                latest_version[value_id] = version
                for key in [key for key in available_version if key[0] == value_id]:
                    del available_version[key]
                available_version[(value_id, device_id)] = version
            record["memory_request_summary"] = {
                "parents": item.accepted_parents,
                "completed_parents": item.completed_parents,
                "represented_read_bytes": item.represented_read_bytes,
                "represented_write_bytes": item.represented_write_bytes,
                "simulated_request_bytes": item.simulated_request_bytes,
                "sampling_policy": "evenly_spaced_bounded",
            }
        else:
            value_id = str(record["value_id"])
            version = int(record["value_version"])
            if latest_version.get(value_id) != version:
                raise PrefillCycleRuntimeError(
                    f"route completed with stale version: {item.task_id}"
                )
            destination = str(record["consumer_device"])
            available_version[(value_id, destination)] = version
            if str(getattr(record.get("kind"), "value", record.get("kind"))) == "migration":
                available_version.pop((value_id, str(record["producer_device"])), None)
            record["memory_request_summary"] = {
                "parents": item.accepted_parents,
                "completed_parents": item.completed_parents,
                "coherence_probe": True,
            }
        completion_time_fs = bridge.global_time_fs
        timing[item.task_id] = {
            "task_id": item.task_id,
            "resource_id": item.resource_id,
            "ready_cycle": dependency_ready_cycle[item.task_id],
            "start_cycle": item.start_cycle,
            "completion_cycle": current_cycle,
            "start_time_fs": item.start_time_fs,
            "completion_time_fs": completion_time_fs,
        }
        resource_owner.pop(item.resource_id, None)
        active.pop(item.task_id)
        completed.add(item.task_id)
        for dependent in sorted(dependents[item.task_id]):
            remaining[dependent] -= 1
            dependency_ready_cycle[dependent] = max(
                dependency_ready_cycle[dependent], current_cycle
            )
            if remaining[dependent] < 0:
                raise PrefillCycleRuntimeError("dependency counter underflow")
            if remaining[dependent] == 0:
                ready.add(dependent)

    try:
        while len(completed) != len(by_id):
            current_cycle = bridge.current_cycle

            for completion in bridge.pop_completions():
                parent_id = int(completion["parent_id"])
                owner_id = parent_owner.get(parent_id)
                if owner_id is None or owner_id not in active:
                    raise PrefillCycleRuntimeError(
                        f"completion {parent_id} has no active owner"
                    )
                owner = active[owner_id]
                if parent_id not in owner.outstanding:
                    raise PrefillCycleRuntimeError(f"duplicate completion {parent_id}")
                owner.outstanding.remove(parent_id)
                owner.completed_parents += 1
                metadata = parent_metadata[parent_id]
                for trace in reversed(memory_trace):
                    if trace["parent_id"] == parent_id:
                        trace.update(completion)
                        break
                else:
                    raise PrefillCycleRuntimeError("memory trace lost parent metadata")

            # An operator follows a strict read -> compute -> write lifecycle.
            # Compute does not begin until every sampled input request is durable;
            # output requests are not visible until the compute contract finishes.
            for item in active.values():
                if (
                    item.phase == "reading"
                    and not item.pending_reads
                    and not item.outstanding
                ):
                    item.phase = "computing"
                    item.compute_done_cycle = current_cycle + item.compute_cycles
                if (
                    item.phase == "computing"
                    and current_cycle >= item.compute_done_cycle
                ):
                    item.phase = "writing"
                    item.writes_released = True

            # Complete before launching successors at the same cycle.
            completed_now = [
                item
                for item in active.values()
                if item.phase == "writing"
                and not item.pending_writes
                and not item.outstanding
            ]
            for item in sorted(completed_now, key=lambda value: value.task_id):
                commit_task(item, current_cycle)

            launchable = [
                task_id
                for task_id in ready
                if dependency_ready_cycle[task_id] <= current_cycle
                and str(by_id[task_id]["resource_id"]) not in resource_owner
            ]
            for task_id in sorted(launchable):
                ready.remove(task_id)
                item = (
                    make_device_active(task_id, current_cycle)
                    if by_id[task_id].get("task_kind") == "device"
                    else make_route_active(task_id, current_cycle)
                )
                active[task_id] = item
                resource_owner[item.resource_id] = task_id

            # Fair one-parent-per-active-task issue attempt each scheduler cycle.
            pending_retry = False
            for task_id in sorted(active):
                item = active[task_id]
                if (item.phase == "reading" and item.pending_reads) or (
                    item.phase == "writing" and item.pending_writes
                ):
                    accepted = submit_pending(item)
                    pending_retry = pending_retry or not accepted

            if len(completed) == len(by_id):
                break
            if not active and not ready:
                raise PrefillCycleRuntimeError("task graph is cyclic or deadlocked")

            current_cycle = bridge.current_cycle
            if pending_retry or any(
                (item.phase == "reading" and item.pending_reads)
                or (item.phase == "writing" and item.pending_writes)
                for item in active.values()
            ):
                delta = 1
            else:
                future = [
                    item.compute_done_cycle
                    for item in active.values()
                    if item.phase == "computing"
                    and item.compute_done_cycle > current_cycle
                ]
                future.extend(
                    dependency_ready_cycle[task_id]
                    for task_id in ready
                    if dependency_ready_cycle[task_id] > current_cycle
                )
                delta = max(1, min(future) - current_cycle) if future else 1
            bridge.advance_until_event(delta)

        memory_stats = bridge.close()
    except Exception:
        # Destruction drains accepted work and prevents the global C bridge from
        # contaminating a later run in the same process.
        try:
            if not getattr(bridge, "_closed", False):
                bridge.close()
        except Exception:
            pass
        raise

    if backend_dispatch_count != len(device_task_ids):
        raise PrefillCycleRuntimeError(
            "Backend dispatch conservation failed: "
            f"expected={len(device_task_ids)}, actual={backend_dispatch_count}"
        )
    if int(memory_stats["accepted_parent_ids"]) != len(memory_trace):
        raise PrefillCycleRuntimeError("memory trace parent conservation failed")
    if int(memory_stats["outstanding"]) != 0:
        raise PrefillCycleRuntimeError("live Ramulator2 exited with outstanding work")

    coverage = dispatcher.coverage(device_task_ids)
    represented_read = sum(
        int(item.get("represented_bytes", 0))
        for item in memory_trace
        if item["operation"] == "read"
    )
    represented_write = sum(
        int(item.get("represented_bytes", 0))
        for item in memory_trace
        if item["operation"] == "write"
    )
    memory_stats.update(
        {
            "sampling_policy": "evenly_spaced_bounded",
            "max_samples_per_value": max_samples_per_value,
            "represented_read_bytes": represented_read,
            "represented_write_bytes": represented_write,
            "represented_total_bytes": represented_read + represented_write,
            "simulated_parent_payload_bytes": sum(
                int(item["size_bytes"]) for item in memory_trace
            ),
            "one_live_timing_owner": True,
        }
    )
    ordered_timing = [timing[str(record["task_id"])] for record in records]
    makespan_fs = max(int(item["completion_time_fs"]) for item in ordered_timing)
    return {
        "schema_version": "hetero-prefill-cycle-runtime/v1",
        "makespan_fs": makespan_fs,
        "tasks": ordered_timing,
        "backend_dispatch_count": backend_dispatch_count,
        "version_checks": version_checks,
        "launch_log": launch_log,
        "final_versions": dict(sorted(latest_version.items())),
        "memory_trace": memory_trace,
        "memory_statistics": memory_stats,
        "artifact_coverage": coverage,
    }
