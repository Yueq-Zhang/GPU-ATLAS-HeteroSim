"""Dependency-gated operator-event runtime for real Backend launches.

Backend subprocesses are invoked only when their simulated release time,
resource availability, route dependencies, and input value versions are all
ready.  Wall-clock execution remains an implementation detail; the launch and
completion timestamps belong to the deterministic simulated timeline.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .ir import ModelNode
from .model_graph import ModelSpec
from .operator_event import BackendTaskResult


class OnlineOperatorRuntimeError(RuntimeError):
    """Raised when launch order, versions, or dispatch conservation is invalid."""


@dataclass(frozen=True, slots=True)
class OnlineDispatchSpec:
    task_id: str
    backend_key: str
    node: ModelNode
    model: ModelSpec
    device_id: str
    input_values: tuple[Mapping[str, object], ...] = ()
    output_values: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _Event:
    time_fs: int
    priority: int
    sequence: int
    kind: str
    task_id: str

    def heap_key(self) -> tuple[int, int, int, str, str]:
        return (self.time_fs, self.priority, self.sequence, self.kind, self.task_id)


def _records(execution_graph: Mapping[str, object]) -> list[dict[str, object]]:
    tasks = execution_graph.get("tasks")
    routes = execution_graph.get("routes")
    if not isinstance(tasks, list) or not isinstance(routes, list):
        raise OnlineOperatorRuntimeError(
            "execution graph must contain task and route arrays"
        )
    result: list[dict[str, object]] = []
    for raw in [*tasks, *routes]:
        if not isinstance(raw, dict):
            raise OnlineOperatorRuntimeError("runtime records must be mutable objects")
        result.append(raw)
    return result


def run_online_operator_dag(
    execution_graph: Mapping[str, object],
    dispatch_specs: Mapping[str, OnlineDispatchSpec],
    dispatch: Callable[[OnlineDispatchSpec], BackendTaskResult],
) -> dict[str, object]:
    """Launch each concrete Backend only after dependency and version gates pass."""

    records = _records(execution_graph)
    by_id: dict[str, dict[str, object]] = {}
    device_ids: set[str] = set()
    for record in records:
        task_id = str(record.get("task_id", ""))
        resource_id = str(record.get("resource_id", ""))
        if not task_id or not resource_id:
            raise OnlineOperatorRuntimeError(
                "online runtime records require task_id and resource_id"
            )
        if task_id in by_id:
            raise OnlineOperatorRuntimeError(f"duplicate runtime task {task_id}")
        by_id[task_id] = record
        if record.get("task_kind") == "device":
            device_ids.add(task_id)
    if device_ids != set(dispatch_specs):
        raise OnlineOperatorRuntimeError(
            "dispatch specs must equal device tasks: "
            f"missing={sorted(device_ids - set(dispatch_specs))}, "
            f"extra={sorted(set(dispatch_specs) - device_ids)}"
        )

    remaining: dict[str, int] = {}
    dependents: dict[str, list[str]] = {task_id: [] for task_id in by_id}
    dependency_ready: dict[str, int] = {}
    for task_id, record in by_id.items():
        raw_dependencies = record.get("dependencies", [])
        if not isinstance(raw_dependencies, Sequence) or isinstance(
            raw_dependencies, (str, bytes)
        ):
            raise OnlineOperatorRuntimeError(
                f"dependencies must be an array: {task_id}"
            )
        dependencies = [str(item) for item in raw_dependencies]
        if len(dependencies) != len(set(dependencies)):
            raise OnlineOperatorRuntimeError(f"duplicate dependency for {task_id}")
        for dependency in dependencies:
            if dependency not in by_id:
                raise OnlineOperatorRuntimeError(
                    f"unknown dependency {dependency} for {task_id}"
                )
            if dependency == task_id:
                raise OnlineOperatorRuntimeError("task cannot depend on itself")
            dependents[dependency].append(task_id)
        remaining[task_id] = len(dependencies)
        dependency_ready[task_id] = int(record.get("release_time_fs", 0))

    plan = execution_graph.get("residency_plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("events"), list):
        raise OnlineOperatorRuntimeError(
            "online runtime requires residency_plan events"
        )
    initial_by_task: dict[str, list[Mapping[str, object]]] = {}
    for raw in plan["events"]:
        if not isinstance(raw, Mapping):
            raise OnlineOperatorRuntimeError("residency plan event must be an object")
        if raw.get("event") == "register_external_input":
            initial_by_task.setdefault(str(raw["trigger_task_id"]), []).append(raw)

    latest_version: dict[str, int] = {}
    available_version: dict[tuple[str, str], int] = {}
    version_checks = 0

    def register_external_inputs(task_id: str) -> None:
        for item in initial_by_task.get(task_id, []):
            value_id = str(item["value_id"])
            version = int(item["version"])
            device_id = str(item["device_id"])
            existing = latest_version.get(value_id)
            if existing is not None and existing != version:
                raise OnlineOperatorRuntimeError(
                    f"external input {value_id} conflicts with version {existing}"
                )
            latest_version[value_id] = version
            available_version[(value_id, device_id)] = version

    def require_value(
        value_id: str, version: int, device_id: str, task_id: str
    ) -> None:
        nonlocal version_checks
        version_checks += 1
        latest = latest_version.get(value_id)
        available = available_version.get((value_id, device_id))
        if latest != version or available != version:
            raise OnlineOperatorRuntimeError(
                f"stale or unavailable value for {task_id}: {value_id} "
                f"needs v{version} on {device_id}, latest={latest}, "
                f"available={available}"
            )

    events: list[tuple[int, int, int, str, str]] = []
    next_sequence = 0

    def schedule(time_fs: int, priority: int, kind: str, task_id: str) -> None:
        nonlocal next_sequence
        event = _Event(time_fs, priority, next_sequence, kind, task_id)
        heapq.heappush(events, event.heap_key())
        next_sequence += 1

    for task_id in by_id:
        if remaining[task_id] == 0:
            schedule(dependency_ready[task_id], 4, "launch", task_id)

    resource_available: dict[str, int] = {}
    dispatched: set[str] = set()
    completed: set[str] = set()
    timings: dict[str, dict[str, object]] = {}
    launch_log: list[dict[str, object]] = []
    makespan = 0
    backend_dispatch_count = 0
    version_commits: list[dict[str, object]] = []
    performance_included_intervals: list[tuple[int, int, str]] = []
    performance_excluded_duration_fs = 0

    while events:
        time_fs, priority, _sequence, kind, task_id = heapq.heappop(events)
        record = by_id[task_id]
        if kind == "launch":
            if task_id in dispatched:
                continue
            if remaining[task_id] != 0:
                raise OnlineOperatorRuntimeError(
                    f"task launched before dependencies completed: {task_id}"
                )
            resource_id = str(record["resource_id"])
            start_time = max(
                time_fs,
                dependency_ready[task_id],
                resource_available.get(resource_id, 0),
            )
            if start_time > time_fs:
                schedule(start_time, 4, "launch", task_id)
                continue

            if record.get("task_kind") == "device":
                register_external_inputs(task_id)
                device_id = str(record["device_id"])
                inputs = record.get("input_values", [])
                if not isinstance(inputs, list):
                    raise OnlineOperatorRuntimeError("input_values must be an array")
                checked_inputs: list[dict[str, object]] = []
                for value in inputs:
                    if not isinstance(value, Mapping):
                        raise OnlineOperatorRuntimeError(
                            "input value must be an object"
                        )
                    value_id = str(value["value_id"])
                    version = int(value["version"])
                    require_value(value_id, version, device_id, task_id)
                    checked_inputs.append({"value_id": value_id, "version": version})
                result = dispatch(dispatch_specs[task_id])
                if result.resource_id != resource_id:
                    raise OnlineOperatorRuntimeError(
                        f"Backend changed resource for {task_id}: "
                        f"expected {resource_id}, got {result.resource_id}"
                    )
                if result.duration_fs <= 0:
                    raise OnlineOperatorRuntimeError(
                        f"Backend returned non-positive duration for {task_id}"
                    )
                record.update(
                    {
                        "backend_id": result.backend_id,
                        "duration_fs": result.duration_fs,
                        "timing_contract": dict(result.timing_contract),
                        "backend_statistics": dict(result.statistics),
                        "compiled_artifact": dict(result.artifact),
                        "fidelity": dict(result.fidelity),
                        "backend_launch_time_fs": start_time,
                        "validated_input_versions": checked_inputs,
                    }
                )
                duration_fs = result.duration_fs
                if bool(result.fidelity.get("device_performance_included", True)):
                    performance_included_intervals.append(
                        (start_time, start_time + duration_fs, task_id)
                    )
                else:
                    performance_excluded_duration_fs += duration_fs
                backend_dispatch_count += 1
                launch_log.append(
                    {
                        "task_id": task_id,
                        "kind": "backend",
                        "time_fs": start_time,
                        "device_id": device_id,
                        "validated_input_versions": checked_inputs,
                    }
                )
            else:
                value_id = str(record["value_id"])
                version = int(record["value_version"])
                producer = str(record["producer_device"])
                require_value(value_id, version, producer, task_id)
                duration_fs = int(record.get("duration_fs", 0))
                if duration_fs <= 0:
                    raise OnlineOperatorRuntimeError(
                        f"route duration must be positive: {task_id}"
                    )
                launch_log.append(
                    {
                        "task_id": task_id,
                        "kind": "route",
                        "time_fs": start_time,
                        "value_id": value_id,
                        "version": version,
                    }
                )

            completion_time = start_time + duration_fs
            timings[task_id] = {
                "task_id": task_id,
                "resource_id": resource_id,
                "ready_time_fs": dependency_ready[task_id],
                "start_time_fs": start_time,
                "completion_time_fs": completion_time,
            }
            resource_available[resource_id] = completion_time
            dispatched.add(task_id)
            schedule(completion_time, 0, "complete", task_id)
            continue

        if kind != "complete" or priority != 0 or task_id in completed:
            raise OnlineOperatorRuntimeError(f"invalid completion event for {task_id}")
        if record.get("task_kind") == "device":
            device_id = str(record["device_id"])
            outputs = record.get("output_values", [])
            if not isinstance(outputs, list):
                raise OnlineOperatorRuntimeError("output_values must be an array")
            for value in outputs:
                if not isinstance(value, Mapping):
                    raise OnlineOperatorRuntimeError("output value must be an object")
                value_id = str(value["value_id"])
                version = int(value["version"])
                previous = latest_version.get(value_id)
                if version != (1 if previous is None else previous + 1):
                    raise OnlineOperatorRuntimeError(
                        f"non-monotonic write by {task_id}: {value_id} v{version}, "
                        f"previous={previous}"
                    )
                latest_version[value_id] = version
                for key in [key for key in available_version if key[0] == value_id]:
                    del available_version[key]
                available_version[(value_id, device_id)] = version
                version_commits.append(
                    {
                        "task_id": task_id,
                        "value_id": value_id,
                        "version": version,
                        "device_id": device_id,
                        "commit_time_fs": time_fs,
                        "cause": "backend_completion",
                    }
                )
        else:
            value_id = str(record["value_id"])
            version = int(record["value_version"])
            destination = str(record["consumer_device"])
            if latest_version.get(value_id) != version:
                raise OnlineOperatorRuntimeError(
                    f"route completed with stale version: {task_id}"
                )
            available_version[(value_id, destination)] = version
            route_kind = (
                record["kind"].value
                if hasattr(record.get("kind"), "value")
                else str(record.get("kind"))
            )
            if route_kind == "migration":
                available_version.pop((value_id, str(record["producer_device"])), None)

        completed.add(task_id)
        makespan = max(makespan, time_fs)
        for dependent in sorted(dependents[task_id]):
            dependency_ready[dependent] = max(dependency_ready[dependent], time_fs)
            remaining[dependent] -= 1
            if remaining[dependent] < 0:
                raise OnlineOperatorRuntimeError("dependency counter underflow")
            if remaining[dependent] == 0:
                schedule(dependency_ready[dependent], 4, "launch", dependent)

    if len(completed) != len(by_id):
        raise OnlineOperatorRuntimeError("task graph contains a dependency cycle")
    if backend_dispatch_count != len(device_ids):
        raise OnlineOperatorRuntimeError(
            "Backend dispatch conservation failed: "
            f"expected={len(device_ids)}, actual={backend_dispatch_count}"
        )
    device_boundary_start = (
        min(item[0] for item in performance_included_intervals)
        if performance_included_intervals
        else None
    )
    device_boundary_end = (
        max(item[1] for item in performance_included_intervals)
        if performance_included_intervals
        else None
    )
    return {
        "schema_version": "hetero-online-operator-runtime/v1",
        "makespan_fs": makespan,
        "tasks": [timings[str(record["task_id"])] for record in records],
        "backend_dispatch_count": backend_dispatch_count,
        "version_checks": version_checks,
        "launch_log": launch_log,
        "version_commits": version_commits,
        "final_versions": dict(sorted(latest_version.items())),
        "performance_boundary": {
            "schema_version": "hetero-performance-boundary/v1",
            "causal_makespan_fs": makespan,
            "device_boundary_start_fs": device_boundary_start,
            "device_boundary_end_fs": device_boundary_end,
            "device_boundary_span_fs": (
                device_boundary_end - device_boundary_start
                if device_boundary_start is not None
                and device_boundary_end is not None
                else None
            ),
            "excluded_control_duration_fs": performance_excluded_duration_fs,
            "included_task_count": len(performance_included_intervals),
            "excluded_task_count": len(device_ids)
            - len(performance_included_intervals),
        },
    }
