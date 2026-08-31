#!/usr/bin/env python3
"""Qualify the deterministic two-pass P16 full-task Prefill timeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


class P16QualificationError(RuntimeError):
    """Raised when a P16 full-task qualification invariant is broken."""


GPU_OPERATORS = {
    "token_embedding",
    "attention_norm",
    "qkv_projection",
    "rope",
    "causal_attention",
    "output_projection",
    "residual_add",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
}
RUNTIME_MEMORY_OPERATORS = {"kv_allocate", "kv_append", "kv_release"}
HOST_CONTROL_OPERATORS = {"request_start", "request_finish"}
EXPECTED_RUNTIME_REQUESTS = {
    "kv_allocate": {"reads": 1, "writes": 2, "logical_bytes": 192},
    "kv_append": {"reads": 256, "writes": 256, "logical_bytes": 32_768},
    "kv_release": {"reads": 1, "writes": 1, "logical_bytes": 128},
}
EXPECTED_INSTANCES = Counter(
    {
        **{operator: 1 for operator in GPU_OPERATORS},
        **{operator: 1 for operator in RUNTIME_MEMORY_OPERATORS},
        **{operator: 1 for operator in HOST_CONTROL_OPERATORS},
        "residual_add": 2,
    }
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P16QualificationError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P16QualificationError(message)
    return value


def _timing(task: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(task.get("timing"), f"{task.get('task_id')} lacks timing")


def _audit_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "backend_runs" / "runtime" / task_id / "runtime_request_audit.json"


def _validate_runtime_memory(
    run_dir: Path, task: Mapping[str, object]
) -> dict[str, object]:
    task_id = str(task["task_id"])
    artifact = _mapping(
        task.get("compiled_artifact"), f"{task_id} lacks runtime artifact"
    )
    if (
        artifact.get("kind") != "runtime_live_ramulator2"
        or artifact.get("request_cycle_ready") is not True
        or artifact.get("device_performance_included") is not True
    ):
        raise P16QualificationError(f"{task_id} is not a ready live runtime task")
    if artifact.get("calibrated") is not False:
        raise P16QualificationError(f"{task_id} must remain explicitly uncalibrated")
    fidelity = _mapping(task.get("fidelity"), f"{task_id} lacks fidelity")
    if (
        fidelity.get("memory_fidelity") != "cycle_simulated_external_ramulator2"
        or fidelity.get("link_fidelity") != "cycle_simulated_external_link"
        or fidelity.get("performance_eligible") is not False
    ):
        raise P16QualificationError(f"{task_id} has an invalid fidelity boundary")

    audit = _load(_audit_path(run_dir, task_id))
    stats = _mapping(audit.get("memory_statistics"), f"{task_id} lacks live stats")
    backend_stats = _mapping(
        task.get("backend_statistics"), f"{task_id} lacks runtime Backend stats"
    )
    backend_live = _mapping(
        backend_stats.get("live_memory_statistics"),
        f"{task_id} lacks embedded live-memory stats",
    )
    if dict(backend_live) != dict(stats):
        raise P16QualificationError(f"{task_id} audit and execution stats differ")
    request_count = int(stats.get("request_count", -1))
    read_count = int(stats.get("read_request_count", -1))
    write_count = int(stats.get("write_request_count", -1))
    expected = EXPECTED_RUNTIME_REQUESTS[str(task["op"])]
    if request_count <= 0 or request_count != read_count + write_count:
        raise P16QualificationError(f"{task_id} has an invalid request split")
    if (
        read_count != expected["reads"]
        or write_count != expected["writes"]
        or int(stats.get("logical_bytes", -1)) != expected["logical_bytes"]
    ):
        raise P16QualificationError(f"{task_id} does not match exact P16 traffic")
    if (
        int(stats.get("instances", -1)) != 1
        or int(stats.get("accepted_parent_ids", -1)) != request_count
        or int(stats.get("observed_completion_ids", -1)) != request_count
        or int(stats.get("completed", -1)) != request_count
        or int(stats.get("durable_completed", -1)) != request_count
        or int(stats.get("children_sent", -1))
        != int(stats.get("children_completed", -2))
        or int(stats.get("outstanding", -1)) != 0
    ):
        raise P16QualificationError(f"{task_id} breaks live request conservation")
    initiators = _mapping(stats.get("initiators"), f"{task_id} lacks initiators")
    gpu = _mapping(initiators.get("gpu0"), f"{task_id} lacks gpu0 initiator")
    atlas = _mapping(
        initiators.get("atlas0.compute"), f"{task_id} lacks ATLAS initiator"
    )
    if (
        int(gpu.get("parents", -1)) != request_count
        or int(gpu.get("completed", -1)) != request_count
        or any(int(atlas.get(key, -1)) != 0 for key in ("parents", "children", "completed"))
    ):
        raise P16QualificationError(f"{task_id} has invalid initiator accounting")
    requests = audit.get("requests")
    if not isinstance(requests, list) or len(requests) != request_count:
        raise P16QualificationError(f"{task_id} request audit is incomplete")
    if any(int(item.get("size_bytes", 0)) != 64 for item in requests if isinstance(item, Mapping)):
        raise P16QualificationError(f"{task_id} must use exact 64-byte requests")
    if len({int(item["parent_id"]) for item in requests if isinstance(item, Mapping)}) != request_count:
        raise P16QualificationError(f"{task_id} parent IDs are not unique")
    completion = int(_timing(task)["completion_time_fs"])
    duration = int(audit.get("duration_fs", -1))
    if duration <= 0 or int(task.get("duration_fs", -1)) != duration:
        raise P16QualificationError(f"{task_id} audit duration does not drive timeline")
    expected_execution_cycles = int(stats["gpu_cycles"]) + int(
        stats["runtime_fixed_cycles"]
    )
    if (
        int(backend_stats.get("cycles", -1)) != expected_execution_cycles
        or int(backend_stats.get("contract_cycles", -1))
        != int(audit.get("contract_cycles", -2))
        or int(backend_stats.get("clock_hz", 0)) <= 0
    ):
        raise P16QualificationError(f"{task_id} execution/contract cycles differ")
    return {
        "task_id": task_id,
        "duration_fs": duration,
        "completion_time_fs": completion,
        "requests": request_count,
        "reads": read_count,
        "writes": write_count,
        "logical_bytes": int(stats.get("logical_bytes", -1)),
        "execution_cycles": expected_execution_cycles,
        "contract_cycles": int(audit["contract_cycles"]),
        "execution_clock_hz": int(backend_stats["clock_hz"]),
        "children": int(stats.get("children_sent", -1)),
        "rejected_admission_attempts": int(stats.get("rejected", -1)),
        "ramulator2_instances": int(stats.get("instances", -1)),
        "outstanding": int(stats.get("outstanding", -1)),
    }


def _validate_gpu_task(task: Mapping[str, object]) -> dict[str, object]:
    task_id = str(task["task_id"])
    artifact = _mapping(task.get("compiled_artifact"), f"{task_id} lacks artifact")
    if artifact.get("kind") != "accel_sim_trace" or artifact.get("request_cycle_ready") is not True:
        raise P16QualificationError(f"{task_id} is not a ready Accel-Sim task")
    stats = _mapping(task.get("backend_statistics"), f"{task_id} lacks Backend stats")
    external = _mapping(
        stats.get("external_memory_stats"), f"{task_id} lacks external-memory stats"
    )
    parents = int(external.get("gpu_parents", -1))
    children = int(external.get("gpu_children", -1))
    if (
        int(stats.get("cycles", 0)) <= 0
        or int(stats.get("instructions", 0)) <= 0
        or int(external.get("instances", -1)) != 1
        or parents <= 0
        or children <= 0
        or int(external.get("gpu_completed", -1)) != parents
        or int(external.get("children_sent", -1)) != children
        or int(external.get("children_completed", -1)) != children
        or int(external.get("durable_completed", -1)) != parents
        or int(external.get("address_translated", 0)) <= 0
        or int(external.get("address_unmapped", -1)) != 0
        or int(external.get("atlas_parents", -1)) != 0
        or int(external.get("atlas_children", -1)) != 0
        or int(external.get("atlas_completed", -1)) != 0
        or int(external.get("outstanding", -1)) != 0
    ):
        raise P16QualificationError(f"{task_id} breaks Accel-Sim request invariants")
    return {
        "task_id": task_id,
        "operator": str(task["op"]),
        "duration_fs": int(task["duration_fs"]),
        "cycles": int(stats["cycles"]),
        "instructions": int(stats["instructions"]),
        "parents": parents,
        "children": children,
        "translated": int(external["address_translated"]),
        "ramulator2_instances": int(external["instances"]),
        "outstanding": int(external["outstanding"]),
    }


def _validate_control(task: Mapping[str, object]) -> dict[str, object]:
    task_id = str(task["task_id"])
    artifact = _mapping(task.get("compiled_artifact"), f"{task_id} lacks artifact")
    fidelity = _mapping(task.get("fidelity"), f"{task_id} lacks fidelity")
    stats = _mapping(task.get("backend_statistics"), f"{task_id} lacks stats")
    live = _mapping(stats.get("live_memory_statistics"), f"{task_id} lacks boundary stats")
    if (
        artifact.get("kind") != "host_control_event"
        or artifact.get("causal_timeline_ready") is not True
        or artifact.get("request_cycle_ready") is not False
        or artifact.get("device_performance_included") is not False
        or artifact.get("calibrated") is not False
        or fidelity.get("compute_fidelity") != "host_control_boundary_uncalibrated"
        or fidelity.get("device_performance_included") is not False
        or fidelity.get("memory_fidelity") != "not_applicable"
        or fidelity.get("performance_eligible") is not False
        or live.get("host_control_only") is not True
        or int(live.get("instances", -1)) != 0
        or int(live.get("outstanding", -1)) != 0
    ):
        raise P16QualificationError(f"{task_id} violates the host-control boundary")
    return {
        "task_id": task_id,
        "duration_fs": int(task["duration_fs"]),
        "causal_timeline_ready": True,
        "device_performance_included": False,
        "memory_model": "not_applicable",
        "calibrated": False,
    }


def summarize_leg(run_dir: Path) -> dict[str, object]:
    execution = _load(run_dir / "execution_graph.json")
    online = _load(run_dir / "online_dispatch.json")
    memory_map = _load(run_dir / "global_memory_map.json")
    metrics = _load(run_dir / "metrics.json")
    provenance = _load(run_dir / "provenance.json")
    raw_tasks = execution.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 20:
        raise P16QualificationError("P16 execution graph must contain 20 tasks")
    tasks = {
        str(item["task_id"]): item for item in raw_tasks if isinstance(item, Mapping)
    }
    if len(tasks) != 20 or Counter(str(item["op"]) for item in tasks.values()) != EXPECTED_INSTANCES:
        raise P16QualificationError("P16 task/operator coverage is not 19 types / 20 instances")

    for task in tasks.values():
        start = int(_timing(task)["start_time_fs"])
        for dependency_id in task.get("dependencies", []):
            dependency = tasks.get(str(dependency_id))
            if dependency is None or start < int(_timing(dependency)["completion_time_fs"]):
                raise P16QualificationError(
                    f"{task['task_id']} starts before dependency {dependency_id}"
                )
    gpu_tasks = sorted(tasks.values(), key=lambda item: int(_timing(item)["start_time_fs"]))
    if any(item.get("resource_id") != "gpu0" for item in gpu_tasks):
        raise P16QualificationError("all P16 tasks must serialize on gpu0")
    for left, right in zip(gpu_tasks, gpu_tasks[1:]):
        if int(_timing(right)["start_time_fs"]) < int(_timing(left)["completion_time_fs"]):
            raise P16QualificationError(f"gpu0 overlap: {left['task_id']} / {right['task_id']}")

    ranges = memory_map.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != int(memory_map.get("allocation_count", -1)):
        raise P16QualificationError("Global PA allocation records are incomplete")
    ordered = sorted(
        (item for item in ranges if isinstance(item, Mapping)),
        key=lambda item: int(item["base_address"]),
    )
    for left, right in zip(ordered, ordered[1:]):
        if int(left["end_address_exclusive"]) > int(right["base_address"]):
            raise P16QualificationError(f"Global PA overlap: {left['value_id']} / {right['value_id']}")
    if (
        memory_map.get("non_overlapping") is not True
        or int(memory_map.get("allocated_bytes", -1)) > int(memory_map.get("capacity_bytes", -1))
        or int(memory_map.get("external_input_shadow_count", -1)) != 1
    ):
        raise P16QualificationError("Global PA capacity/shadow invariant failed")
    request_bindings = memory_map.get("request_cycle_bindings")
    runtime_bindings = memory_map.get("runtime_task_bindings")
    if not isinstance(request_bindings, list) or len(request_bindings) != 15:
        raise P16QualificationError("expected 15 GPU request-cycle bindings")
    if not isinstance(runtime_bindings, list) or len(runtime_bindings) != 5:
        raise P16QualificationError("expected five runtime-task bindings")
    runtime_binding_by_operator = {
        str(item["operator"]): item
        for item in runtime_bindings
        if isinstance(item, Mapping)
    }
    if set(runtime_binding_by_operator) != RUNTIME_MEMORY_OPERATORS | HOST_CONTROL_OPERATORS:
        raise P16QualificationError("runtime-task binding operators are incomplete")
    for operator, expected in EXPECTED_RUNTIME_REQUESTS.items():
        binding = runtime_binding_by_operator[operator]
        if (
            int(binding.get("planned_parent_requests", -1))
            != expected["reads"] + expected["writes"]
            or int(binding.get("planned_read_bytes", -1)) != expected["reads"] * 64
            or int(binding.get("planned_write_bytes", -1)) != expected["writes"] * 64
            or binding.get("request_cycle_ready") is not True
        ):
            raise P16QualificationError(f"{operator} runtime binding differs from plan")
    for operator in HOST_CONTROL_OPERATORS:
        binding = runtime_binding_by_operator[operator]
        if (
            int(binding.get("planned_parent_requests", -1)) != 0
            or binding.get("host_control_excluded") is not True
            or binding.get("request_cycle_ready") is not False
        ):
            raise P16QualificationError(f"{operator} control binding is invalid")

    gpu_records = [
        _validate_gpu_task(task)
        for task in tasks.values()
        if str(task["op"]) in GPU_OPERATORS
    ]
    runtime_records = {
        str(task["op"]): _validate_runtime_memory(run_dir, task)
        for task in tasks.values()
        if str(task["op"]) in RUNTIME_MEMORY_OPERATORS
    }
    control_records = {
        str(task["op"]): _validate_control(task)
        for task in tasks.values()
        if str(task["op"]) in HOST_CONTROL_OPERATORS
    }
    if len(gpu_records) != 15 or len(runtime_records) != 3 or len(control_records) != 2:
        raise P16QualificationError("Backend instance partition is incomplete")

    commits = online.get("version_commits")
    if not isinstance(commits, list) or len(commits) != 18:
        raise P16QualificationError("P16 must contain 18 completion-time commits")
    commits_by_task: dict[str, list[Mapping[str, object]]] = {}
    for item in commits:
        if isinstance(item, Mapping):
            commits_by_task.setdefault(str(item["task_id"]), []).append(item)
    for task in tasks.values():
        output_values = task.get("output_values")
        if not isinstance(output_values, list) or not output_values:
            continue
        task_id = str(task["task_id"])
        completion = int(_timing(task)["completion_time_fs"])
        task_commits = commits_by_task.get(task_id, [])
        if len(task_commits) != len(output_values) or any(
            int(item["commit_time_fs"]) != completion
            or item.get("cause") != "backend_completion"
            for item in task_commits
        ):
            raise P16QualificationError(f"{task_id} output commits are not causal")
        expected_inputs = {
            (str(item["value_id"]), int(item["version"]))
            for item in task.get("input_values", [])
            if isinstance(item, Mapping)
        }
        validated_inputs = {
            (str(item["value_id"]), int(item["version"]))
            for item in task.get("validated_input_versions", [])
            if isinstance(item, Mapping)
        }
        if expected_inputs != validated_inputs:
            raise P16QualificationError(f"{task_id} input version validation differs")

    boundary = _mapping(online.get("performance_boundary"), "performance boundary missing")
    if (
        int(online.get("backend_dispatch_count", -1)) != 20
        or int(boundary.get("included_task_count", -1)) != 18
        or int(boundary.get("excluded_task_count", -1)) != 2
        or int(boundary.get("excluded_control_duration_fs", -1))
        != sum(record["duration_fs"] for record in control_records.values())
        or metrics.get("performance_claim_allowed") is not False
    ):
        raise P16QualificationError("performance/control boundary is invalid")

    gpu_records.sort(key=lambda item: item["task_id"])
    return {
        "run_dir": str(run_dir.resolve()),
        "simulation_input_key": provenance.get("simulation_input_key"),
        "simulator_revision": provenance.get("simulator_revision"),
        "task_count": len(tasks),
        "operator_type_count": len(EXPECTED_INSTANCES),
        "backend_dispatch_count": int(online["backend_dispatch_count"]),
        "makespan_fs": int(metrics["makespan_fs"]),
        "performance_boundary": dict(boundary),
        "global_pa": {
            "capacity_bytes": int(memory_map["capacity_bytes"]),
            "allocated_bytes": int(memory_map["allocated_bytes"]),
            "allocation_count": len(ordered),
            "operator_workspace_count": int(memory_map["operator_workspace_count"]),
            "request_cycle_binding_count": len(request_bindings),
            "runtime_task_binding_count": len(runtime_bindings),
            "external_input_shadow_count": int(memory_map["external_input_shadow_count"]),
            "non_overlapping": True,
        },
        "gpu_request_cycle": gpu_records,
        "runtime_memory": runtime_records,
        "host_control": control_records,
        "version_causality": {
            "input_version_checks": int(online["version_checks"]),
            "completion_time_commits": len(commits),
            "all_inputs_validated": True,
            "all_outputs_committed_at_backend_completion": True,
        },
        "dependency_causality": {"all_dependencies_complete_before_start": True},
        "resource_causality": {"resource_id": "gpu0", "all_tasks_non_overlapping": True},
    }


def summarize(first_run_dir: Path, second_run_dir: Path) -> dict[str, object]:
    legs = [summarize_leg(first_run_dir), summarize_leg(second_run_dir)]
    comparable_keys = (
        "simulation_input_key",
        "task_count",
        "operator_type_count",
        "backend_dispatch_count",
        "makespan_fs",
        "performance_boundary",
        "global_pa",
        "gpu_request_cycle",
        "runtime_memory",
        "host_control",
        "version_causality",
        "dependency_causality",
        "resource_causality",
    )
    mismatches = [key for key in comparable_keys if legs[0][key] != legs[1][key]]
    if mismatches:
        raise P16QualificationError(f"P16 double-run mismatch: {mismatches}")
    runtime_totals = {
        "requests_per_leg": sum(
            int(record["requests"])
            for record in legs[0]["runtime_memory"].values()
        ),
        "logical_bytes_per_leg": sum(
            int(record["logical_bytes"])
            for record in legs[0]["runtime_memory"].values()
        ),
        "ramulator2_instances_per_runtime_task": 1,
        "atlas_requests": 0,
        "outstanding_at_exit": 0,
    }
    return {
        "schema_version": "hetero-p16-full-task-timeline-qualification/v1",
        "status": "passed",
        "performance_eligible": False,
        "claim_boundary": (
            "all 20 fixed TinyLlama Prefill tasks are causally modeled and double-run "
            "deterministic; 15 GPU Trace instances and three runtime memory tasks are "
            "request-cycle ready, while two uncalibrated host-control events are excluded "
            "from the device performance boundary; no end-to-end performance claim is allowed"
        ),
        "fixed_scope": {
            "model": "TinyLlama-1.1B",
            "checkpoint_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "phase": "prefill",
            "layer_count": 1,
            "batch_size": 1,
            "context_length": 16,
            "dtype": "fp16",
        },
        "double_run_deterministic": True,
        "runtime_totals": runtime_totals,
        "legs": legs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_run_dir", type=Path)
    parser.add_argument("second_run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize(args.first_run_dir, args.second_run_dir)
    output = args.output or args.first_run_dir.parent.parent / "p16_qualification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
