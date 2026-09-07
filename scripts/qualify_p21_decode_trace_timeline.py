#!/usr/bin/env python3
"""Qualify deterministic P21 real-Trace four-token Decode timelines."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


class P21QualificationError(RuntimeError):
    """Raised when a P21 instruction/request-cycle invariant is broken."""


REQUEST_ID = "TINYLLAMA11B-DECODE4-R0"
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


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P21QualificationError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P21QualificationError(message)
    return value


def _timing(task: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(task.get("timing"), f"{task.get('task_id')} lacks timing")


def _expected_counts(layers: int) -> Counter[str]:
    counts: Counter[str] = Counter(
        {
            "request_start": 1,
            "kv_allocate": 1,
            "request_finish": 1,
            "kv_release": 1,
            "token_embedding": 4,
            "final_norm": 4,
            "lm_head": 4,
            "sampling": 4,
        }
    )
    for operator in (
        "attention_norm",
        "qkv_projection",
        "rope",
        "causal_attention",
        "output_projection",
        "mlp_norm",
        "gate_up_projection",
        "silu_multiply",
        "down_projection",
        "kv_append",
    ):
        counts[operator] = 4 * layers
    counts["residual_add"] = 8 * layers
    return counts


def _validate_gpu_task(task: Mapping[str, object]) -> dict[str, object]:
    task_id = str(task["task_id"])
    artifact = _mapping(task.get("compiled_artifact"), f"{task_id} lacks artifact")
    fidelity = _mapping(task.get("fidelity"), f"{task_id} lacks fidelity")
    stats = _mapping(task.get("backend_statistics"), f"{task_id} lacks stats")
    external = _mapping(
        stats.get("external_memory_stats"),
        f"{task_id} lacks external-memory stats",
    )
    parents = int(external.get("gpu_parents", -1))
    children = int(external.get("gpu_children", -1))
    if (
        artifact.get("kind") != "accel_sim_trace"
        or artifact.get("request_cycle_ready") is not True
        or fidelity.get("trace_coverage") != 1.0
        or fidelity.get("performance_eligible") is not False
        or int(stats.get("cycles", 0)) <= 0
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
        raise P21QualificationError(f"{task_id} breaks Accel-Sim request invariants")
    step_id = int(task["step_id"])
    expected_kv = 17 + step_id
    expected_artifact_id = (
        f"tinyllama.1_1b.layer0.{task['op']}.decode_step.bs1.ctx16.q1."
        f"kv{expected_kv}.fp16.sm86.accel_sim_v2."
        "shared_hbdram_range_rebase_v1"
    )
    if artifact.get("operator_artifact_id") != expected_artifact_id:
        raise P21QualificationError(f"{task_id} uses the wrong shape-locked Trace")
    return {
        "task_id": task_id,
        "operator": str(task["op"]),
        "step_id": step_id,
        "kv_length": expected_kv,
        "duration_fs": int(task["duration_fs"]),
        "cycles": int(stats["cycles"]),
        "instructions": int(stats["instructions"]),
        "parents": parents,
        "children": children,
        "translated": int(external["address_translated"]),
        "ramulator2_instances": int(external["instances"]),
        "outstanding": int(external["outstanding"]),
    }


def _runtime_audit_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "backend_runs" / "runtime" / task_id / "runtime_request_audit.json"


def _validate_runtime_memory(
    run_dir: Path, task: Mapping[str, object]
) -> dict[str, object]:
    task_id = str(task["task_id"])
    artifact = _mapping(task.get("compiled_artifact"), f"{task_id} lacks artifact")
    fidelity = _mapping(task.get("fidelity"), f"{task_id} lacks fidelity")
    backend = _mapping(task.get("backend_statistics"), f"{task_id} lacks stats")
    if (
        artifact.get("kind") != "runtime_live_ramulator2"
        or artifact.get("request_cycle_ready") is not True
        or artifact.get("calibrated") is not False
        or fidelity.get("memory_fidelity") != "cycle_simulated_external_ramulator2"
        or fidelity.get("performance_eligible") is not False
    ):
        raise P21QualificationError(f"{task_id} has an invalid runtime boundary")
    audit = _load(_runtime_audit_path(run_dir, task_id))
    stats = _mapping(audit.get("memory_statistics"), f"{task_id} lacks live stats")
    embedded = _mapping(
        backend.get("live_memory_statistics"),
        f"{task_id} lacks embedded live stats",
    )
    if dict(stats) != dict(embedded):
        raise P21QualificationError(f"{task_id} audit and task stats differ")
    requests = int(stats.get("request_count", -1))
    reads = int(stats.get("read_request_count", -1))
    writes = int(stats.get("write_request_count", -1))
    initiators = _mapping(stats.get("initiators"), f"{task_id} lacks initiators")
    gpu = _mapping(initiators.get("gpu0"), f"{task_id} lacks gpu0 initiator")
    atlas = _mapping(
        initiators.get("atlas0.compute"), f"{task_id} lacks ATLAS initiator"
    )
    if (
        requests <= 0
        or requests != reads + writes
        or int(stats.get("instances", -1)) != 1
        or int(stats.get("accepted_parent_ids", -1)) != requests
        or int(stats.get("observed_completion_ids", -1)) != requests
        or int(stats.get("completed", -1)) != requests
        or int(stats.get("durable_completed", -1)) != requests
        or int(stats.get("children_sent", -1))
        != int(stats.get("children_completed", -2))
        or int(stats.get("outstanding", -1)) != 0
        or int(gpu.get("parents", -1)) != requests
        or int(gpu.get("completed", -1)) != requests
        or any(
            int(atlas.get(key, -1)) != 0 for key in ("parents", "children", "completed")
        )
    ):
        raise P21QualificationError(f"{task_id} breaks runtime request conservation")
    if task.get("op") == "kv_append" and (
        reads != 16 or writes != 16 or int(stats.get("logical_bytes", -1)) != 2048
    ):
        raise P21QualificationError(f"{task_id} does not match one-token KV traffic")
    return {
        "task_id": task_id,
        "operator": str(task["op"]),
        "step_id": int(task["step_id"]),
        "requests": requests,
        "reads": reads,
        "writes": writes,
        "logical_bytes": int(stats["logical_bytes"]),
        "cycles": int(backend["cycles"]),
        "children": int(stats["children_sent"]),
        "ramulator2_instances": int(stats["instances"]),
        "outstanding": int(stats["outstanding"]),
    }


def _validate_control(task: Mapping[str, object]) -> dict[str, object]:
    task_id = str(task["task_id"])
    artifact = _mapping(task.get("compiled_artifact"), f"{task_id} lacks artifact")
    fidelity = _mapping(task.get("fidelity"), f"{task_id} lacks fidelity")
    backend = _mapping(task.get("backend_statistics"), f"{task_id} lacks stats")
    live = _mapping(
        backend.get("live_memory_statistics"), f"{task_id} lacks boundary stats"
    )
    if (
        artifact.get("kind") != "host_control_event"
        or artifact.get("causal_timeline_ready") is not True
        or artifact.get("request_cycle_ready") is not False
        or artifact.get("device_performance_included") is not False
        or artifact.get("calibrated") is not False
        or fidelity.get("compute_fidelity") != "host_control_boundary_uncalibrated"
        or fidelity.get("performance_eligible") is not False
        or live.get("host_control_only") is not True
        or int(live.get("instances", -1)) != 0
        or int(live.get("outstanding", -1)) != 0
    ):
        raise P21QualificationError(f"{task_id} violates the control boundary")
    return {
        "task_id": task_id,
        "operator": str(task["op"]),
        "duration_fs": int(task["duration_fs"]),
        "device_performance_included": False,
        "calibrated": False,
    }


def _validate_dependencies_and_resource(
    tasks: Mapping[str, Mapping[str, object]],
) -> None:
    for task in tasks.values():
        start = int(_timing(task)["start_time_fs"])
        for dependency_id in task.get("dependencies", []):
            dependency = tasks.get(str(dependency_id))
            if dependency is None:
                raise P21QualificationError(
                    f"{task['task_id']} has unknown dependency {dependency_id}"
                )
            if start < int(_timing(dependency)["completion_time_fs"]):
                raise P21QualificationError(
                    f"{task['task_id']} starts before {dependency_id} completes"
                )
    ordered = sorted(
        tasks.values(), key=lambda item: int(_timing(item)["start_time_fs"])
    )
    if any(item.get("resource_id") != "gpu0" for item in ordered):
        raise P21QualificationError("all P21 tasks must serialize on gpu0")
    for left, right in zip(ordered, ordered[1:]):
        if int(_timing(right)["start_time_fs"]) < int(
            _timing(left)["completion_time_fs"]
        ):
            raise P21QualificationError(
                f"gpu0 overlap: {left['task_id']} / {right['task_id']}"
            )


def _validate_global_pa(
    memory_map: Mapping[str, object], trace_count: int, runtime_count: int
) -> dict[str, object]:
    ranges = memory_map.get("ranges")
    if not isinstance(ranges, list):
        raise P21QualificationError("Global PA ranges must be an array")
    ordered = sorted(
        (item for item in ranges if isinstance(item, Mapping)),
        key=lambda item: int(item["base_address"]),
    )
    for left, right in zip(ordered, ordered[1:]):
        if int(left["end_address_exclusive"]) > int(right["base_address"]):
            raise P21QualificationError(
                f"Global PA overlap: {left['value_id']} / {right['value_id']}"
            )
    if (
        len(ordered) != int(memory_map.get("allocation_count", -1))
        or memory_map.get("non_overlapping") is not True
        or int(memory_map.get("allocated_bytes", -1))
        > int(memory_map.get("capacity_bytes", -1))
    ):
        raise P21QualificationError("Global PA allocation/capacity invariant failed")
    allocation_base = {
        str(item["value_id"]): int(item["base_address"]) for item in ordered
    }
    request_bindings = memory_map.get("request_cycle_bindings")
    runtime_bindings = memory_map.get("runtime_task_bindings")
    if not isinstance(request_bindings, list) or len(request_bindings) != trace_count:
        raise P21QualificationError("request-cycle binding coverage is incomplete")
    if not isinstance(runtime_bindings, list) or len(runtime_bindings) != runtime_count:
        raise P21QualificationError("runtime-task binding coverage is incomplete")
    semantic_count = 0
    shadow_count = 0
    for record in request_bindings:
        binding = _mapping(record, "request binding must be an object")
        if binding.get("request_cycle_ready") is not True:
            raise P21QualificationError("request binding is not request-cycle ready")
        semantic = binding.get("semantic_bindings")
        if not isinstance(semantic, list) or not semantic:
            raise P21QualificationError("request binding lacks semantic values")
        for raw in semantic:
            item = _mapping(raw, "semantic binding must be an object")
            semantic_count += 1
            mode = str(item.get("binding_mode"))
            if mode == "external_input_widened_shadow":
                shadow_count += 1
                if int(item.get("size_bytes", 0)) < int(
                    item.get("logical_value_size_bytes", 0)
                ):
                    raise P21QualificationError("widened shadow is too small")
                continue
            value_id = str(item["value_id"])
            expected = allocation_base.get(value_id)
            actual = int(item["global_pa_base"]) - int(item["value_offset_bytes"])
            if mode != "semantic_value" or expected is None or actual != expected:
                raise P21QualificationError(
                    f"{item['tensor_id']} does not derive from its graph Value PA"
                )
    return {
        "capacity_bytes": int(memory_map["capacity_bytes"]),
        "allocated_bytes": int(memory_map["allocated_bytes"]),
        "allocation_count": len(ordered),
        "request_cycle_binding_count": len(request_bindings),
        "runtime_task_binding_count": len(runtime_bindings),
        "semantic_binding_count": semantic_count,
        "external_input_shadow_count": shadow_count,
        "non_overlapping": True,
    }


def _validate_versions(
    tasks: Mapping[str, Mapping[str, object]],
    online: Mapping[str, object],
    layers: int,
) -> dict[str, object]:
    commits = online.get("version_commits")
    if not isinstance(commits, list):
        raise P21QualificationError("online dispatch lacks version commits")
    commits_by_task: dict[str, list[Mapping[str, object]]] = {}
    for raw in commits:
        if isinstance(raw, Mapping):
            commits_by_task.setdefault(str(raw["task_id"]), []).append(raw)
    for task in tasks.values():
        outputs = task.get("output_values")
        if not isinstance(outputs, list) or not outputs:
            continue
        task_id = str(task["task_id"])
        task_commits = commits_by_task.get(task_id, [])
        completion = int(_timing(task)["completion_time_fs"])
        if len(task_commits) != len(outputs) or any(
            int(item["commit_time_fs"]) != completion
            or item.get("cause") != "backend_completion"
            for item in task_commits
        ):
            raise P21QualificationError(f"{task_id} output commits are not causal")
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
            raise P21QualificationError(f"{task_id} input versions differ")

    for step in range(4):
        for layer in range(layers):
            append_id = f"task.{REQUEST_ID}.decode.s{step}.l{layer}.attention.kv_append"
            attention_id = f"task.{REQUEST_ID}.decode.s{step}.l{layer}.attention.core"
            append = tasks[append_id]
            attention = tasks[attention_id]
            if append_id not in attention.get("dependencies", []):
                raise P21QualificationError(
                    f"{attention_id} does not depend on KV append"
                )
            expected_version = step + 1
            for suffix in ("k", "v"):
                value_id = f"{REQUEST_ID}.kv.l{layer}.{suffix}"
                validated = {
                    (str(item["value_id"]), int(item["version"]))
                    for item in attention.get("validated_input_versions", [])
                    if isinstance(item, Mapping)
                }
                if (value_id, expected_version) not in validated:
                    raise P21QualificationError(
                        f"{attention_id} did not consume {value_id} v{expected_version}"
                    )
            if int(_timing(attention)["start_time_fs"]) < int(
                _timing(append)["completion_time_fs"]
            ):
                raise P21QualificationError(f"{attention_id} starts before KV append")

    for step in range(1, 4):
        sampling_id = f"task.{REQUEST_ID}.decode.s{step - 1}.sampling"
        embedding_id = f"task.{REQUEST_ID}.decode.s{step}.embedding"
        embedding = tasks[embedding_id]
        if sampling_id not in embedding.get("dependencies", []):
            raise P21QualificationError(
                f"{embedding_id} does not depend on prior sampling"
            )
        expected_token = f"{REQUEST_ID}.token.{step - 1}"
        if not any(
            isinstance(item, Mapping)
            and item.get("value_id") == expected_token
            and int(item.get("version", -1)) == 1
            for item in embedding.get("validated_input_versions", [])
        ):
            raise P21QualificationError(
                f"{embedding_id} did not consume the prior sampled token"
            )

    final_versions = _mapping(
        online.get("final_versions"), "online dispatch lacks final versions"
    )
    kv_final_versions: dict[str, int] = {}
    for layer in range(layers):
        for suffix in ("k", "v"):
            value_id = f"{REQUEST_ID}.kv.l{layer}.{suffix}"
            version = int(final_versions.get(value_id, -1))
            if version != 4:
                raise P21QualificationError(f"{value_id} final version is not 4")
            kv_final_versions[value_id] = version
    return {
        "commit_count": len(commits),
        "input_version_checks": int(online["version_checks"]),
        "all_inputs_validated": True,
        "all_outputs_committed_at_backend_completion": True,
        "autoregressive_sampling_to_next_embedding": True,
        "kv_append_before_attention": True,
        "kv_final_versions": kv_final_versions,
    }


def summarize_leg(run_dir: Path, layers: int) -> dict[str, object]:
    execution = _load(run_dir / "execution_graph.json")
    online = _load(run_dir / "online_dispatch.json")
    memory_map = _load(run_dir / "global_memory_map.json")
    metrics = _load(run_dir / "metrics.json")
    provenance = _load(run_dir / "provenance.json")
    raw_tasks = execution.get("tasks")
    expected_counts = _expected_counts(layers)
    expected_total = 20 + 48 * layers
    if not isinstance(raw_tasks, list) or len(raw_tasks) != expected_total:
        raise P21QualificationError(
            f"expected {expected_total} tasks, got "
            f"{len(raw_tasks) if isinstance(raw_tasks, list) else 'invalid'}"
        )
    tasks = {
        str(item["task_id"]): item for item in raw_tasks if isinstance(item, Mapping)
    }
    counts = Counter(str(item["op"]) for item in tasks.values())
    if len(tasks) != expected_total or counts != expected_counts:
        raise P21QualificationError("P21 task/operator coverage differs from graph")
    _validate_dependencies_and_resource(tasks)

    gpu_tasks = [task for task in tasks.values() if str(task["op"]) in GPU_OPERATORS]
    runtime_tasks = [
        task for task in tasks.values() if str(task["op"]) in RUNTIME_MEMORY_OPERATORS
    ]
    control_tasks = [
        task for task in tasks.values() if str(task["op"]) in HOST_CONTROL_OPERATORS
    ]
    expected_trace = 16 + 44 * layers
    expected_runtime = 4 + 4 * layers
    if (
        len(gpu_tasks) != expected_trace
        or len(runtime_tasks) != 2 + 4 * layers
        or len(control_tasks) != 2
    ):
        raise P21QualificationError("P21 Backend partition is incomplete")

    global_pa = _validate_global_pa(memory_map, expected_trace, expected_runtime)
    gpu_records = [_validate_gpu_task(task) for task in gpu_tasks]
    runtime_records = [
        _validate_runtime_memory(run_dir, task) for task in runtime_tasks
    ]
    control_records = [_validate_control(task) for task in control_tasks]
    versions = _validate_versions(tasks, online, layers)

    boundary = _mapping(
        online.get("performance_boundary"), "performance boundary is missing"
    )
    if (
        int(online.get("backend_dispatch_count", -1)) != expected_total
        or int(boundary.get("included_task_count", -1)) != expected_total - 2
        or int(boundary.get("excluded_task_count", -1)) != 2
        or metrics.get("performance_claim_allowed") is not False
    ):
        raise P21QualificationError("performance/control boundary is invalid")

    gpu_records.sort(key=lambda item: item["task_id"])
    runtime_records.sort(key=lambda item: item["task_id"])
    control_records.sort(key=lambda item: item["task_id"])
    step_completion_fs = [
        int(
            _timing(tasks[f"task.{REQUEST_ID}.decode.s{step}.sampling"])[
                "completion_time_fs"
            ]
        )
        for step in range(4)
    ]
    return {
        "run_dir": str(run_dir.resolve()),
        "simulation_input_key": provenance.get("simulation_input_key"),
        "simulator_revision": provenance.get("simulator_revision"),
        "layers": layers,
        "task_count": expected_total,
        "operator_counts": dict(sorted(counts.items())),
        "backend_dispatch_count": int(online["backend_dispatch_count"]),
        "makespan_fs": int(metrics["makespan_fs"]),
        "step_completion_fs": step_completion_fs,
        "performance_boundary": dict(boundary),
        "global_pa": global_pa,
        "gpu_instruction_request_cycle": gpu_records,
        "runtime_memory": runtime_records,
        "host_control": control_records,
        "version_causality": versions,
        "dependency_causality": {"all_dependencies_complete_before_start": True},
        "resource_causality": {
            "resource_id": "gpu0",
            "all_tasks_non_overlapping": True,
        },
    }


def summarize(
    first_run_dir: Path, second_run_dir: Path, layers: int
) -> dict[str, object]:
    legs = [
        summarize_leg(first_run_dir, layers),
        summarize_leg(second_run_dir, layers),
    ]
    comparable = (
        "simulation_input_key",
        "layers",
        "task_count",
        "operator_counts",
        "backend_dispatch_count",
        "makespan_fs",
        "step_completion_fs",
        "performance_boundary",
        "global_pa",
        "gpu_instruction_request_cycle",
        "runtime_memory",
        "host_control",
        "version_causality",
        "dependency_causality",
        "resource_causality",
    )
    mismatches = [key for key in comparable if legs[0][key] != legs[1][key]]
    if mismatches:
        raise P21QualificationError(f"P21 double-run mismatch: {mismatches}")
    return {
        "schema_version": "hetero-p21-decode-trace-timeline-qualification/v1",
        "status": "passed",
        "performance_eligible": False,
        "claim_boundary": (
            "P21 replaces P20 GPU tiled contracts with shape-locked real SM86 "
            "instruction Traces captured on an RTX4090 and completes every GPU "
            "parent/child request through range-rebased Ramulator2. The capture/replay "
            "and uncalibrated runtime contracts do not qualify hardware performance."
        ),
        "fixed_scope": {
            "model": "TinyLlama-1.1B",
            "checkpoint_revision": ("fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
            "phase": "continuous_decode",
            "layers": layers,
            "batch_size": 1,
            "initial_context_length": 16,
            "q_len_per_step": 1,
            "kv_lengths": [17, 18, 19, 20],
            "generated_tokens": 4,
            "dtype": "fp16",
            "physical_capture_gpu": "RTX4090_SM89",
            "executed_binary_and_replay_isa": "SM86",
        },
        "double_run_deterministic": True,
        "invariants": {
            "shape_locked_trace_per_decode_step": True,
            "dependency_causality": True,
            "gpu0_resource_nonoverlap": True,
            "global_pa_nonoverlap": True,
            "semantic_trace_binding": True,
            "one_ramulator2_per_gpu_or_runtime_task": True,
            "parent_child_durable_conservation": True,
            "zero_atlas_requests": True,
            "zero_inflight_at_task_completion": True,
            "autoregressive_sampling_to_next_embedding": True,
            "kv_append_before_attention": True,
            "kv_versions_0_to_4_monotonic": True,
            "version_commits_at_backend_completion": True,
        },
        "legs": legs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg1", type=Path, required=True)
    parser.add_argument("--leg2", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(1, 22), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = summarize(args.leg1, args.leg2, args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
