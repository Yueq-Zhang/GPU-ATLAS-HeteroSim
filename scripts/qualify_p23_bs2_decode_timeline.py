#!/usr/bin/env python3
"""Double-run qualification for the P23 fused-BS=2 Decode timeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

from frontend.hetero.p23_batched_decode import (
    BATCH_REQUEST_ID,
    GPU_OPERATORS,
    MEMBER_REQUEST_IDS,
    RUNTIME_OPERATORS,
    run_timeline,
)


class P23QualificationError(RuntimeError):
    """Raised when one global-timeline invariant is broken."""


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P23QualificationError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P23QualificationError(message)
    return value


def _runtime_audit(run_dir: Path, task_id: str) -> dict[str, object]:
    return _load(
        run_dir / "backend_runs" / "runtime" / task_id / "runtime_request_audit.json"
    )


def _validate_gpu_task(task: Mapping[str, object]) -> dict[str, object]:
    artifact = _mapping(task.get("compiled_artifact"), "GPU task lacks artifact")
    fidelity = _mapping(task.get("fidelity"), "GPU task lacks fidelity")
    stats = _mapping(task.get("backend_statistics"), "GPU task lacks statistics")
    external = _mapping(
        stats.get("external_memory_stats"), "GPU task lacks external-memory stats"
    )
    operator = str(task["op"])
    expected_artifact = (
        f"tinyllama.1_1b.layer0.{operator}.decode_step.bs2.ctx16.q1.kv17."
        "fp16.sm86.accel_sim_v2.shared_hbdram_range_rebase_v1"
    )
    parents = int(external.get("gpu_parents", -1))
    children = int(external.get("gpu_children", -1))
    checks = {
        "artifact_kind": artifact.get("kind") == "accel_sim_trace",
        "artifact_identity": artifact.get("operator_artifact_id") == expected_artifact,
        "request_cycle_ready": artifact.get("request_cycle_ready") is True,
        "trace_coverage": fidelity.get("trace_coverage") == 1.0,
        "performance_closed": fidelity.get("performance_eligible") is False,
        "positive_cycles": int(stats.get("cycles", 0)) > 0,
        "positive_instructions": int(stats.get("instructions", 0)) > 0,
        "one_ramulator2": int(external.get("instances", -1)) == 1,
        "parents_complete": parents > 0
        and int(external.get("gpu_completed", -1)) == parents
        and int(external.get("durable_completed", -1)) == parents,
        "children_complete": children > 0
        and int(external.get("children_sent", -1)) == children
        and int(external.get("children_completed", -1)) == children,
        "all_addresses_translated": int(external.get("address_translated", 0)) > 0
        and int(external.get("address_unmapped", -1)) == 0,
        "zero_atlas": all(
            int(external.get(key, -1)) == 0
            for key in ("atlas_parents", "atlas_children", "atlas_completed")
        ),
        "zero_outstanding": int(external.get("outstanding", -1)) == 0,
    }
    if not all(checks.values()):
        raise P23QualificationError(f"{task['task_id']} GPU checks failed: {checks}")
    return {
        "task_id": task["task_id"],
        "operator": operator,
        "cycles": int(stats["cycles"]),
        "instructions": int(stats["instructions"]),
        "parents": parents,
        "children": children,
        "duration_fs": int(task["duration_fs"]),
        "checks": checks,
    }


def _validate_runtime_task(
    run_dir: Path, task: Mapping[str, object]
) -> dict[str, object]:
    artifact = _mapping(task.get("compiled_artifact"), "runtime task lacks artifact")
    operator = str(task["op"])
    if operator in {"request_start", "request_finish"}:
        stats = _mapping(task.get("backend_statistics"), "control task lacks stats")
        live = _mapping(
            stats.get("live_memory_statistics"), "control task lacks live stats"
        )
        if (
            artifact.get("kind") != "host_control_event"
            or (
                live
                and (
                    int(live.get("instances", -1)) != 0
                    or int(live.get("outstanding", -1)) != 0
                )
            )
        ):
            raise P23QualificationError(f"{task['task_id']} control boundary failed")
        return {
            "task_id": task["task_id"],
            "operator": operator,
            "host_control": True,
            "requests": 0,
            "cycles": int(stats["cycles"]),
        }

    audit = _runtime_audit(run_dir, str(task["task_id"]))
    stats = _mapping(audit.get("memory_statistics"), "runtime audit lacks stats")
    backend = _mapping(task.get("backend_statistics"), "runtime task lacks stats")
    embedded = _mapping(
        backend.get("live_memory_statistics"), "runtime task lacks embedded stats"
    )
    initiators = _mapping(stats.get("initiators"), "runtime task lacks initiators")
    gpu = _mapping(initiators.get("gpu0"), "runtime task lacks GPU initiator")
    atlas = _mapping(
        initiators.get("atlas0.compute"), "runtime task lacks ATLAS initiator"
    )
    requests = int(stats.get("request_count", -1))
    reads = int(stats.get("read_request_count", -1))
    writes = int(stats.get("write_request_count", -1))
    checks = {
        "live_artifact": artifact.get("kind") == "runtime_live_ramulator2",
        "request_cycle_ready": artifact.get("request_cycle_ready") is True,
        "uncalibrated": artifact.get("calibrated") is False,
        "audit_embedded_equal": dict(stats) == dict(embedded),
        "one_ramulator2": int(stats.get("instances", -1)) == 1,
        "request_conservation": requests > 0
        and requests == reads + writes
        and int(stats.get("accepted_parent_ids", -1)) == requests
        and int(stats.get("observed_completion_ids", -1)) == requests
        and int(stats.get("completed", -1)) == requests
        and int(stats.get("durable_completed", -1)) == requests,
        "child_conservation": int(stats.get("children_sent", -1))
        == int(stats.get("children_completed", -2)),
        "gpu_only": int(gpu.get("parents", -1)) == requests
        and int(gpu.get("completed", -1)) == requests
        and all(
            int(atlas.get(key, -1)) == 0 for key in ("parents", "children", "completed")
        ),
        "zero_outstanding": int(stats.get("outstanding", -1)) == 0,
    }
    if operator == "kv_append":
        checks["exact_bs2_kv_traffic"] = (
            reads == 32 and writes == 32 and int(stats.get("logical_bytes", -1)) == 4096
        )
    if not all(checks.values()):
        raise P23QualificationError(
            f"{task['task_id']} runtime checks failed: {checks}"
        )
    return {
        "task_id": task["task_id"],
        "operator": operator,
        "host_control": False,
        "requests": requests,
        "reads": reads,
        "writes": writes,
        "logical_bytes": int(stats["logical_bytes"]),
        "cycles": int(backend["cycles"]),
        "checks": checks,
    }


def _validate_dependencies(tasks: list[Mapping[str, object]]) -> dict[str, object]:
    by_id = {str(task["task_id"]): task for task in tasks}
    ordered = sorted(
        tasks, key=lambda task: int(_mapping(task["timing"], "timing")["start_time_fs"])
    )
    previous_completion = -1
    dependency_edges = 0
    for task in ordered:
        timing = _mapping(task["timing"], "task lacks timing")
        if str(task["resource_id"]) != "gpu0":
            raise P23QualificationError("P23 task escaped the gpu0 resource")
        if int(timing["start_time_fs"]) < previous_completion:
            raise P23QualificationError("gpu0 task intervals overlap")
        previous_completion = int(timing["completion_time_fs"])
        for dependency in task["dependencies"]:
            dependency_edges += 1
            dependency_timing = _mapping(
                by_id[str(dependency)]["timing"], "dependency lacks timing"
            )
            if int(dependency_timing["completion_time_fs"]) > int(
                timing["start_time_fs"]
            ):
                raise P23QualificationError(
                    "task launched before dependency completion"
                )
    return {
        "all_tasks_on_gpu0": True,
        "resource_intervals_non_overlapping": True,
        "dependencies_complete_before_launch": True,
        "dependency_edge_count": dependency_edges,
    }


def _validate_global_pa(
    run_dir: Path, tasks: list[Mapping[str, object]]
) -> dict[str, object]:
    memory = _load(run_dir / "global_memory_map.json")
    ranges = sorted(memory["ranges"], key=lambda item: int(item["base_address"]))
    if any(
        int(left["end_address_exclusive"]) > int(right["base_address"])
        for left, right in pairwise(ranges)
    ):
        raise P23QualificationError("Global PA ranges overlap")
    trace_bindings = memory.get("request_cycle_bindings")
    runtime_bindings = memory.get("runtime_task_bindings")
    slices = memory.get("batch_member_kv_slices")
    if (
        not isinstance(trace_bindings, list)
        or len(trace_bindings) != 15
        or not isinstance(runtime_bindings, list)
        or len(runtime_bindings) != 5
        or not isinstance(slices, list)
        or len(slices) != 4
    ):
        raise P23QualificationError("Global PA binding coverage is incomplete")
    for kind in ("k", "v"):
        selected = sorted(
            (item for item in slices if item["kind"] == kind),
            key=lambda item: int(item["member_index"]),
        )
        if [item["request_id"] for item in selected] != list(MEMBER_REQUEST_IDS) or int(
            selected[0]["end_address_exclusive"]
        ) > int(selected[1]["base_address"]):
            raise P23QualificationError("per-request KV Global PA slices overlap")

    append_task = next(task for task in tasks if task["op"] == "kv_append")
    audit = _runtime_audit(run_dir, str(append_task["task_id"]))
    requests = audit["requests"]
    writes = [item for item in requests if item["operation"] == "write"]
    per_kind = {
        kind: [item for item in slices if item["kind"] == kind] for kind in ("k", "v")
    }
    for write in writes:
        semantic = str(write["semantic"])
        member = int(semantic.rsplit("member", 1)[1])
        kind = "k" if ".key_" in semantic else "v"
        target = next(
            item for item in per_kind[kind] if int(item["member_index"]) == member
        )
        address = int(write["global_address"])
        if (
            not int(target["base_address"])
            <= address
            < int(target["end_address_exclusive"])
        ):
            raise P23QualificationError("KV Append write escaped its request slice")
    return {
        "non_overlapping": memory.get("non_overlapping") is True,
        "allocation_count": int(memory["allocation_count"]),
        "trace_binding_count": len(trace_bindings),
        "runtime_binding_count": len(runtime_bindings),
        "member_kv_slice_count": len(slices),
        "kv_append_writes_stay_in_member_slice": True,
        "address_semantics": memory["address_semantics"],
    }


def _validate_versions(
    run_dir: Path, tasks: list[Mapping[str, object]]
) -> dict[str, object]:
    lifecycle = _load(run_dir / "request_lifecycle.json")
    runtime = _load(run_dir / "runtime_result.json")
    append = next(task for task in tasks if task["op"] == "kv_append")
    attention = next(task for task in tasks if task["op"] == "causal_attention")
    sampling = next(task for task in tasks if task["op"] == "sampling")
    append_timing = _mapping(append["timing"], "append timing")
    attention_timing = _mapping(attention["timing"], "attention timing")
    sampling_timing = _mapping(sampling["timing"], "sampling timing")
    validated = {
        (str(item["value_id"]), int(item["version"]))
        for item in attention["validated_input_versions"]
    }
    expected_kv = {
        (f"{BATCH_REQUEST_ID}.kv.l0.k", 2),
        (f"{BATCH_REQUEST_ID}.kv.l0.v", 2),
    }
    records = lifecycle.get("requests")
    commits = lifecycle.get("version_commits")
    checks = {
        "two_requests": isinstance(records, list) and len(records) == 2,
        "both_finished": isinstance(records, list)
        and all(item["final_state"] == "FINISHED" for item in records),
        "kv_16_to_17": isinstance(records, list)
        and all(
            int(item["initial_kv_length"]) == 16
            and int(item["final_committed_kv_length"]) == 17
            for item in records
        ),
        "two_member_commits": isinstance(commits, list)
        and {item["request_id"] for item in commits} == set(MEMBER_REQUEST_IDS),
        "append_before_attention": int(append_timing["completion_time_fs"])
        <= int(attention_timing["start_time_fs"]),
        "attention_reads_committed_kv": expected_kv <= validated,
        "token_ready_at_sampling": isinstance(records, list)
        and all(
            int(item["token_ready_time_fs"])
            == int(sampling_timing["completion_time_fs"])
            for item in records
        ),
        "zero_in_flight": lifecycle.get("zero_in_flight") is True,
        "runtime_version_commits_present": len(runtime.get("version_commits", [])) > 0,
    }
    if not all(checks.values()):
        raise P23QualificationError(f"version/lifecycle checks failed: {checks}")
    return checks


def validate_leg(run_dir: Path) -> dict[str, object]:
    graph = _load(run_dir / "execution_graph.json")
    metrics = _load(run_dir / "metrics.json")
    progress = _load(run_dir / "progress.json")
    provenance = _load(run_dir / "provenance.json")
    tasks = graph.get("tasks")
    if not isinstance(tasks, list):
        raise P23QualificationError("execution graph lacks tasks")
    counts = Counter(str(task["op"]) for task in tasks)
    expected = Counter({operator: 1 for operator in GPU_OPERATORS})
    expected["residual_add"] = 2
    expected.update({operator: 1 for operator in RUNTIME_OPERATORS})
    if counts != expected or len(tasks) != 20:
        raise P23QualificationError(f"P23 task inventory differs: {counts}")
    gpu_tasks = [
        _validate_gpu_task(task) for task in tasks if task["op"] in GPU_OPERATORS
    ]
    runtime_tasks = [
        _validate_runtime_task(run_dir, task)
        for task in tasks
        if task["op"] in RUNTIME_OPERATORS
    ]
    checks = {
        "task_inventory_exact": True,
        "fused_batch_members_exact": all(
            task.get("batch_member_request_ids") == list(MEMBER_REQUEST_IDS)
            and int(task.get("fused_batch_size", 0)) == 2
            for task in tasks
        ),
        "progress_complete": progress.get("status") == "complete"
        and int(progress.get("completed_task_count", -1)) == 20,
        "performance_claim_closed": metrics.get("performance_claim_allowed") is False,
        "remote_sass_only": provenance.get("physical_capture_device")
        == "remote RTX4090"
        and provenance.get("local_rtx3070_sass_used") is False,
    }
    if not all(checks.values()):
        raise P23QualificationError(f"leg checks failed: {checks}")
    return {
        "run_dir": str(run_dir),
        "checks": checks,
        "dependencies": _validate_dependencies(tasks),
        "global_pa": _validate_global_pa(run_dir, tasks),
        "versions": _validate_versions(run_dir, tasks),
        "makespan_fs": int(metrics["makespan_fs"]),
        "gpu_tasks": gpu_tasks,
        "runtime_tasks": runtime_tasks,
        "performance_claim_allowed": False,
    }


def _signature(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "makespan_fs": summary["makespan_fs"],
        "gpu_tasks": [
            {
                key: item[key]
                for key in (
                    "task_id",
                    "operator",
                    "cycles",
                    "instructions",
                    "parents",
                    "children",
                    "duration_fs",
                )
            }
            for item in summary["gpu_tasks"]
        ],
        "runtime_tasks": [
            {
                key: item.get(key)
                for key in (
                    "task_id",
                    "operator",
                    "host_control",
                    "requests",
                    "reads",
                    "writes",
                    "logical_bytes",
                    "cycles",
                )
            }
            for item in summary["runtime_tasks"]
        ],
        "global_pa": summary["global_pa"],
        "versions": summary["versions"],
    }


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _completed_leg_exists(leg_dir: Path) -> bool:
    required = (
        "execution_graph.json",
        "global_memory_map.json",
        "metrics.json",
        "progress.json",
        "provenance.json",
        "request_lifecycle.json",
        "runtime_result.json",
    )
    if not all((leg_dir / name).is_file() for name in required):
        return False
    try:
        progress = _load(leg_dir / "progress.json")
    except (OSError, ValueError, P23QualificationError):
        return False
    return progress.get("status") == "complete" and int(
        progress.get("completed_task_count", -1)
    ) == 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/hetero/experiments/p23_tinyllama_decode1_1layer_bs2_real_trace.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("validation/p23/timeline")
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--resume-completed-legs",
        action="store_true",
        help="reuse individually complete legs and run only missing/incomplete legs",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else project_root / args.config
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    )
    progress_path = output_root / "qualification_progress.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    try:
        for leg_name in ("leg1", "leg2"):
            leg_dir = output_root / leg_name
            _write(
                progress_path,
                {
                    "schema_version": "hetero-p23-bs2-timeline-qualification-progress/v1",
                    "status": "running",
                    "current_leg": leg_name,
                    "completed_legs": len(summaries),
                    "total_legs": 2,
                },
            )
            reuse_leg = args.reuse_existing or (
                args.resume_completed_legs and _completed_leg_exists(leg_dir)
            )
            if not reuse_leg:
                run_timeline(config, leg_dir)
            summaries.append(validate_leg(leg_dir))
        signatures_equal = _signature(summaries[0]) == _signature(summaries[1])
        if not signatures_equal:
            raise P23QualificationError("two P23 timeline legs are not deterministic")
        record = {
            "schema_version": "hetero-p23-bs2-decode-timeline-qualification/v1",
            "status": "passed",
            "qualification_passed": True,
            "performance_claim_allowed": False,
            "scope": {
                "model": "TinyLlama-1.1B",
                "layers": [0],
                "batch_size": 2,
                "request_ids": list(MEMBER_REQUEST_IDS),
                "context_length": 16,
                "q_len": 1,
                "kv_length": 17,
                "decode_steps": 1,
                "gpu_operator_types": 14,
                "gpu_task_instances": 15,
                "kv_append_instances": 1,
            },
            "double_run_signature_equal": signatures_equal,
            "legs": summaries,
            "qualification_boundary": (
                "P23 proves one fused BS=2 single-layer multi-operator Decode "
                "timeline with exact Trace identities, dependency/resource gates, "
                "per-request Global PA KV slices, live KV Append, request completion "
                "and version commits. Performance calibration and persistent DRAM "
                "state across separate kernel replay processes remain open."
            ),
        }
        _write(output_root / "qualification_record.json", record)
        _write(
            progress_path,
            {
                "schema_version": "hetero-p23-bs2-timeline-qualification-progress/v1",
                "status": "complete",
                "current_leg": None,
                "completed_legs": 2,
                "total_legs": 2,
                "qualification_record": str(output_root / "qualification_record.json"),
            },
        )
        print(output_root / "qualification_record.json")
        return 0
    except Exception as error:
        _write(
            progress_path,
            {
                "schema_version": "hetero-p23-bs2-timeline-qualification-progress/v1",
                "status": "failed",
                "completed_legs": len(summaries),
                "total_legs": 2,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
