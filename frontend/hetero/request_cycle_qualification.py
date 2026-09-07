"""Fail-closed functional qualification for a P19 Decode request-cycle pair."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path


class RequestCycleQualificationError(RuntimeError):
    """Raised when a request-cycle run violates an auditable invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RequestCycleQualificationError(message)


def _load(run_dir: Path, name: str) -> dict[str, object]:
    path = run_dir / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestCycleQualificationError(
            f"failed to load {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RequestCycleQualificationError(f"{path} root must be an object")
    return payload


def _read_stream(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "request_cycle_trace.jsonl.gz"
    records: list[dict[str, object]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise RequestCycleQualificationError(
                        f"{path}:{line_number} must be an object"
                    )
                records.append(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestCycleQualificationError(
            f"failed to read {path}: {error}"
        ) from error
    _require(len(records) >= 2, "request-cycle stream is incomplete")
    _require(records[0].get("event") == "header", "stream header is missing")
    _require(records[-1].get("event") == "footer", "stream footer is missing")
    return records


def audit_p19_decode_run(run_dir: Path, expected_layers: int) -> dict[str, object]:
    """Audit one fixed-shape Decode run without making a performance claim."""

    run_dir = run_dir.resolve()
    expected_tasks = 8 + 12 * expected_layers
    execution = _load(run_dir, "execution_graph.json")
    memory = _load(run_dir, "memory_statistics.json")
    memory_map = _load(run_dir, "global_memory_map.json")
    lifecycle_bundle = _load(run_dir, "decode_kv_lifecycle.json")
    coverage = _load(run_dir, "request_artifact_coverage.json")
    runtime = _load(run_dir, "request_cycle_trace.json")
    metrics = _load(run_dir, "metrics.json")
    provenance = _load(run_dir, "provenance.json")
    stream = _read_stream(run_dir)

    tasks = execution.get("tasks")
    _require(isinstance(tasks, list), "execution graph tasks must be an array")
    _require(len(tasks) == expected_tasks, "Decode task count mismatch")
    by_id: dict[str, Mapping[str, object]] = {}
    for raw in tasks:
        _require(isinstance(raw, Mapping), "execution task must be an object")
        task_id = str(raw.get("task_id", ""))
        _require(bool(task_id) and task_id not in by_id, "task ids must be unique")
        by_id[task_id] = raw
    request_ids = {str(task.get("request_id")) for task in tasks}
    _require(len(request_ids) == 1, "P19 qualification requires exactly one request")
    _require(
        {str(task.get("phase")) for task in tasks} == {"control", "decode"},
        "P19 graph contains an unexpected phase",
    )
    _require(
        {str(task.get("device_id")) for task in tasks} == {"gpu0"},
        "P19 GPU-only qualification observed another device",
    )

    task_timings: dict[str, Mapping[str, object]] = {}
    for task_id, task in by_id.items():
        timing = task.get("timing")
        _require(isinstance(timing, Mapping), f"{task_id} has no timing")
        task_timings[task_id] = timing
        _require(
            task.get("compiled_artifact", {}).get("kind")
            == "request_tiled_cycle_contract",
            f"{task_id} is not backed by the P19 request cycle contract",
        )
        fidelity = task.get("fidelity")
        _require(isinstance(fidelity, Mapping), f"{task_id} has no fidelity record")
        _require(
            fidelity.get("compute_fidelity") == "tiled_cycle_contract_unqualified"
            and fidelity.get("performance_eligible") is False
            and float(fidelity.get("trace_coverage", -1)) == 0.0,
            f"{task_id} overstates compute fidelity",
        )
        start = int(timing["start_cycle"])
        for dependency in task.get("dependencies", []):
            _require(
                dependency in task_timings, f"unknown or future dependency {dependency}"
            )
            _require(
                int(task_timings[str(dependency)]["completion_cycle"]) <= start,
                f"dependency completes after consumer starts: {task_id}",
            )

    ordered = sorted(
        task_timings.items(), key=lambda item: (int(item[1]["start_cycle"]), item[0])
    )
    for (_, left), (_, right) in pairwise(ordered):
        _require(
            int(left["completion_cycle"]) <= int(right["start_cycle"]),
            "gpu0 resource intervals overlap",
        )

    _require(coverage.get("all_tasks_covered") is True, "cycle coverage is incomplete")
    _require(
        int(coverage.get("covered_tasks", -1)) == expected_tasks,
        "coverage count mismatch",
    )
    _require(
        int(coverage.get("analytical_fallback_tasks", -1)) == 0,
        "fallback task observed",
    )
    _require(
        runtime.get("schema_version") == "hetero-request-cycle-runtime/v1",
        "runtime schema mismatch",
    )
    _require(
        int(runtime.get("backend_dispatch_count", -1)) == expected_tasks,
        "dispatch count mismatch",
    )

    _require(
        memory.get("one_live_timing_owner") is True,
        "Ramulator2 timing owner is not unique",
    )
    _require(
        int(memory.get("instances", -1)) == 1,
        "expected exactly one Ramulator2 instance",
    )
    accepted = int(memory.get("accepted_parent_ids", -1))
    _require(accepted > 0, "no live-memory parent request was accepted")
    _require(
        int(memory.get("observed_completion_ids", -1)) == accepted,
        "parent completion mismatch",
    )
    _require(int(memory.get("completed", -1)) == accepted, "completed parent mismatch")
    _require(
        int(memory.get("durable_completed", -1)) == accepted, "durable parent mismatch"
    )
    _require(
        int(memory.get("children_sent", -1))
        == int(memory.get("children_completed", -2)),
        "child conservation mismatch",
    )
    _require(
        int(memory.get("outstanding", -1)) == 0, "memory requests remain in flight"
    )
    initiators = memory.get("initiators")
    _require(isinstance(initiators, Mapping), "initiator statistics are missing")
    atlas = initiators.get("atlas0.compute")
    gpu = initiators.get("gpu0")
    _require(
        isinstance(atlas, Mapping) and isinstance(gpu, Mapping),
        "initiator records are missing",
    )
    _require(
        int(atlas.get("parents", -1)) == 0, "ATLAS request observed in GPU-only P19"
    )
    _require(int(gpu.get("parents", -1)) == accepted, "GPU parent count mismatch")

    ranges = memory_map.get("ranges")
    _require(isinstance(ranges, list) and ranges, "Global PA map is empty")
    _require(memory_map.get("non_overlapping") is True, "Global PA ranges overlap")
    allocations = {
        str(item["value_id"]): item for item in ranges if isinstance(item, Mapping)
    }
    _require(len(allocations) == len(ranges), "Global PA value ids are not unique")
    issue_records = [
        record for record in stream if record.get("event") == "request_issue"
    ]
    completion_records = [
        record for record in stream if record.get("event") == "request_completion"
    ]
    issue_ids = {int(record["parent_id"]) for record in issue_records}
    completion_ids = {int(record["parent_id"]) for record in completion_records}
    _require(
        len(issue_records) == accepted and len(issue_ids) == accepted,
        "stream issue conservation mismatch",
    )
    _require(
        len(completion_records) == accepted and completion_ids == issue_ids,
        "stream completion conservation mismatch",
    )
    completions_by_id = {
        int(record["parent_id"]): record for record in completion_records
    }
    for issue in issue_records:
        value_id = str(issue["value_id"])
        allocation = allocations.get(value_id)
        _require(
            isinstance(allocation, Mapping),
            f"stream value has no Global PA: {value_id}",
        )
        address = int(issue["global_address"])
        end = address + int(issue["size_bytes"])
        _require(
            int(allocation["base_address"]) <= address
            and end <= int(allocation["end_address_exclusive"]),
            f"request escapes Global PA allocation: {value_id}",
        )
        completion = completions_by_id[int(issue["parent_id"])]
        task_timing = task_timings[str(issue["task_id"])]
        _require(
            int(issue["issue_cycle"]) >= int(task_timing["start_cycle"])
            and int(completion["completion_cycle"])
            <= int(task_timing["completion_cycle"]),
            f"memory request completion escapes task interval: {issue['task_id']}",
        )

    requests = lifecycle_bundle.get("requests")
    _require(
        lifecycle_bundle.get("all_requests_valid") is True
        and isinstance(requests, list)
        and len(requests) == 1,
        "Decode lifecycle bundle is invalid",
    )
    lifecycle = requests[0]
    _require(
        isinstance(lifecycle, Mapping), "Decode lifecycle record must be an object"
    )
    _require(
        int(lifecycle.get("layer_count", -1)) == expected_layers
        and int(lifecycle.get("initial_kv_length", -1)) == 16
        and int(lifecycle.get("final_kv_length", -1)) == 17
        and lifecycle.get("global_pa_bound") is True,
        "Decode KV shape or Global PA binding mismatch",
    )
    layers = lifecycle.get("layers")
    _require(
        isinstance(layers, list) and len(layers) == expected_layers,
        "lifecycle layer count mismatch",
    )
    final_versions = runtime.get("final_versions")
    _require(isinstance(final_versions, Mapping), "runtime final versions are missing")
    kv_value_ids: list[str] = []
    for layer in layers:
        _require(isinstance(layer, Mapping), "lifecycle layer must be an object")
        append_id = str(layer["append_task_id"])
        attention_id = str(layer["attention_task_id"])
        _require(
            int(task_timings[append_id]["completion_cycle"])
            <= int(task_timings[attention_id]["start_cycle"]),
            "attention begins before its KV append commits",
        )
        values = layer.get("values")
        _require(
            isinstance(values, list) and len(values) == 2, "layer must own K and V"
        )
        for value in values:
            _require(isinstance(value, Mapping), "KV lifecycle value must be an object")
            value_id = str(value["value_id"])
            kv_value_ids.append(value_id)
            _require(
                int(final_versions.get(value_id, -1)) == 1,
                f"KV version did not commit: {value_id}",
            )
            _require(
                int(value["append_range"]["offset_bytes"]) == 8192
                and int(value["append_range"]["size_bytes"]) == 512,
                f"KV append byte range mismatch: {value_id}",
            )
    _require(
        len(kv_value_ids) == len(set(kv_value_ids)) == 2 * expected_layers,
        "KV values alias across layers",
    )

    _require(
        metrics.get("performance_claim_allowed") is False,
        "metrics permit a performance claim",
    )
    provenance_cycle = provenance.get("request_cycle")
    _require(
        isinstance(provenance_cycle, Mapping)
        and provenance_cycle.get("performance_eligible") is False,
        "provenance permits a performance claim",
    )
    stream_ref = runtime.get("memory_trace")
    _require(isinstance(stream_ref, Mapping), "memory stream reference is missing")
    return {
        "run_dir": str(run_dir),
        "request_id": next(iter(request_ids)),
        "expected_layers": expected_layers,
        "task_count": expected_tasks,
        "operator_counts": dict(
            sorted(Counter(str(task["op"]) for task in tasks).items())
        ),
        "dependency_causality": True,
        "gpu0_resource_nonoverlap": True,
        "global_pa_nonoverlap": True,
        "all_memory_requests_within_value_allocation": True,
        "kv_append_before_attention": True,
        "kv_final_versions": {value_id: 1 for value_id in sorted(kv_value_ids)},
        "initial_kv_length": 16,
        "final_kv_length": 17,
        "accepted_parent_requests": accepted,
        "children_sent": int(memory["children_sent"]),
        "children_completed": int(memory["children_completed"]),
        "outstanding": int(memory["outstanding"]),
        "ramulator2_instances": int(memory["instances"]),
        "atlas_parent_requests": int(atlas["parents"]),
        "gpu_cycles": int(memory["gpu_cycles"]),
        "ramulator2_cycles": int(memory["clock"]),
        "makespan_fs": int(metrics["makespan_fs"]),
        "stream_event_count": int(stream_ref["event_count"]),
        "stream_uncompressed_sha256": str(stream_ref["uncompressed_sha256"]),
        "compute_fidelity": "tiled_cycle_contract_unqualified",
        "accel_sim_instruction_trace_coverage": 0.0,
        "performance_claim_allowed": False,
    }


def qualify_p19_decode_pair(
    leg1: Path, leg2: Path, expected_layers: int
) -> dict[str, object]:
    first = audit_p19_decode_run(leg1, expected_layers)
    second = audit_p19_decode_run(leg2, expected_layers)
    deterministic_fields = (
        "request_id",
        "expected_layers",
        "task_count",
        "operator_counts",
        "kv_final_versions",
        "accepted_parent_requests",
        "children_sent",
        "children_completed",
        "outstanding",
        "ramulator2_instances",
        "atlas_parent_requests",
        "gpu_cycles",
        "ramulator2_cycles",
        "makespan_fs",
        "stream_event_count",
        "stream_uncompressed_sha256",
    )
    mismatches = {
        field: [first[field], second[field]]
        for field in deterministic_fields
        if first[field] != second[field]
    }
    _require(not mismatches, f"P19 double-run mismatch: {mismatches}")
    return {
        "schema_version": "hetero-p19-decode-request-cycle-qualification/v1",
        "status": "passed",
        "functional_scope": "single_request_single_token_decode_step",
        "expected_layers": expected_layers,
        "shape_contract": {
            "batch_size": 1,
            "initial_context_length": 16,
            "q_len": 1,
            "final_kv_length": 17,
            "dtype": "fp16",
        },
        "double_run_deterministic": True,
        "deterministic_fields": list(deterministic_fields),
        "legs": [first, second],
        "invariants": {
            "dependency_causality": True,
            "gpu0_resource_nonoverlap": True,
            "global_pa_nonoverlap": True,
            "all_memory_requests_within_value_allocation": True,
            "parent_child_durable_conservation": True,
            "zero_inflight_at_finish": True,
            "one_live_ramulator2": True,
            "zero_atlas_requests": True,
            "kv_append_before_attention": True,
            "kv_version_commit_before_consumption": True,
        },
        "claim_boundary": {
            "compute_fidelity": "tiled_cycle_contract_unqualified",
            "accel_sim_instruction_trace_coverage": 0.0,
            "performance_claim_allowed": False,
            "reason": (
                "P19 qualifies graph, request-cycle scheduling, KV state, Global PA "
                "and live Ramulator2 causality. It does not calibrate GPU compute time."
            ),
        },
    }
