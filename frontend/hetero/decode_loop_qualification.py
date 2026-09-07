"""Fail-closed functional qualification for the P20 four-token Decode loop."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

from .request_cycle_qualification import _load, _read_stream, _require


def _versions(task: Mapping[str, object], field: str) -> dict[str, int]:
    values = task.get(field)
    _require(isinstance(values, list), f"{field} must be an array")
    result: dict[str, int] = {}
    for value in values:
        _require(isinstance(value, Mapping), f"{field} entries must be objects")
        result[str(value["value_id"])] = int(value["version"])
    return result


def audit_p20_decode_loop_run(
    run_dir: Path, expected_layers: int
) -> dict[str, object]:
    """Audit one BS=1, initial-KV=16, four-token Decode run."""

    run_dir = run_dir.resolve()
    generated_tokens = 4
    expected_tasks = 4 + generated_tokens * (4 + 12 * expected_layers)
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
    _require(len(tasks) == expected_tasks, "P20 Decode task count mismatch")
    by_id: dict[str, Mapping[str, object]] = {}
    task_timings: dict[str, Mapping[str, object]] = {}
    for raw in tasks:
        _require(isinstance(raw, Mapping), "execution task must be an object")
        task_id = str(raw.get("task_id", ""))
        _require(bool(task_id) and task_id not in by_id, "task ids must be unique")
        by_id[task_id] = raw
        timing = raw.get("timing")
        _require(isinstance(timing, Mapping), f"{task_id} has no timing")
        task_timings[task_id] = timing
        _require(
            raw.get("compiled_artifact", {}).get("kind")
            == "request_tiled_cycle_contract",
            f"{task_id} is not backed by the P20 request cycle contract",
        )
        fidelity = raw.get("fidelity")
        _require(isinstance(fidelity, Mapping), f"{task_id} has no fidelity record")
        _require(
            fidelity.get("compute_fidelity") == "tiled_cycle_contract_unqualified"
            and fidelity.get("performance_eligible") is False
            and float(fidelity.get("trace_coverage", -1)) == 0.0,
            f"{task_id} overstates compute fidelity",
        )
        start = int(timing["start_cycle"])
        for dependency in raw.get("dependencies", []):
            dependency_id = str(dependency)
            _require(
                dependency_id in task_timings,
                f"unknown or future dependency {dependency_id}",
            )
            _require(
                int(task_timings[dependency_id]["completion_cycle"]) <= start,
                f"dependency completes after consumer starts: {task_id}",
            )

    request_ids = {str(task.get("request_id")) for task in tasks}
    _require(len(request_ids) == 1, "P20 qualification requires exactly one request")
    request_id = next(iter(request_ids))
    _require(
        {str(task.get("phase")) for task in tasks} == {"control", "decode"},
        "P20 graph contains an unexpected phase",
    )
    _require(
        {str(task.get("device_id")) for task in tasks} == {"gpu0"},
        "P20 GPU-only qualification observed another device",
    )
    ordered = sorted(
        task_timings.items(), key=lambda item: (int(item[1]["start_cycle"]), item[0])
    )
    for (_, left), (_, right) in pairwise(ordered):
        _require(
            int(left["completion_cycle"]) <= int(right["start_cycle"]),
            "gpu0 resource intervals overlap",
        )

    _require(coverage.get("all_tasks_covered") is True, "cycle coverage incomplete")
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
        memory.get("one_live_timing_owner") is True
        and int(memory.get("instances", -1)) == 1,
        "expected exactly one live Ramulator2 timing owner",
    )
    accepted = int(memory.get("accepted_parent_ids", -1))
    _require(accepted > 0, "no live-memory parent request was accepted")
    for field in ("observed_completion_ids", "completed", "durable_completed"):
        _require(int(memory.get(field, -1)) == accepted, f"{field} mismatch")
    _require(
        int(memory.get("children_sent", -1))
        == int(memory.get("children_completed", -2)),
        "child conservation mismatch",
    )
    _require(int(memory.get("outstanding", -1)) == 0, "requests remain in flight")
    initiators = memory.get("initiators")
    _require(isinstance(initiators, Mapping), "initiator statistics missing")
    atlas = initiators.get("atlas0.compute")
    gpu = initiators.get("gpu0")
    _require(
        isinstance(atlas, Mapping) and isinstance(gpu, Mapping),
        "initiator records missing",
    )
    _require(int(atlas.get("parents", -1)) == 0, "ATLAS request observed")
    _require(int(gpu.get("parents", -1)) == accepted, "GPU parent mismatch")

    ranges = memory_map.get("ranges")
    _require(isinstance(ranges, list) and ranges, "Global PA map is empty")
    _require(memory_map.get("non_overlapping") is True, "Global PA ranges overlap")
    allocations = {
        str(item["value_id"]): item for item in ranges if isinstance(item, Mapping)
    }
    _require(len(allocations) == len(ranges), "Global PA value ids are not unique")
    issues = [record for record in stream if record.get("event") == "request_issue"]
    completions = [
        record for record in stream if record.get("event") == "request_completion"
    ]
    issue_ids = {int(record["parent_id"]) for record in issues}
    completion_by_id = {int(record["parent_id"]): record for record in completions}
    _require(
        len(issues) == len(issue_ids) == accepted,
        "stream issue conservation mismatch",
    )
    _require(
        len(completions) == len(completion_by_id) == accepted
        and set(completion_by_id) == issue_ids,
        "stream completion conservation mismatch",
    )
    for issue in issues:
        value_id = str(issue["value_id"])
        allocation = allocations.get(value_id)
        _require(isinstance(allocation, Mapping), f"no Global PA for {value_id}")
        address = int(issue["global_address"])
        end = address + int(issue["size_bytes"])
        _require(
            int(allocation["base_address"]) <= address
            and end <= int(allocation["end_address_exclusive"]),
            f"request escapes Global PA allocation: {value_id}",
        )
        timing = task_timings[str(issue["task_id"])]
        completion = completion_by_id[int(issue["parent_id"])]
        _require(
            int(issue["issue_cycle"]) >= int(timing["start_cycle"])
            and int(completion["completion_cycle"])
            <= int(timing["completion_cycle"]),
            f"request completion escapes task interval: {issue['task_id']}",
        )

    requests = lifecycle_bundle.get("requests")
    _require(
        lifecycle_bundle.get("schema_version")
        == "hetero-decode-kv-lifecycle-bundle/v2"
        and lifecycle_bundle.get("all_requests_valid") is True
        and isinstance(requests, list)
        and len(requests) == 1,
        "Decode loop lifecycle bundle is invalid",
    )
    lifecycle = requests[0]
    _require(isinstance(lifecycle, Mapping), "lifecycle record must be an object")
    _require(
        lifecycle.get("schema_version") == "hetero-decode-kv-lifecycle/v2"
        and lifecycle.get("execution_scope") == "decode_loop"
        and int(lifecycle.get("layer_count", -1)) == expected_layers
        and int(lifecycle.get("generated_tokens", -1)) == generated_tokens
        and int(lifecycle.get("initial_kv_length", -1)) == 16
        and int(lifecycle.get("final_kv_length", -1)) == 20
        and lifecycle.get("global_pa_bound") is True
        and lifecycle.get("all_steps_autoregressive") is True,
        "Decode loop lifecycle contract mismatch",
    )
    steps = lifecycle.get("steps")
    _require(
        isinstance(steps, list) and len(steps) == generated_tokens,
        "Decode loop step count mismatch",
    )
    kv_value_ids: set[str] = set()
    previous_sampling_id: str | None = None
    for step_id, step in enumerate(steps):
        _require(isinstance(step, Mapping), "Decode step must be an object")
        _require(
            int(step.get("step_id", -1)) == step_id
            and int(step.get("past_kv_length", -1)) == 16 + step_id
            and int(step.get("attention_kv_length", -1)) == 17 + step_id,
            f"Decode step {step_id} shape mismatch",
        )
        embedding_id = str(step["embedding_task_id"])
        sampling_id = str(step["sampling_task_id"])
        _require(embedding_id in by_id and sampling_id in by_id, "step task missing")
        if step_id == 0:
            _require(
                step.get("token_source") == "external_decode_token",
                "step zero must consume the external token",
            )
        else:
            assert previous_sampling_id is not None
            _require(
                step.get("token_source") == previous_sampling_id,
                f"step {step_id} token source mismatch",
            )
            _require(
                previous_sampling_id in by_id[embedding_id].get("dependencies", [])
                and int(task_timings[previous_sampling_id]["completion_cycle"])
                <= int(task_timings[embedding_id]["start_cycle"]),
                f"step {step_id} starts before prior sampling completes",
            )
            _require(
                str(step["token_input_value_id"])
                in by_id[embedding_id].get("read_values", []),
                f"step {step_id} does not read the prior token value",
            )
        previous_sampling_id = sampling_id

        layers = step.get("layers")
        _require(
            isinstance(layers, list) and len(layers) == expected_layers,
            f"step {step_id} lifecycle layer count mismatch",
        )
        step_kv_ids: set[str] = set()
        for layer in layers:
            _require(isinstance(layer, Mapping), "layer lifecycle must be an object")
            append_id = str(layer["append_task_id"])
            attention_id = str(layer["attention_task_id"])
            _require(
                int(task_timings[append_id]["completion_cycle"])
                <= int(task_timings[attention_id]["start_cycle"]),
                "attention begins before KV append commits",
            )
            append_inputs = _versions(by_id[append_id], "input_values")
            append_outputs = _versions(by_id[append_id], "output_values")
            attention_inputs = _versions(by_id[attention_id], "input_values")
            values = layer.get("values")
            _require(
                isinstance(values, list) and len(values) == 2,
                "layer must own K and V",
            )
            for value in values:
                _require(isinstance(value, Mapping), "KV value must be an object")
                value_id = str(value["value_id"])
                step_kv_ids.add(value_id)
                _require(
                    append_inputs.get(value_id) == step_id
                    and append_outputs.get(value_id) == step_id + 1
                    and attention_inputs.get(value_id) == step_id + 1,
                    f"KV version causality mismatch: {value_id}, step {step_id}",
                )
                expected_offset = (16 + step_id) * 512
                append_range = value.get("append_range")
                global_pa = value.get("global_pa")
                _require(
                    isinstance(append_range, Mapping)
                    and int(append_range.get("offset_bytes", -1)) == expected_offset
                    and int(append_range.get("size_bytes", -1)) == 512,
                    f"KV append range mismatch: {value_id}, step {step_id}",
                )
                _require(
                    isinstance(global_pa, Mapping)
                    and int(global_pa.get("append_address", -1))
                    == int(global_pa.get("base_address", -2)) + expected_offset,
                    f"KV append Global PA mismatch: {value_id}, step {step_id}",
                )
        if not kv_value_ids:
            kv_value_ids = step_kv_ids
        _require(step_kv_ids == kv_value_ids, "KV identity changes across steps")
    _require(
        len(kv_value_ids) == 2 * expected_layers,
        "KV values alias across layers",
    )

    final_versions = runtime.get("final_versions")
    _require(isinstance(final_versions, Mapping), "runtime final versions missing")
    _require(
        all(int(final_versions.get(value_id, -1)) == 4 for value_id in kv_value_ids),
        "KV final version mismatch",
    )
    finish_id = f"task.{request_id}.request_finish"
    release_id = f"task.{request_id}.kv_release"
    assert previous_sampling_id is not None
    _require(
        int(task_timings[previous_sampling_id]["completion_cycle"])
        <= int(task_timings[finish_id]["start_cycle"])
        <= int(task_timings[release_id]["start_cycle"]),
        "final sampling, finish and release order mismatch",
    )

    metric_requests = metrics.get("requests")
    _require(
        isinstance(metric_requests, list) and len(metric_requests) == 1,
        "metrics request record missing",
    )
    metric_request = metric_requests[0]
    _require(isinstance(metric_request, Mapping), "metrics request must be an object")
    _require(
        int(metric_request.get("generated_length", -1)) == 4
        and int(metric_request.get("final_committed_kv_len", -1)) == 20
        and isinstance(metric_request.get("itl_fs"), list)
        and len(metric_request["itl_fs"]) == 3,
        "multi-token request metrics mismatch",
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
    _require(isinstance(stream_ref, Mapping), "memory stream reference missing")
    return {
        "run_dir": str(run_dir),
        "request_id": request_id,
        "expected_layers": expected_layers,
        "generated_tokens": generated_tokens,
        "task_count": expected_tasks,
        "operator_counts": dict(
            sorted(Counter(str(task["op"]) for task in tasks).items())
        ),
        "dependency_causality": True,
        "gpu0_resource_nonoverlap": True,
        "autoregressive_token_chain": True,
        "global_pa_nonoverlap": True,
        "all_memory_requests_within_value_allocation": True,
        "kv_append_before_attention": True,
        "kv_versions_monotonic": True,
        "kv_final_versions": {value_id: 4 for value_id in sorted(kv_value_ids)},
        "initial_kv_length": 16,
        "final_kv_length": 20,
        "accepted_parent_requests": accepted,
        "children_sent": int(memory["children_sent"]),
        "children_completed": int(memory["children_completed"]),
        "outstanding": int(memory["outstanding"]),
        "ramulator2_instances": int(memory["instances"]),
        "atlas_parent_requests": int(atlas["parents"]),
        "gpu_cycles": int(memory["gpu_cycles"]),
        "ramulator2_cycles": int(memory["clock"]),
        "makespan_fs": int(metrics["makespan_fs"]),
        "ttft_fs_observed_unqualified": int(metric_request["ttft_fs"]),
        "itl_fs_observed_unqualified": list(metric_request["itl_fs"]),
        "stream_event_count": int(stream_ref["event_count"]),
        "stream_uncompressed_sha256": str(stream_ref["uncompressed_sha256"]),
        "compute_fidelity": "tiled_cycle_contract_unqualified",
        "accel_sim_instruction_trace_coverage": 0.0,
        "performance_claim_allowed": False,
    }


def qualify_p20_decode_loop_pair(
    leg1: Path, leg2: Path, expected_layers: int
) -> dict[str, object]:
    first = audit_p20_decode_loop_run(leg1, expected_layers)
    second = audit_p20_decode_loop_run(leg2, expected_layers)
    deterministic_fields = (
        "request_id",
        "expected_layers",
        "generated_tokens",
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
        "ttft_fs_observed_unqualified",
        "itl_fs_observed_unqualified",
        "stream_event_count",
        "stream_uncompressed_sha256",
    )
    mismatches = {
        field: [first[field], second[field]]
        for field in deterministic_fields
        if first[field] != second[field]
    }
    _require(not mismatches, f"P20 double-run mismatch: {mismatches}")
    return {
        "schema_version": "hetero-p20-decode-loop-qualification/v1",
        "status": "passed",
        "functional_scope": "single_request_four_token_autoregressive_decode_loop",
        "expected_layers": expected_layers,
        "shape_contract": {
            "model": "TinyLlama-1.1B",
            "batch_size": 1,
            "initial_context_length": 16,
            "generated_tokens": 4,
            "q_len_per_step": 1,
            "final_kv_length": 20,
            "dtype": "fp16",
        },
        "double_run_deterministic": True,
        "deterministic_fields": list(deterministic_fields),
        "legs": [first, second],
        "invariants": {
            "dependency_causality": True,
            "gpu0_resource_nonoverlap": True,
            "autoregressive_sampling_to_next_embedding": True,
            "global_pa_nonoverlap": True,
            "all_memory_requests_within_value_allocation": True,
            "parent_child_durable_conservation": True,
            "zero_inflight_at_finish": True,
            "one_live_ramulator2": True,
            "zero_atlas_requests": True,
            "kv_append_before_attention": True,
            "kv_versions_0_to_4_monotonic": True,
            "request_finish_after_final_sampling": True,
            "kv_release_after_request_finish": True,
        },
        "claim_boundary": {
            "compute_fidelity": "tiled_cycle_contract_unqualified",
            "accel_sim_instruction_trace_coverage": 0.0,
            "performance_claim_allowed": False,
            "reason": (
                "P20 qualifies functional multi-token graph, token and KV causality, "
                "Global PA and live Ramulator2 request timing. GPU compute cycles "
                "remain uncalibrated and cannot support a performance claim."
            ),
        },
    }
