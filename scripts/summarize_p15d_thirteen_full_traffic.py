#!/usr/bin/env python3
"""Validate two deterministic P15d 13-operator full-value prefill runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


FULL_TRAFFIC_OPERATORS = (
    "attention_norm",
    "qkv_projection",
    "rope",
    "kv_append",
    "causal_attention",
    "output_projection",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
)
DETERMINISTIC_FILES = (
    "request_cycle_trace.json",
    "memory_statistics.json",
    "prefill_artifact_coverage.json",
    "metrics.json",
    "execution_graph.json",
    "global_memory_map.json",
    "trace_manifest.json",
    "event_log.jsonl",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal_hashes(
    run1: Path, run2: Path, filenames: Iterable[str]
) -> dict[str, dict[str, int | str]]:
    results: dict[str, dict[str, int | str]] = {}
    for name in filenames:
        first = run1 / name
        second = run2 / name
        first_hash = _sha256(first)
        second_hash = _sha256(second)
        if first_hash != second_hash:
            raise ValueError(f"P15d deterministic runs differ: {name}")
        first_size = first.stat().st_size
        if first_size != second.stat().st_size:
            raise ValueError(f"P15d deterministic run sizes differ: {name}")
        results[name] = {"sha256": first_hash, "bytes": first_size}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", required=True, type=Path)
    parser.add_argument("--run2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    run1 = args.run1.resolve()
    run2 = args.run2.resolve()
    filenames = list(DETERMINISTIC_FILES)
    stream_name = "request_cycle_trace.jsonl.gz"
    stream_presence = ((run1 / stream_name).is_file(), (run2 / stream_name).is_file())
    if stream_presence[0] != stream_presence[1]:
        raise ValueError("P15d runs disagree on streamed request trace presence")
    if stream_presence[0]:
        filenames.append(stream_name)
    deterministic_files = _require_equal_hashes(run1, run2, filenames)
    request_trace = _load(run1 / "request_cycle_trace.json")
    stream_reference = request_trace.get("memory_trace")
    if stream_presence[0]:
        if (
            not isinstance(stream_reference, dict)
            or stream_reference.get("encoding") != "canonical_jsonl_gzip"
            or stream_reference.get("path") != stream_name
            or stream_reference.get("compressed_sha256")
            != deterministic_files[stream_name]["sha256"]
            or int(stream_reference.get("compressed_bytes", -1))
            != deterministic_files[stream_name]["bytes"]
        ):
            raise ValueError("P15d streamed request trace manifest is inconsistent")
    memory = _load(run1 / "memory_statistics.json")
    coverage = _load(run1 / "prefill_artifact_coverage.json")
    metrics = _load(run1 / "metrics.json")

    full_traffic = set(coverage.get("full_traffic_by_operator", {}))
    if full_traffic != set(FULL_TRAFFIC_OPERATORS):
        raise ValueError("P15d full-traffic operator set does not match the contract")
    parents = int(memory.get("accepted_parent_ids", -1))
    children = int(memory.get("children_sent", -1))
    initiators = memory.get("initiators", {})
    gpu = initiators.get("gpu0", {}) if isinstance(initiators, dict) else {}
    atlas = (
        initiators.get("atlas0.compute", {})
        if isinstance(initiators, dict)
        else {}
    )
    if (
        int(coverage.get("expected_tasks", -1)) != 20
        or int(coverage.get("covered_tasks", -1)) != 20
        or int(coverage.get("full_traffic_tasks", -1)) != 13
        or int(coverage.get("sampled_traffic_tasks", -1)) != 7
        or int(memory.get("instances", -1)) != 1
        or memory.get("one_live_timing_owner") is not True
        or parents <= 0
        or int(memory.get("observed_completion_ids", -1)) != parents
        or int(memory.get("completed", -1)) != parents
        or int(memory.get("durable_completed", -1)) != parents
        or int(gpu.get("parents", -1)) != parents
        or int(gpu.get("completed", -1)) != parents
        or int(atlas.get("parents", -1)) != 0
        or int(atlas.get("completed", -1)) != 0
        or children <= 0
        or int(memory.get("children_completed", -1)) != children
        or int(gpu.get("children", -1)) != children
        or int(atlas.get("children", -1)) != 0
        or int(memory.get("outstanding", -1)) != 0
        or metrics.get("performance_claim_allowed") is not False
    ):
        raise ValueError("P15d request conservation or claim boundary failed")

    output = {
        "schema_version": "hetero-p15d-thirteen-full-traffic-qualification/v1",
        "status": "passed",
        "scope": {
            "model": "TinyLlama-1.1B",
            "phase": "prefill",
            "layer_id": 0,
            "batch_size": 1,
            "context_length": 16,
            "expected_tasks": 20,
            "full_traffic_operators": list(FULL_TRAFFIC_OPERATORS),
        },
        "determinism": {
            "run1": str(run1),
            "run2": str(run2),
            "exact_files": deterministic_files,
        },
        "memory": {
            "parents": parents,
            "full_traffic_parents": int(memory["full_traffic_parents"]),
            "sampled_traffic_parents": int(memory["sampled_traffic_parents"]),
            "children": children,
            "reads": int(memory["reads"]),
            "writes": int(memory["writes"]),
            "logical_bytes": int(memory["logical_bytes"]),
            "internal_bytes": int(memory["internal_bytes"]),
            "gpu_cycles": int(memory["gpu_cycles"]),
            "dram_cycles": int(memory["clock"]),
            "ramulator2_instances": 1,
            "outstanding": 0,
        },
        "timeline": {"makespan_fs": int(metrics["makespan_fs"])},
        "invariants": {
            "deterministic_double_run": True,
            "all_gpu_parents_completed": True,
            "all_gpu_children_completed": True,
            "all_completions_durable": True,
            "zero_atlas_requests": True,
            "one_live_ramulator2": True,
            "zero_outstanding": True,
        },
        "claim_boundary": {
            "performance_claim_allowed": False,
            "compute_path": "prefill_tiled_cycle_contract_unqualified",
            "memory_path": "thirteen_operators_full_value_traffic_other_tasks_sampled",
            "gpu_accel_sim_compute_timeline_integrated": False,
            "global_pa_binding_ready": False,
            "virtual_memory_mode": "identity_untranslated",
            "dram_mapper": "OneLevelInterleave(channel_lowest_bit=0)",
        },
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"P15d 13-operator full-traffic qualification passed: {destination}")


if __name__ == "__main__":
    main()
