#!/usr/bin/env python3
"""Validate and summarize the P15 first-batch qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from frontend.hetero.operator_artifact import OperatorArtifactCatalog


GPU_OPERATORS = (
    "attention_norm",
    "qkv_projection",
    "rope",
    "causal_attention",
)
GPU_QUALIFICATION_DIRS = {
    "attention_norm": "accel-sim-rtx3070-tinyllama-attention-norm-prefill-bs1-ctx16",
    "qkv_projection": "accel-sim-rtx3070-tinyllama-qkv-prefill-bs1-ctx16",
    "rope": "accel-sim-rtx3070-tinyllama-rope-prefill-bs1-ctx16",
    "causal_attention": (
        "accel-sim-rtx3070-tinyllama-causal-attention-prefill-bs1-ctx16"
    ),
}
FIRST_BATCH = (*GPU_OPERATORS[:3], "kv_append", GPU_OPERATORS[3])
DETERMINISTIC_FILES = (
    "memory_statistics.json",
    "prefill_artifact_coverage.json",
    "metrics.json",
    "execution_graph.json",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--gpu-qualification-root", required=True, type=Path)
    parser.add_argument("--atlas-qualification", required=True, type=Path)
    parser.add_argument("--prefill-run1", required=True, type=Path)
    parser.add_argument("--prefill-run2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    catalog = OperatorArtifactCatalog.load(args.catalog.resolve())
    coverage = catalog.coverage()
    if not coverage["registration_complete"]:
        raise ValueError(f"first-batch registration is incomplete: {coverage}")
    if coverage["request_cycle_coverage_complete"]:
        raise ValueError("P15 first batch must not claim request-cycle trace readiness")

    gpu_records: dict[str, dict[str, Any]] = {}
    gpu_root = args.gpu_qualification_root.resolve()
    for operator in GPU_OPERATORS:
        path = gpu_root / GPU_QUALIFICATION_DIRS[operator] / "qualification_record.json"
        record = _load(path)
        if (
            record.get("schema_version") != "hetero-accel-sim-qualification/v1"
            or record.get("status") != "passed"
            or record.get("replay_safety_qualified") is not False
        ):
            raise ValueError(f"invalid GPU qualification: {path}")
        cycles = record["comparison"]["gpu_tot_sim_cycle"]
        instructions = record["comparison"]["gpu_tot_sim_insn"]
        if len(set(cycles)) != 1 or len(set(instructions)) != 1:
            raise ValueError(f"non-deterministic GPU qualification: {operator}")
        gpu_records[operator] = {
            "trace_id": record["trace_id"],
            "cycles": int(cycles[0]),
            "instructions": int(instructions[0]),
            "qualification_record": str(path),
            "replay_safety_qualified": False,
        }

    atlas_path = args.atlas_qualification.resolve()
    atlas = _load(atlas_path)
    if (
        atlas.get("schema_version") != "hetero-atlas-qualification/v1"
        or atlas.get("status") != "passed"
    ):
        raise ValueError("invalid ATLAS qualification record")
    atlas_cycles = atlas["comparison"]["cycles"]
    atlas_energy = atlas["comparison"]["energy_j"]
    if len(set(atlas_cycles)) != 1 or len(set(atlas_energy)) != 1:
        raise ValueError("non-deterministic ATLAS qualification")

    run1 = args.prefill_run1.resolve()
    run2 = args.prefill_run2.resolve()
    deterministic_hashes: dict[str, str] = {}
    for name in DETERMINISTIC_FILES:
        first = run1 / name
        second = run2 / name
        first_hash = _sha256(first)
        if first_hash != _sha256(second):
            raise ValueError(f"P15 full-traffic runs differ: {name}")
        deterministic_hashes[name] = first_hash
    memory = _load(run1 / "memory_statistics.json")
    prefill_coverage = _load(run1 / "prefill_artifact_coverage.json")
    metrics = _load(run1 / "metrics.json")
    required_full = set(FIRST_BATCH)
    if set(prefill_coverage["full_traffic_by_operator"]) != required_full:
        raise ValueError("full-traffic operator set does not match first batch")
    if (
        int(prefill_coverage["full_traffic_tasks"]) != 5
        or int(prefill_coverage["sampled_traffic_tasks"]) != 15
        or int(memory["instances"]) != 1
        or int(memory["accepted_parent_ids"])
        != int(memory["observed_completion_ids"])
        or int(memory["outstanding"]) != 0
        or int(memory["full_traffic_parents"]) <= 0
        or metrics.get("performance_claim_allowed") is not False
    ):
        raise ValueError("P15 request-cycle conservation or claim boundary failed")

    output = {
        "schema_version": "hetero-p15-first-batch-qualification/v1",
        "status": "passed",
        "scope": {
            "model": "TinyLlama-1.1B",
            "phase": "prefill",
            "batch_size": 1,
            "context_length": 16,
            "layer_count": 1,
            "operators": list(FIRST_BATCH),
        },
        "artifact_catalog": {
            "path": str(catalog.source_path),
            "coverage": coverage,
        },
        "gpu_standalone": gpu_records,
        "atlas_qkv_standalone": {
            "cycles": int(atlas_cycles[0]),
            "energy_j": float(atlas_energy[0]),
            "qualification_record": str(atlas_path),
        },
        "selective_full_traffic_prefill": {
            "deterministic_files": deterministic_hashes,
            "full_traffic_tasks": int(prefill_coverage["full_traffic_tasks"]),
            "sampled_traffic_tasks": int(prefill_coverage["sampled_traffic_tasks"]),
            "parents": int(memory["accepted_parent_ids"]),
            "full_traffic_parents": int(memory["full_traffic_parents"]),
            "sampled_traffic_parents": int(memory["sampled_traffic_parents"]),
            "logical_bytes": int(memory["logical_bytes"]),
            "internal_bytes": int(memory["internal_bytes"]),
            "gpu_cycles": int(memory["gpu_cycles"]),
            "dram_cycles": int(memory["clock"]),
            "ramulator2_instances": int(memory["instances"]),
            "outstanding": int(memory["outstanding"]),
            "run1": str(run1),
            "run2": str(run2),
        },
        "claim_boundary": {
            "performance_claim_allowed": False,
            "compute_path": "standalone_real_traces_plus_prefill_tiled_contract",
            "memory_path": "first_batch_full_value_transactions_other_tasks_sampled",
            "request_cycle_trace_stall_resume_integrated": False,
            "context1024_qualified": False,
            "all_operator_coverage": False,
        },
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"P15 first-batch qualification passed: {destination}")


if __name__ == "__main__":
    main()
