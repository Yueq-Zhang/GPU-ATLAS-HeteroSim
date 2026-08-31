#!/usr/bin/env python3
"""Strictly validate a catalog of request-cycle-ready range-rebased GPU Artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from frontend.hetero.operator_artifact import OperatorArtifactCatalog


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _identical_pair(value: object, field: str) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2 or value[0] != value[1]:
        raise ValueError(f"{field} must contain two identical runs")
    return value[0], value[1]


def _qualification_paths(path: Path) -> dict[str, Path]:
    payload = _load(path)
    if payload.get("schema_version") != "hetero-coupled-qualification-map/v1":
        raise ValueError("invalid qualification map schema_version")
    records = _mapping(payload.get("records"), "records")
    result: dict[str, Path] = {}
    for operator, value in records.items():
        if not isinstance(operator, str) or not operator:
            raise ValueError("qualification operator names must be non-empty")
        if not isinstance(value, str) or not value:
            raise ValueError(f"qualification path must be a string: {operator}")
        record = Path(value)
        result[operator] = (
            record.resolve() if record.is_absolute() else (path.parent / record).resolve()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--qualification-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    catalog = OperatorArtifactCatalog.load(args.catalog.resolve())
    operators = tuple(catalog.required_operators)
    if not operators or len(set(operators)) != len(operators):
        raise ValueError("catalog required_operators must be non-empty and unique")
    by_operator = {
        artifact.compatibility_key.operator: artifact for artifact in catalog.artifacts
    }
    if set(by_operator) != set(operators) or len(catalog.artifacts) != len(operators):
        raise ValueError("catalog must contain exactly one Artifact per required operator")
    qualification_paths = _qualification_paths(args.qualification_map.resolve())
    if set(qualification_paths) != set(operators):
        raise ValueError("qualification map must exactly cover the catalog operators")

    first = catalog.artifacts[0].compatibility_key
    records: dict[str, dict[str, object]] = {}
    totals = {
        "gpu_parents": 0,
        "gpu_children": 0,
        "address_translated": 0,
        "logical_bytes": 0,
        "internal_bytes": 0,
    }
    suffix = ".shared_hbdram_range_rebase_v1"
    for operator in operators:
        artifact = by_operator[operator]
        key = artifact.compatibility_key
        if (
            key.model_spec_name != first.model_spec_name
            or key.checkpoint_revision != first.checkpoint_revision
            or key.phase != first.phase
            or key.layer_id != first.layer_id
            or key.batch_size != first.batch_size
            or key.context_length != first.context_length
            or key.dtype != first.dtype
        ):
            raise ValueError(f"catalog scope mismatch: {operator}")
        backend = _mapping(artifact.payload["backend"], "backend")
        execution = _mapping(
            artifact.payload["execution_contract"], "execution_contract"
        )
        address = _mapping(artifact.payload["address_contract"], "address_contract")
        qualification = _mapping(artifact.payload["qualification"], "qualification")
        if (
            backend.get("kind") != "accel_sim"
            or not artifact.compute_memory_coupled
            or execution.get("memory_traffic") != "full_instruction_trace"
            or execution.get("supports_stall_resume") is not True
            or execution.get("global_pa_binding_ready") is not True
            or not artifact.request_cycle_ready
            or qualification.get("performance_eligible") is not False
            or address.get("virtual_memory_mode") != "range_rebase"
            or address.get("global_pa_binding") != "required_at_simulation"
            or address.get("capture_allocator_coverage")
            not in {
                "target_window_pytorch_allocator",
                "target_window_pytorch_allocator_plus_tensor_segments",
            }
        ):
            raise ValueError(f"invalid range-rebase readiness boundary: {operator}")
        if not artifact.artifact_id.endswith(suffix):
            raise ValueError(f"invalid range-rebase artifact id: {operator}")

        record_path = qualification_paths[operator]
        record = _load(record_path)
        trace_id = artifact.artifact_id.removesuffix(suffix)
        if (
            record.get("schema_version") != "hetero-accel-sim-qualification/v1"
            or record.get("status") != "passed"
            or record.get("trace_id") != trace_id
            or record.get("replay_safety_qualified") is not False
            or "cycle_coupled_request_response"
            not in record.get("qualified_scopes", [])
        ):
            raise ValueError(f"invalid range-rebase qualification: {record_path}")
        ownership = _mapping(record.get("timing_ownership"), "timing_ownership")
        if (
            ownership.get("duration_mode") != "coupled"
            or ownership.get("external_ramulator2") != "shared3d.ramulator2"
            or ownership.get("gpu_local_dram") is not None
        ):
            raise ValueError(f"invalid timing ownership: {operator}")
        comparison = _mapping(record.get("comparison"), "comparison")
        cycles, _ = _identical_pair(
            comparison.get("gpu_tot_sim_cycle"), "gpu_tot_sim_cycle"
        )
        instructions, _ = _identical_pair(
            comparison.get("gpu_tot_sim_insn"), "gpu_tot_sim_insn"
        )
        memory, _ = _identical_pair(
            comparison.get("external_memory_stats"), "external_memory_stats"
        )
        memory = _mapping(memory, "external_memory_stats[0]")
        accepted = int(memory.get("reads", 0)) + int(memory.get("writes", 0))
        children = int(memory.get("children_sent", -1))
        translated = int(memory.get("address_translated", -1))
        if (
            int(memory.get("instances", 0)) != 1
            or accepted <= 0
            or int(memory.get("completed", -1)) != accepted
            or int(memory.get("durable_completed", -1)) != accepted
            or int(memory.get("gpu_parents", -1)) != accepted
            or int(memory.get("gpu_completed", -1)) != accepted
            or int(memory.get("atlas_parents", -1)) != 0
            or int(memory.get("atlas_completed", -1)) != 0
            or children <= 0
            or int(memory.get("children_completed", -1)) != children
            or int(memory.get("gpu_children", -1)) != children
            or int(memory.get("atlas_children", -1)) != 0
            or int(memory.get("outstanding", -1)) != 0
            or translated <= 0
            or int(memory.get("address_already_global", -1)) != 0
            or int(memory.get("address_unmapped", -1)) != 0
            or int(memory.get("address_binding_ranges", 0)) <= 0
        ):
            raise ValueError(f"range-rebase request conservation failed: {operator}")

        for field, value in (
            ("gpu_parents", accepted),
            ("gpu_children", children),
            ("address_translated", translated),
            ("logical_bytes", int(memory.get("logical_bytes", 0))),
            ("internal_bytes", int(memory.get("internal_bytes", 0))),
        ):
            totals[field] += value
        records[operator] = {
            "artifact_id": artifact.artifact_id,
            "artifact_manifest": str(artifact.source_path),
            "qualification_record": str(record_path),
            "q_len": key.q_len,
            "kv_length": key.kv_length,
            "gpu_cycles": int(cycles),
            "instructions": int(instructions),
            "gpu_parents": accepted,
            "gpu_children": children,
            "reads": int(memory["reads"]),
            "writes": int(memory["writes"]),
            "logical_bytes": int(memory["logical_bytes"]),
            "internal_bytes": int(memory["internal_bytes"]),
            "dram_cycles": int(memory["cycles"]),
            "link_cycles": int(memory["link_cycles"]),
            "backpressure_retries": int(memory["rejected"]),
            "address_translated": translated,
            "address_binding_ranges": int(memory["address_binding_ranges"]),
            "address_unmapped": 0,
            "ramulator2_instances": 1,
            "outstanding": 0,
            "compute_memory_coupled": True,
            "global_pa_binding_ready": True,
            "request_cycle_ready": True,
        }

    output = {
        "schema_version": "hetero-range-rebased-gpu-artifact-summary/v1",
        "status": "passed",
        "scope": {
            "model": first.model_spec_name,
            "checkpoint_revision": first.checkpoint_revision,
            "phase": first.phase,
            "layer_id": first.layer_id,
            "batch_size": first.batch_size,
            "context_length": first.context_length,
            "dtype": first.dtype,
            "operators": list(operators),
        },
        "catalog": str(catalog.source_path),
        "qualification_map": str(args.qualification_map.resolve()),
        "operators": records,
        "aggregate": {"operator_count": len(operators), **totals},
        "invariants": {
            "full_instruction_trace": True,
            "real_mem_fetch_stall_resume": True,
            "one_ramulator2_per_qualified_run": True,
            "all_trace_addresses_rebased": True,
            "all_gpu_parents_completed": True,
            "all_gpu_children_completed": True,
            "zero_atlas_requests": True,
            "zero_outstanding": True,
            "deterministic_double_run": True,
        },
        "claim_boundary": {
            "performance_claim_allowed": False,
            "global_pa_binding_ready": True,
            "request_cycle_ready": True,
            "prefill_global_timeline_integrated": False,
            "virtual_memory_mode": "range_rebase",
            "dram_mapper": "OneLevelInterleave(channel_lowest_bit=0)",
            "replay_safe_across_memory_candidates": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Range-rebased GPU Artifact summary passed: {args.output}")


if __name__ == "__main__":
    main()
