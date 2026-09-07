#!/usr/bin/env python3
"""Build strict P21 catalogs for 56 qualified Decode Trace shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from frontend.hetero.operator_artifact import OperatorArtifactManifest

from scripts.build_p21_sm89_decode_capture_catalog import KV_LENGTHS, OPERATORS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} root must be an object")
    return payload


def _validate_qualification(path: Path, trace_id: str) -> dict[str, object]:
    payload = _load(path)
    comparison = payload.get("comparison", {})
    ownership = payload.get("timing_ownership", {})
    cycles = comparison.get("gpu_tot_sim_cycle")
    instructions = comparison.get("gpu_tot_sim_insn")
    memories = comparison.get("external_memory_stats")
    if (
        payload.get("status") != "passed"
        or payload.get("trace_id") != trace_id
        or not isinstance(cycles, list)
        or len(cycles) != 2
        or cycles[0] != cycles[1]
        or not isinstance(instructions, list)
        or len(instructions) != 2
        or instructions[0] != instructions[1]
        or not isinstance(memories, list)
        or len(memories) != 2
        or memories[0] != memories[1]
        or ownership.get("duration_mode") != "coupled"
        or ownership.get("external_ramulator2") != "shared3d.ramulator2"
        or ownership.get("gpu_local_dram") is not None
    ):
        raise ValueError(f"invalid P21 double-run qualification: {path}")
    stats = memories[0]
    parents = int(stats.get("gpu_parents", -1))
    children = int(stats.get("gpu_children", -1))
    if (
        int(stats.get("instances", -1)) != 1
        or parents <= 0
        or children <= 0
        or int(stats.get("gpu_completed", -1)) != parents
        or int(stats.get("completed", -1)) != parents
        or int(stats.get("children_sent", -1)) != children
        or int(stats.get("children_completed", -1)) != children
        or int(stats.get("durable_completed", -1)) != parents
        or int(stats.get("address_translated", 0)) <= 0
        or int(stats.get("address_unmapped", -1)) != 0
        or int(stats.get("atlas_parents", -1)) != 0
        or int(stats.get("atlas_children", -1)) != 0
        or int(stats.get("atlas_completed", -1)) != 0
        or int(stats.get("outstanding", -1)) != 0
    ):
        raise ValueError(f"invalid P21 request conservation: {path}")
    return {
        "cycles": int(cycles[0]),
        "instructions": int(instructions[0]),
        "parents": parents,
        "children": children,
        "translated": int(stats["address_translated"]),
        "ramulator2_instances": int(stats["instances"]),
        "outstanding": int(stats["outstanding"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-catalog", required=True, type=Path)
    parser.add_argument("--coupled-root", required=True, type=Path)
    parser.add_argument("--qualification-root", required=True, type=Path)
    parser.add_argument("--artifact-catalog", required=True, type=Path)
    parser.add_argument("--qualification-map", required=True, type=Path)
    parser.add_argument("--ready-catalog", required=True, type=Path)
    args = parser.parse_args()

    capture = _load(args.capture_catalog)
    if (
        capture.get("schema_version") != "hetero-p21-sm89-decode-capture-catalog/v1"
        or int(capture.get("record_count", -1)) != 56
        or capture.get("capture_device_sm") != 89
        or capture.get("binary_sm") != 86
    ):
        raise ValueError("capture catalog is not the fixed 56-record P21 input")

    artifacts: list[str] = []
    qualifications: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for kv_length in KV_LENGTHS:
        for operator in OPERATORS:
            stem = f"tinyllama_decode_bs1_ctx16_kv{kv_length}_{operator}_sm89"
            artifact_path = (
                args.coupled_root / f"{stem}_shared_hbdram_range_rebase.json"
            ).resolve()
            qualification_path = (
                args.qualification_root
                / f"kv{kv_length}-{operator.replace('_', '-')}"
                / "qualification_record.json"
            ).resolve()
            artifact = OperatorArtifactManifest.load(artifact_path)
            key = artifact.compatibility_key
            if (
                not artifact.request_cycle_ready
                or key.operator != operator
                or key.phase != "decode_step"
                or key.layer_id != 0
                or key.batch_size != 1
                or key.context_length != 16
                or key.q_len != 1
                or key.kv_length != kv_length
                or key.dtype != "fp16"
                or artifact.payload["qualification"].get("performance_eligible")
                is not False
            ):
                raise ValueError(f"coupled Artifact identity mismatch: {artifact_path}")
            trace_id = artifact.artifact_id.removesuffix(
                ".shared_hbdram_range_rebase_v1"
            )
            stats = _validate_qualification(qualification_path, trace_id)
            shape_key = f"kv{kv_length}.{operator}"
            artifacts.append(str(artifact_path))
            qualifications[shape_key] = str(qualification_path)
            records.append(
                {
                    "shape_key": shape_key,
                    "operator_type": operator,
                    "context_length": 16,
                    "q_len": 1,
                    "kv_length": kv_length,
                    "capture_device_sm": 89,
                    "binary_sm": 86,
                    "artifact": str(artifact_path),
                    "artifact_sha256": _sha256(artifact_path),
                    "qualification_record": str(qualification_path),
                    "qualification_record_sha256": _sha256(qualification_path),
                    "range_rebase_ready": True,
                    "global_pa_binding_ready": True,
                    "double_run_qualified": True,
                    "request_cycle_ready": True,
                    "performance_eligible": False,
                    **stats,
                }
            )

    artifact_catalog = {
        "schema_version": "hetero-operator-artifact-catalog/v1",
        "required_operators": list(OPERATORS),
        "zero_fallback_required": True,
        "artifacts": artifacts,
    }
    qualification_map = {
        "schema_version": "hetero-coupled-qualification-map/v1",
        "records": qualifications,
    }
    ready_catalog = {
        "schema_version": "hetero-p21-decode-ready-catalog/v1",
        "status": "passed",
        "model": "TinyLlama-1.1B",
        "checkpoint_revision": ("fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
        "capture_gpu": "NVIDIA GeForce RTX 4090",
        "capture_device_sm": 89,
        "executed_binary_and_replay_sm": 86,
        "shape_count": len(KV_LENGTHS),
        "operator_count_per_shape": len(OPERATORS),
        "record_count": len(records),
        "records": records,
        "claim_boundary": {
            "performance_claim_allowed": False,
            "reason": (
                "All 56 exact-shape traces pass functional request-cycle coupling, "
                "but the SM86 replay configuration and runtime contracts are not "
                "calibrated to the physical RTX4090 or a measured 3D-DRAM system."
            ),
        },
    }
    for path, payload in (
        (args.artifact_catalog, artifact_catalog),
        (args.qualification_map, qualification_map),
        (args.ready_catalog, ready_catalog),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
