#!/usr/bin/env python3
"""Create one evidence-linked comparison for the qualified Q projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualified_stats(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record_path = root / "qualification_record.json"
    stats_path = root / "adapter" / "stats.json"
    record = _load(record_path)
    stats = _load(stats_path)
    if record.get("status") != "passed":
        raise ValueError(f"qualification did not pass: {record_path}")
    return record, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-manifest", required=True, type=Path)
    parser.add_argument("--atlas-manifest", required=True, type=Path)
    parser.add_argument("--gpu-native", required=True, type=Path)
    parser.add_argument("--gpu-shared-3ddram", required=True, type=Path)
    parser.add_argument("--atlas-internal-3ddram", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    trace = _load(args.trace_manifest)
    atlas_artifact = _load(args.atlas_manifest)
    native_record, native = _qualified_stats(args.gpu_native)
    shared_record, shared = _qualified_stats(args.gpu_shared_3ddram)
    atlas_record, atlas = _qualified_stats(args.atlas_internal_3ddram)

    capture = trace["capture"]
    source = atlas_artifact["source_contract"]
    gpu_shape = {
        "M": int(capture["shape"]["m"]),
        "K": int(capture["shape"]["k"]),
        "N": int(capture["shape"]["n"]),
    }
    comparable = {
        "model": capture["model"] == source["model"],
        "checkpoint_revision": capture["model_revision"]
        == source["checkpoint_revision"],
        "operator": capture["operator"] == source["operator"],
        "shape": gpu_shape == source["shape"],
        "dtype": capture["dtype"] == "fp16" and source["dtype"] == "float16",
        "batch_size": int(capture["batch_size"]) == int(source["batch_size"]),
        "context_length": int(capture["context_length"])
        == int(source["initial_kv_length"]),
    }
    if not all(comparable.values()):
        raise ValueError(f"GPU and ATLAS workload contracts differ: {comparable}")

    native_fs = int(native["duration_fs"])
    shared_fs = int(shared["duration_fs"])
    atlas_fs = int(atlas["duration_fs"])
    external = shared.get("external_memory_stats")
    atlas_e2e = atlas["native_stats"]["e2e_stats"]
    payload = {
        "schema_version": "hetero-shape-matched-operator-comparison/v1",
        "status": "qualified_shape_matched_component_comparison",
        "workload_contract": {
            "model": source["model"],
            "checkpoint_revision": source["checkpoint_revision"],
            "operator": source["operator"],
            "phase": source["phase"],
            "batch_size": source["batch_size"],
            "initial_kv_length": source["initial_kv_length"],
            "dtype": source["dtype"],
            "shape": source["shape"],
            "contract_fields_equal": comparable,
        },
        "results": {
            "rtx3070_native_vram": {
                "cycles": int(native["cycles"]),
                "frequency_hz": int(native["core_frequency_hz"]),
                "duration_fs": native_fs,
                "duration_us": native_fs / 1_000_000_000,
                "instructions": int(native["instructions"]),
                "memory_timing_owner": "accel_sim_native_gpu_memory",
            },
            "rtx3070_shared_3ddram": {
                "cycles": int(shared["cycles"]),
                "frequency_hz": int(shared["core_frequency_hz"]),
                "duration_fs": shared_fs,
                "duration_us": shared_fs / 1_000_000_000,
                "instructions": int(shared["instructions"]),
                "external_link_payload_bandwidth_Bps": 12_800_000_000,
                "internal_3ddram_peak_payload_bandwidth_Bps": 409_600_000_000,
                "memory_timing_owner": "one_in_process_ramulator2",
                "external_memory_stats": external,
            },
            "atlas_internal_3ddram": {
                "cycles": int(atlas["cycles"]),
                "frequency_hz": int(atlas["core_frequency_hz"]),
                "duration_fs": atlas_fs,
                "duration_us": atlas_fs / 1_000_000_000,
                "matrix_cycles": int(atlas_e2e["matrix_cycles"]),
                "dram_cycles": int(atlas_e2e["dram_cycles"]),
                "memory_access_bytes": int(atlas_e2e["memory_access_bytes"]),
                "reported_operation_count": int(atlas_e2e["flop_count"]),
                "internal_3ddram_peak_payload_bandwidth_Bps": 409_600_000_000,
                "memory_timing_owner": "atlasim_internal_ramulator2_per_core_partition",
            },
        },
        "ratios": {
            "shared_3ddram_gpu_over_native_gpu_latency": shared_fs / native_fs,
            "native_gpu_over_atlas_latency": native_fs / atlas_fs,
            "shared_3ddram_gpu_over_atlas_latency": shared_fs / atlas_fs,
        },
        "qualification": {
            "gpu_native_status": native_record["status"],
            "gpu_shared_3ddram_status": shared_record["status"],
            "atlas_internal_3ddram_status": atlas_record["status"],
            "replay_safe_across_hardware_configs": False,
            "scope": "one exact layer-0 Q-projection; not one layer or end-to-end inference",
            "interpretation": (
                "Shape and checkpoint are matched, but backends model different compute "
                "microarchitectures; ratios are configuration-study results, not measured-hardware claims."
            ),
        },
        "provenance": {
            "trace_manifest": str(args.trace_manifest.resolve()),
            "trace_manifest_sha256": _sha256(args.trace_manifest),
            "atlas_manifest": str(args.atlas_manifest.resolve()),
            "atlas_manifest_sha256": _sha256(args.atlas_manifest),
            "gpu_native_qualification": str(
                (args.gpu_native / "qualification_record.json").resolve()
            ),
            "gpu_shared_3ddram_qualification": str(
                (args.gpu_shared_3ddram / "qualification_record.json").resolve()
            ),
            "atlas_internal_3ddram_qualification": str(
                (args.atlas_internal_3ddram / "qualification_record.json").resolve()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
