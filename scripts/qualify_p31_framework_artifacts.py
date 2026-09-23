#!/usr/bin/env python3
"""Qualify P31 GPU SASS and ATLAS Tensor-IR executable Artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.hetero.operator_artifact import OperatorArtifactManifest  # noqa: E402
from frontend.hetero.trace_manifest import TraceManifest  # noqa: E402


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"record is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30-qualification", required=True, type=Path)
    parser.add_argument("--gpu-artifact", required=True, type=Path)
    parser.add_argument("--gpu-trace-manifest", required=True, type=Path)
    parser.add_argument("--gpu-binding", required=True, type=Path)
    parser.add_argument("--gpu-replay-stats", required=True, type=Path)
    parser.add_argument("--atlas-run-a", required=True, type=Path)
    parser.add_argument("--atlas-run-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {
        name: value.resolve() for name, value in vars(args).items() if name != "output"
    }
    p30 = _load(paths["p30_qualification"])
    if (
        p30.get("schema_version") != "hetero-p30-huggingface-online-qualification/v1"
        or p30.get("status") != "passed"
    ):
        raise ValueError("P31 requires a passed P30 qualification")
    simulation_key = p30["double_run"]["runs"][0]["simulation_key"]

    gpu_artifact = OperatorArtifactManifest.load(paths["gpu_artifact"])
    gpu_trace = TraceManifest.load(paths["gpu_trace_manifest"])
    backend = _mapping(gpu_artifact.payload["backend"], "GPU backend")
    execution = _mapping(
        gpu_artifact.payload["execution_contract"], "GPU execution contract"
    )
    if (
        backend.get("capture_device_sm") != 89
        or backend.get("target_sm") != 86
        or backend.get("binary_versions") != [86]
        or execution.get("memory_traffic") != "full_instruction_trace"
        or execution.get("global_pa_binding_ready") is not True
        or execution.get("request_cycle_ready") is not False
    ):
        raise ValueError("P31 GPU Artifact identity or readiness boundary is invalid")
    compilation = _mapping(gpu_trace.compilation, "trace compilation")
    binary = _mapping(compilation.get("binary_identity"), "binary identity")
    if (
        compilation.get("framework_simulation_key") != simulation_key
        or binary.get("kernel_launch_count") != 40
        or binary.get("binary_versions") != [86]
    ):
        raise ValueError("P31 GPU Trace is not bound to the P30 identity")
    binding = _load(paths["gpu_binding"])
    if (
        binding.get("schema_version") != "hetero-online-address-binding/v1"
        or binding.get("range_count") != len(gpu_trace.address_ranges)
        or int(binding.get("allocated_bytes", 0)) <= 0
    ):
        raise ValueError("P31 GPU Global PA binding is incomplete")

    gpu_stats = _load(paths["gpu_replay_stats"])
    external = _mapping(gpu_stats.get("external_memory_stats"), "external memory")
    accepted = int(external.get("reads", -1)) + int(external.get("writes", -1))
    replay_gates = {
        "positive_gpu_cycles": int(gpu_stats.get("cycles", 0)) > 0,
        "positive_gpu_instructions": int(gpu_stats.get("instructions", 0)) > 0,
        "single_ramulator2": int(external.get("instances", 0)) == 1,
        "all_parents_completed": int(external.get("completed", -1)) == accepted,
        "zero_unmapped": int(external.get("address_unmapped", -1)) == 0,
        "nonzero_translated": int(external.get("address_translated", 0)) > 0,
        "zero_in_flight": int(external.get("outstanding", -1)) == 0,
    }
    if not all(replay_gates.values()):
        raise ValueError(f"P31 GPU replay failed: {replay_gates}")

    atlas_runs = [_load(paths["atlas_run_a"]), _load(paths["atlas_run_b"])]
    for atlas in atlas_runs:
        if (
            atlas.get("schema_version")
            != "hetero-p31-atlas-executable-qualification/v1"
            or atlas.get("status") != "passed"
            or atlas.get("framework_simulation_key") != simulation_key
        ):
            raise ValueError("P31 ATLAS Artifact does not match P30")
    atlas_fields = (
        "artifact_key",
        "artifact_sha256",
        "trace_file_sha256",
        "trace",
        "tensor_global_pa",
        "tensor_extents_bytes",
    )
    if any(atlas_runs[0][field] != atlas_runs[1][field] for field in atlas_fields):
        raise ValueError("P31 ATLAS double run is not deterministic")

    record = {
        "schema_version": "hetero-p31-framework-artifact-qualification/v1",
        "status": "passed",
        "execution_host": "remote_rtx4090_sm89",
        "p30_simulation_key": simulation_key,
        "gpu": {
            "artifact_id": gpu_artifact.artifact_id,
            "capture_device_sm": 89,
            "actual_binary_versions": [86],
            "replay_target_sm": 86,
            "kernel_launch_count": 40,
            "allocator_range_count": len(gpu_trace.address_ranges),
            "allocated_global_pa_bytes": binding["allocated_bytes"],
            "single_replay_address_audit": replay_gates,
            "request_cycle_ready": False,
        },
        "atlas": {
            "artifact_key": atlas_runs[0]["artifact_key"],
            "trace": atlas_runs[0]["trace"],
            "deterministic_double_run": True,
            "ramulator2_cycle_replay_qualified": False,
        },
        "qualification": {
            "framework_selected_sass_captured": True,
            "sass_binary_identity_sealed": True,
            "trace_address_to_global_pa_audited": True,
            "atlas_tensor_ir_to_full_memory_trace": True,
            "automatic_capture_pipeline_available": True,
            "gpu_double_replay_qualified": False,
            "end_to_end_framework_timeline_qualified": False,
            "performance_claim_allowed": False,
        },
        "evidence_sha256": {
            name: _sha256(path) for name, path in sorted(paths.items())
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
