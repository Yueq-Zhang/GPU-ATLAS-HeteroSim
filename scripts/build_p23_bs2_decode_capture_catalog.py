#!/usr/bin/env python3
"""Build the fail-closed catalog for one exact P23 BS=2 Decode shape."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from frontend.hetero.operator_artifact import OperatorArtifactManifest

OPERATORS = (
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
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str, parent: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (parent / value).resolve()


def _trace_manifest_path(artifact: OperatorArtifactManifest) -> Path:
    files = artifact.payload["files"]
    matches = [
        item
        for item in files  # type: ignore[union-attr]
        if item.get("kind") == "accel_sim_trace_manifest"
    ]
    if len(matches) != 1:
        raise ValueError(f"Artifact needs one Trace Manifest: {artifact.source_path}")
    return _resolve(str(matches[0]["path"]), artifact.source_path.parent)


def _validate_binary_identity(
    trace_manifest: dict[str, object], path: Path
) -> tuple[list[int], str]:
    compilation = dict(trace_manifest.get("compilation", {}))
    identity = dict(compilation.get("binary_identity", {}))
    versions = [int(value) for value in identity.get("binary_versions", [])]
    if (
        compilation.get("capture_device_sm") != 89
        or compilation.get("target_sm") != 86
        or not versions
        or not set(versions).issubset({80, 86})
        or identity.get("source") != "nvbit_trace_headers"
        or int(identity.get("kernel_launch_count", 0)) <= 0
    ):
        raise ValueError(f"Trace binary identity mismatch: {path}")
    compatibility = str(
        identity.get(
            "replay_compatibility",
            "single_binary_exact" if len(versions) == 1 else "",
        )
    )
    if len(versions) > 1 and compatibility != "ampere_sm80_sm86_opcode_compatible":
        raise ValueError(f"mixed Ampere Trace lacks explicit replay contract: {path}")
    return versions, compatibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    records: list[dict[str, object]] = []
    for operator in OPERATORS:
        stem = f"tinyllama_decode_bs2_ctx16_kv17_{operator}_sm89"
        artifact_path = root / f"{stem}.json"
        artifact = OperatorArtifactManifest.load(artifact_path)
        key = artifact.compatibility_key
        if (
            key.operator != operator
            or key.phase != "decode_step"
            or key.layer_id != 0
            or key.batch_size != 2
            or key.context_length != 16
            or key.q_len != 1
            or key.kv_length != 17
            or key.dtype != "fp16"
            or artifact.payload["backend"].get("target_sm") != 86  # type: ignore[union-attr]
        ):
            raise ValueError(f"Artifact identity mismatch: {artifact_path}")
        trace_path = _trace_manifest_path(artifact)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        versions, compatibility = _validate_binary_identity(trace, trace_path)
        kernels = _resolve(str(trace["kernels_list"]), trace_path.parent)
        records.append(
            {
                "operator_type": operator,
                "batch_size": 2,
                "context_length": 16,
                "q_len": 1,
                "kv_length": 17,
                "capture_device_sm": 89,
                "binary_versions": versions,
                "replay_target_sm": 86,
                "replay_compatibility": compatibility,
                "artifact": str(artifact_path),
                "artifact_sha256": artifact.content_sha256,
                "trace_manifest": str(trace_path),
                "trace_manifest_sha256": _sha256(trace_path),
                "kernels_list": str(kernels),
                "kernels_list_sha256": _sha256(kernels),
                "range_rebase_ready": False,
                "double_run_qualified": False,
                "request_cycle_ready": False,
                "performance_eligible": False,
            }
        )

    payload = {
        "schema_version": "hetero-p23-bs2-decode-capture-catalog/v1",
        "status": "capture_complete_pending_range_rebase_qualification",
        "model": "TinyLlama-1.1B",
        "checkpoint_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "scope": {
            "layers": [0],
            "batch_size": 2,
            "context_length": 16,
            "q_len": 1,
            "kv_length": 17,
        },
        "capture_gpu": "NVIDIA GeForce RTX 4090",
        "capture_device_sm": 89,
        "replay_target_sm": 86,
        "sass_acquisition_policy": {
            "execution_host": "remote_rtx4090_only",
            "compile_execute_and_nvbit_capture_colocated": True,
            "local_rtx3070_sass_allowed": False,
            "binary_version_semantics": (
                "Per-kernel SASS binary versions are read from the traces actually "
                "executed on the remote RTX4090; they are not rewritten to SM89."
            ),
        },
        "operator_count": len(records),
        "records": records,
        "claim_boundary": {
            "performance_claim_allowed": False,
            "timeline_integration_ready": False,
            "reason": (
                "Capture identity is not Range-Rebase qualification, a complete "
                "batched Decode timeline, or RTX4090 performance calibration."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
