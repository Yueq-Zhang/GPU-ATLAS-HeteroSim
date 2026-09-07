#!/usr/bin/env python3
"""Build the fail-closed catalog for the 56 exact P21 SM89 Decode captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
KV_LENGTHS = (17, 18, 19, 20)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved(path: str, parent: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    records: list[dict[str, object]] = []
    for kv_length in KV_LENGTHS:
        for operator in OPERATORS:
            artifact_path = root / (
                f"tinyllama_decode_bs1_ctx16_kv{kv_length}_{operator}_sm89.json"
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            source = artifact.get("source_contract", {})
            backend = artifact.get("backend", {})
            if (
                source.get("operator") != operator
                or source.get("phase") != "decode_step"
                or source.get("batch_size") != 1
                or source.get("context_length") != 16
                or source.get("q_len") != 1
                or source.get("kv_length") != kv_length
                or source.get("dtype") != "fp16"
                or backend.get("target_sm") != 86
                or backend.get("gpu") != "NVIDIA GeForce RTX 4090"
            ):
                raise ValueError(f"artifact identity mismatch: {artifact_path}")
            trace_files = [
                item
                for item in artifact.get("files", [])
                if item.get("kind") == "accel_sim_trace_manifest"
            ]
            if len(trace_files) != 1:
                raise ValueError(f"artifact needs one Trace Manifest: {artifact_path}")
            trace_manifest_path = _resolved(
                str(trace_files[0]["path"]), artifact_path.parent
            )
            if _sha256(trace_manifest_path) != trace_files[0].get("sha256"):
                raise ValueError(f"Trace Manifest digest mismatch: {artifact_path}")
            trace_manifest = json.loads(
                trace_manifest_path.read_text(encoding="utf-8")
            )
            compilation = trace_manifest.get("compilation", {})
            binary_identity = compilation.get("binary_identity", {})
            if (
                compilation.get("capture_device_sm") != 89
                or compilation.get("target_sm") != 86
                or binary_identity.get("binary_versions") != [86]
                or binary_identity.get("source") != "nvbit_trace_headers"
                or not isinstance(binary_identity.get("kernel_launch_count"), int)
                or binary_identity.get("kernel_launch_count", 0) <= 0
            ):
                raise ValueError(f"Trace binary identity mismatch: {artifact_path}")
            kernels_list = _resolved(
                str(trace_manifest["kernels_list"]), trace_manifest_path.parent
            )
            records.append(
                {
                    "operator_type": operator,
                    "context_length": 16,
                    "q_len": 1,
                    "kv_length": kv_length,
                    "capture_device_sm": 89,
                    "binary_sm": 86,
                    "artifact": str(artifact_path),
                    "artifact_sha256": _sha256(artifact_path),
                    "trace_manifest": str(trace_manifest_path),
                    "trace_manifest_sha256": _sha256(trace_manifest_path),
                    "kernels_list": str(kernels_list),
                    "kernels_list_sha256": _sha256(kernels_list),
                    "kernel_sequence_sha256": binary_identity[
                        "kernel_sequence_sha256"
                    ],
                    "kernel_launch_count": binary_identity["kernel_launch_count"],
                    "range_rebase_ready": False,
                    "global_pa_bound": False,
                    "double_run_qualified": False,
                    "request_cycle_ready": False,
                    "performance_eligible": False,
                }
            )

    payload = {
        "schema_version": "hetero-p21-sm89-decode-capture-catalog/v1",
        "status": "capture_complete_pending_range_rebase_and_accel_sim_qualification",
        "model": "TinyLlama-1.1B",
        "checkpoint_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "capture_gpu": "NVIDIA GeForce RTX 4090",
        "capture_device_sm": 89,
        "binary_sm": 86,
        "shape_count": len(KV_LENGTHS),
        "operator_count_per_shape": len(OPERATORS),
        "record_count": len(records),
        "records": records,
        "claim_boundary": {
            "performance_claim_allowed": False,
            "reason": (
                "Capture identity alone does not qualify SM89 replay, request-cycle "
                "coupling, RTX 4090 timing accuracy or end-to-end Decode performance."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"P21 SM89 Decode capture catalog written: {args.output}")


if __name__ == "__main__":
    main()
