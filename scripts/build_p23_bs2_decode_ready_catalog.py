#!/usr/bin/env python3
"""Build a strict ready catalog for the 14 P23 BS=2 GPU operators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from frontend.hetero.operator_artifact import OperatorArtifactManifest
from scripts.build_p23_bs2_decode_capture_catalog import OPERATORS
from scripts.validate_p23_qualification_record import validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-catalog", required=True, type=Path)
    parser.add_argument("--coupled-root", required=True, type=Path)
    parser.add_argument("--qualification-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    capture = json.loads(args.capture_catalog.read_text(encoding="utf-8"))
    if (
        capture.get("schema_version")
        != "hetero-p23-bs2-decode-capture-catalog/v1"
        or int(capture.get("operator_count", -1)) != len(OPERATORS)
    ):
        raise ValueError("capture catalog is not the exact P23 BS=2 input")

    records: list[dict[str, object]] = []
    for operator in OPERATORS:
        stem = f"tinyllama_decode_bs2_ctx16_kv17_{operator}_sm89"
        artifact_path = (
            args.coupled_root
            / f"{stem}_shared_hbdram_range_rebase.json"
        ).resolve()
        qualification_path = (
            args.qualification_root
            / operator.replace("_", "-")
            / "qualification_record.json"
        ).resolve()
        artifact = OperatorArtifactManifest.load(artifact_path)
        key = artifact.compatibility_key
        qualification = dict(artifact.payload["qualification"])
        if (
            not artifact.request_cycle_ready
            or key.operator != operator
            or key.phase != "decode_step"
            or key.layer_id != 0
            or key.batch_size != 2
            or key.context_length != 16
            or key.q_len != 1
            or key.kv_length != 17
            or key.dtype != "fp16"
            or qualification.get("performance_eligible") is not False
        ):
            raise ValueError(f"coupled Artifact identity mismatch: {artifact_path}")
        stats = validate(qualification_path)
        records.append(
            {
                "operator_type": operator,
                "batch_size": 2,
                "context_length": 16,
                "q_len": 1,
                "kv_length": 17,
                "artifact": str(artifact_path),
                "artifact_sha256": artifact.content_sha256,
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

    payload = {
        "schema_version": "hetero-p23-bs2-decode-ready-catalog/v1",
        "status": "operator_set_ready_timeline_integration_pending",
        "model": "TinyLlama-1.1B",
        "checkpoint_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "scope": {
            "layers": [0],
            "batch_size": 2,
            "context_length": 16,
            "q_len": 1,
            "kv_length": 17,
        },
        "operator_count": len(records),
        "records": records,
        "sass_acquisition_policy": {
            "execution_host": "remote_rtx4090_only",
            "local_rtx3070_sass_allowed": False,
            "capture_device_sm": 89,
            "replay_target_sm": 86,
        },
        "claim_boundary": {
            "performance_claim_allowed": False,
            "timeline_integration_ready": False,
            "reason": (
                "The 14 exact GPU operators are independently request-cycle ready. "
                "KV Append and one shared batched global timeline remain pending."
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
