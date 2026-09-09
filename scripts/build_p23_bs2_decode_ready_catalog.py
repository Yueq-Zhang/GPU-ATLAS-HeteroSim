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


def _render_path(path: Path, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _validate_timeline_qualification(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scope = payload.get("scope")
    legs = payload.get("legs")
    if (
        payload.get("schema_version")
        != "hetero-p23-bs2-decode-timeline-qualification/v1"
        or payload.get("status") != "passed"
        or payload.get("qualification_passed") is not True
        or payload.get("double_run_signature_equal") is not True
        or payload.get("performance_claim_allowed") is not False
        or not isinstance(scope, dict)
        or scope.get("model") != "TinyLlama-1.1B"
        or scope.get("layers") != [0]
        or int(scope.get("batch_size", -1)) != 2
        or int(scope.get("context_length", -1)) != 16
        or int(scope.get("q_len", -1)) != 1
        or int(scope.get("kv_length", -1)) != 17
        or int(scope.get("decode_steps", -1)) != 1
        or int(scope.get("gpu_operator_types", -1)) != len(OPERATORS)
        or int(scope.get("gpu_task_instances", -1)) != 15
        or int(scope.get("kv_append_instances", -1)) != 1
        or not isinstance(legs, list)
        or len(legs) != 2
    ):
        raise ValueError("timeline qualification does not close the exact P23 scope")

    makespans: list[int] = []
    for index, leg in enumerate(legs, 1):
        if not isinstance(leg, dict):
            raise ValueError(f"P23 leg {index} is not an object")
        checks = leg.get("checks")
        dependencies = leg.get("dependencies")
        global_pa = leg.get("global_pa")
        versions = leg.get("versions")
        if (
            not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
            or not isinstance(dependencies, dict)
            or not all(
                dependencies.get(key) is True
                for key in (
                    "all_tasks_on_gpu0",
                    "resource_intervals_non_overlapping",
                    "dependencies_complete_before_launch",
                )
            )
            or int(dependencies.get("dependency_edge_count", -1)) != 19
            or not isinstance(global_pa, dict)
            or global_pa.get("non_overlapping") is not True
            or global_pa.get("kv_append_writes_stay_in_member_slice") is not True
            or int(global_pa.get("trace_binding_count", -1)) != 15
            or int(global_pa.get("runtime_binding_count", -1)) != 5
            or int(global_pa.get("member_kv_slice_count", -1)) != 4
            or not isinstance(versions, dict)
            or not all(value is True for value in versions.values())
            or len(leg.get("gpu_tasks", [])) != 15
            or len(leg.get("runtime_tasks", [])) != 5
            or leg.get("performance_claim_allowed") is not False
        ):
            raise ValueError(f"P23 leg {index} failed a sealed timeline invariant")
        makespans.append(int(leg["makespan_fs"]))
    if makespans[0] != makespans[1]:
        raise ValueError("P23 timeline makespan differs across the two legs")
    return {
        "double_run_signature_equal": True,
        "gpu_task_instances": 15,
        "runtime_task_instances": 5,
        "total_task_instances": 20,
        "makespan_fs": makespans[0],
        "performance_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-catalog", required=True, type=Path)
    parser.add_argument("--coupled-root", required=True, type=Path)
    parser.add_argument("--qualification-root", required=True, type=Path)
    parser.add_argument("--timeline-qualification", type=Path)
    parser.add_argument("--path-base", type=Path, default=Path.cwd())
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
                "artifact": _render_path(artifact_path, args.path_base),
                "artifact_sha256": artifact.content_sha256,
                "qualification_record": _render_path(
                    qualification_path, args.path_base
                ),
                "qualification_record_sha256": _sha256(qualification_path),
                "range_rebase_ready": True,
                "global_pa_binding_ready": True,
                "double_run_qualified": True,
                "request_cycle_ready": True,
                "performance_eligible": False,
                **stats,
            }
        )

    timeline = None
    if args.timeline_qualification is not None:
        timeline_path = args.timeline_qualification.resolve()
        timeline = {
            "qualification_record": _render_path(timeline_path, args.path_base),
            "qualification_record_sha256": _sha256(timeline_path),
            **_validate_timeline_qualification(timeline_path),
        }

    timeline_ready = timeline is not None
    payload = {
        "schema_version": "hetero-p23-bs2-decode-ready-catalog/v1",
        "status": (
            "timeline_qualified"
            if timeline_ready
            else "operator_set_ready_timeline_integration_pending"
        ),
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
        "capture_catalog": {
            "path": _render_path(args.capture_catalog, args.path_base),
            "sha256": _sha256(args.capture_catalog),
        },
        "timeline_qualification": timeline,
        "sass_acquisition_policy": {
            "execution_host": "remote_rtx4090_only",
            "local_rtx3070_sass_allowed": False,
            "capture_device_sm": 89,
            "replay_target_sm": 86,
        },
        "claim_boundary": {
            "performance_claim_allowed": False,
            "timeline_integration_ready": timeline_ready,
            "reason": (
                "The 14 exact GPU operators, live KV Append and the fused BS=2 "
                "single-layer timeline passed deterministic double-run functional "
                "qualification; hardware performance calibration remains pending."
                if timeline_ready
                else "The 14 exact GPU operators are independently request-cycle "
                "ready. KV Append and one shared batched global timeline remain "
                "pending."
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
