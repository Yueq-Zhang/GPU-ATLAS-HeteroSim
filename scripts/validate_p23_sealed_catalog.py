#!/usr/bin/env python3
"""Validate the repository-contained P23 evidence without raw remote traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_p23_bs2_decode_capture_catalog import OPERATORS
from scripts.build_p23_bs2_decode_ready_catalog import (
    _validate_timeline_qualification,
)
from scripts.validate_p23_qualification_record import validate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_path(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"sealed P23 reference must be repository-relative: {path}")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"sealed P23 reference escapes the repository: {path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_catalog(catalog_path: Path, project_root: Path) -> dict[str, object]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = catalog.get("records")
    claim = catalog.get("claim_boundary")
    if (
        catalog.get("schema_version")
        != "hetero-p23-bs2-decode-ready-catalog/v1"
        or catalog.get("status") != "timeline_qualified"
        or int(catalog.get("operator_count", -1)) != len(OPERATORS)
        or not isinstance(records, list)
        or len(records) != len(OPERATORS)
        or not isinstance(claim, dict)
        or claim.get("timeline_integration_ready") is not True
        or claim.get("performance_claim_allowed") is not False
    ):
        raise ValueError("P23 sealed Catalog header is incomplete")

    capture = catalog.get("capture_catalog")
    timeline = catalog.get("timeline_qualification")
    if not isinstance(capture, dict) or not isinstance(timeline, dict):
        raise ValueError("P23 sealed Catalog lacks capture or timeline evidence")
    capture_path = _repository_path(project_root, capture.get("path"))
    if _sha256(capture_path) != capture.get("sha256"):
        raise ValueError("P23 capture Catalog hash mismatch")
    timeline_path = _repository_path(
        project_root, timeline.get("qualification_record")
    )
    if _sha256(timeline_path) != timeline.get("qualification_record_sha256"):
        raise ValueError("P23 timeline qualification hash mismatch")
    timeline_summary = _validate_timeline_qualification(timeline_path)
    for key, value in timeline_summary.items():
        if timeline.get(key) != value:
            raise ValueError(f"P23 timeline summary mismatch: {key}")

    by_operator = {str(item.get("operator_type")): item for item in records}
    if set(by_operator) != set(OPERATORS):
        raise ValueError("P23 sealed Catalog operator coverage is not exact")
    for operator in OPERATORS:
        record = by_operator[operator]
        if any(
            record.get(key) is not expected
            for key, expected in (
                ("double_run_qualified", True),
                ("range_rebase_ready", True),
                ("global_pa_binding_ready", True),
                ("request_cycle_ready", True),
                ("performance_eligible", False),
            )
        ):
            raise ValueError(f"P23 operator is not sealed Ready: {operator}")
        artifact_path = _repository_path(project_root, record.get("artifact"))
        qualification_path = _repository_path(
            project_root, record.get("qualification_record")
        )
        if _sha256(artifact_path) != record.get("artifact_sha256"):
            raise ValueError(f"P23 Artifact hash mismatch: {operator}")
        if _sha256(qualification_path) != record.get(
            "qualification_record_sha256"
        ):
            raise ValueError(f"P23 qualification hash mismatch: {operator}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        source = artifact.get("source_contract")
        execution = artifact.get("execution_contract")
        if (
            not isinstance(source, dict)
            or source.get("operator") != operator
            or source.get("phase") != "decode_step"
            or int(source.get("layer_id", -1)) != 0
            or int(source.get("batch_size", -1)) != 2
            or int(source.get("context_length", -1)) != 16
            or int(source.get("q_len", -1)) != 1
            or int(source.get("kv_length", -1)) != 17
            or source.get("dtype") != "fp16"
            or not isinstance(execution, dict)
            or execution.get("request_cycle_ready") is not True
        ):
            raise ValueError(f"P23 Artifact identity mismatch: {operator}")
        stats = validate(qualification_path)
        for key in ("cycles", "instructions", "parents", "children", "translated"):
            if int(record.get(key, -1)) != int(stats[key]):
                raise ValueError(f"P23 operator statistic mismatch: {operator}.{key}")

    return {
        "status": "passed",
        "operator_count": len(records),
        "timeline_qualified": True,
        "double_run_signature_equal": True,
        "total_task_instances": int(timeline["total_task_instances"]),
        "makespan_fs": int(timeline["makespan_fs"]),
        "performance_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", type=Path, default=Path("validation/p23/ready_catalog.json")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = validate_catalog(args.catalog.resolve(), args.project_root.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
