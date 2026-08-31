#!/usr/bin/env python3
"""Audit a P17 calibration record against deterministic simulation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.hetero.performance_calibration import PerformanceCalibration  # noqa: E402


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _run_snapshot(path: Path) -> dict[str, object]:
    metrics = _read_object(path / "metrics.json")
    provenance = _read_object(path / "provenance.json")
    return {
        "path": str(path.resolve()),
        "simulation_input_key": provenance.get("simulation_input_key"),
        "makespan_fs": metrics.get("makespan_fs"),
        "requests": metrics.get("requests"),
        "performance_claim_allowed": metrics.get("performance_claim_allowed"),
        "run_status": metrics.get("run_status"),
    }


def build_audit(
    calibration_path: Path,
    run_dirs: list[Path],
    project_root: Path,
) -> dict[str, object]:
    calibration = PerformanceCalibration.load(calibration_path)
    calibration_audit = calibration.audit(project_root)
    snapshots = [_run_snapshot(path) for path in run_dirs]
    blockers = list(calibration_audit["blockers"])
    deterministic = True
    if len(snapshots) < 2:
        deterministic = False
        blockers.append("runs:at_least_two_required")
    elif any(
        snapshot["simulation_input_key"] != snapshots[0]["simulation_input_key"]
        or snapshot["makespan_fs"] != snapshots[0]["makespan_fs"]
        or snapshot["requests"] != snapshots[0]["requests"]
        for snapshot in snapshots[1:]
    ):
        deterministic = False
        blockers.append("runs:determinism_mismatch")
    run_claims_allowed = all(
        snapshot["performance_claim_allowed"] is True for snapshot in snapshots
    )
    if not run_claims_allowed:
        blockers.append("runs:performance_boundary_still_closed")
    allowed = (
        bool(calibration_audit["performance_claim_allowed"])
        and deterministic
        and run_claims_allowed
    )
    return {
        "schema_version": "hetero-p17-performance-calibration-qualification/v1",
        "calibration_record": str(calibration_path),
        "run_count": len(snapshots),
        "runs": snapshots,
        "deterministic_runs": deterministic,
        "calibration_audit": calibration_audit,
        "blockers": blockers,
        "performance_claim_allowed": allowed,
        "status": "qualified" if allowed else "audit_complete_blocked",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-qualified", action="store_true")
    args = parser.parse_args()
    record = build_audit(
        args.calibration.resolve(),
        [path.resolve() for path in args.runs],
        args.project_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"status={record['status']}; performance_claim_allowed="
        f"{str(record['performance_claim_allowed']).lower()}; "
        f"blockers={len(record['blockers'])}"
    )
    return (
        1 if args.require_qualified and not record["performance_claim_allowed"] else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
