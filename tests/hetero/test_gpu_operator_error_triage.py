import json
from pathlib import Path

import pytest

from frontend.hetero.gpu_operator_error_triage import (
    GPUOperatorErrorTriageError,
    build_gpu_operator_error_triage,
)
from frontend.hetero.gpu_operator_calibration import file_sha256


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _payload(relative: str) -> dict[str, object]:
    return json.loads((_root() / relative).read_text(encoding="utf-8"))


def test_p18_current_error_triage_is_identity_and_topology_closed() -> None:
    report = build_gpu_operator_error_triage(
        _payload("validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json"),
        _payload("validation/p17/gpu_operator_pairing/simulator_native_vram.json"),
    )
    summary = report["summary"]
    assert summary["operator_count"] == 14
    assert summary["paired_operator_count"] == 4
    assert summary["blocked_operator_count"] == 10
    assert summary["kernel_launch_count"] == 63
    assert summary["simulator_under_native_count"] == 11
    assert summary["simulator_over_native_count"] == 3
    assert summary["maximum_error_operator"] == "silu_multiply"
    assert report["claim_boundary"].startswith("This report classifies")
    operators = {row["operator_type"]: row for row in report["operators"]}
    assert operators["lm_head"]["absolute_error_percent"] == pytest.approx(
        3.7964838162289655
    )
    assert operators["causal_attention"]["direction"] == "simulator_over_native"
    assert all(row["execution_identity_match"] for row in report["operators"])
    assert all(row["topology_match"] for row in report["operators"])


def test_p18_triage_rejects_operator_coverage_drift() -> None:
    audit = _payload(
        "validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json"
    )
    simulator = _payload(
        "validation/p17/gpu_operator_pairing/simulator_native_vram.json"
    )
    simulator["operators"] = simulator["operators"][:-1]
    with pytest.raises(GPUOperatorErrorTriageError, match="coverage mismatch"):
        build_gpu_operator_error_triage(audit, simulator)


def test_p18_checked_in_report_matches_hashed_inputs() -> None:
    root = _root()
    audit = _payload(
        "validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json"
    )
    simulator = _payload(
        "validation/p17/gpu_operator_pairing/simulator_native_vram.json"
    )
    report = _payload("validation/p18/gpu_operator_error_triage.json")
    inputs = report.pop("inputs")
    assert report == build_gpu_operator_error_triage(audit, simulator)
    for name in ("pairing_audit", "simulator_catalog"):
        path = root / inputs[name]
        assert path.is_file()
        assert file_sha256(path) == inputs[f"{name}_sha256"]
