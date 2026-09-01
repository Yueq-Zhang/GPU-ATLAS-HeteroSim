import hashlib
import json
from pathlib import Path

from frontend.hetero.performance_calibration import (
    PerformanceCalibration,
    evaluate_performance_gate,
)
from frontend.hetero.schema import validate_config
from scripts.audit_p17_performance_calibration import build_audit


def _payload(
    config_sha256: str,
    measurement_sha256: str,
    status: str = "validated",
) -> dict[str, object]:
    return {
        "schema_version": "hetero-performance-calibration/v1",
        "calibration_id": "test.calibration",
        "target_system": {"gpu": "test"},
        "scope": {"shape": "shape.test"},
        "qualification_policy": {
            "required_shape_key": "shape.test",
            "required_components": ["gpu_kernel"],
            "required_metrics": {"gpu_kernel": ["kernel_latency_fs"]},
            "allowed_validation_evidence": ["hardware_measurement"],
            "minimum_reference_points_per_component": 1,
        },
        "configuration_sources": [
            {
                "source_id": "config",
                "path": "config.json",
                "sha256": config_sha256,
            }
        ],
        "components": {
            "gpu_kernel": {
                "required": True,
                "status": status,
                "timing_owner": "gpu0",
                "parameter_bindings": [
                    {
                        "name": "clock_hz",
                        "configured_value": 1_000_000_000,
                        "unit": "Hz",
                        "configuration_source_id": "config",
                    }
                ],
                "sources": [
                    {
                        "source_id": "measurement",
                        "evidence_class": "hardware_measurement",
                        "locator": "measurements/gpu.json",
                        "description": "Repeated synchronized hardware measurement",
                        "artifact_sha256": measurement_sha256,
                    }
                ],
                "reference_points": [
                    {
                        "reference_id": "kernel.a",
                        "source_id": "measurement",
                        "metric": "kernel_latency_fs",
                        "measured_value": 100.0,
                        "simulated_value": 105.0,
                        "unit": "fs",
                        "repetitions": 20,
                        "statistic": "median",
                        "max_relative_error": 0.1,
                        "max_absolute_error": 0.0,
                    }
                ],
                "applicable_shape_keys": ["shape.test"],
                "notes": "test",
            }
        },
    }


def test_validated_calibration_passes(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    measurement = tmp_path / "measurements/gpu.json"
    measurement.parent.mkdir()
    measurement.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    measurement_digest = hashlib.sha256(measurement.read_bytes()).hexdigest()
    record = PerformanceCalibration.from_payload(_payload(digest, measurement_digest))
    audit = record.audit(tmp_path)
    assert audit["performance_claim_allowed"] is True
    assert audit["qualified_component_count"] == 1
    point = audit["components"]["gpu_kernel"]["reference_points"][0]
    assert point["relative_error"] == 0.05
    assert point["passed"] is True


def test_calibration_fails_closed_on_status_error_and_hash(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("changed\n", encoding="utf-8")
    record = PerformanceCalibration.from_payload(
        _payload("0" * 64, "1" * 64, "specified_only")
    )
    audit = record.audit(tmp_path)
    assert audit["performance_claim_allowed"] is False
    assert "configuration:config:sha256_mismatch" in audit["blockers"]
    assert "component:gpu_kernel:status=specified_only" in audit["blockers"]
    assert (
        "component:gpu_kernel:source_artifact_missing=measurement" in audit["blockers"]
    )
    assert (
        "component:gpu_kernel:unverified_reference_sources=measurement"
        in audit["blockers"]
    )


def test_text_configuration_hash_is_stable_across_line_endings(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    normalized = b'{\n  "value": 1\n}\n'
    payload = _payload(hashlib.sha256(normalized).hexdigest(), "1" * 64)
    payload["configuration_sources"][0]["hash_mode"] = "text_lf_utf8"
    record = PerformanceCalibration.from_payload(payload)
    audit = record.audit(tmp_path)
    assert audit["configuration_sources"][0]["matched"] is True
    assert audit["configuration_sources"][0]["hash_mode"] == "text_lf_utf8"

    config.write_bytes(normalized)
    audit = record.audit(tmp_path)
    assert audit["configuration_sources"][0]["matched"] is True


def test_text_artifact_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    measurement = tmp_path / "measurements/gpu.json"
    measurement.parent.mkdir()
    measurement.write_bytes(b'{\r\n  "latency": 10\r\n}\r\n')
    normalized = b'{\n  "latency": 10\n}\n'
    payload = _payload(
        hashlib.sha256(config.read_bytes()).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
    )
    payload["components"]["gpu_kernel"]["sources"][0]["artifact_hash_mode"] = (
        "text_lf_utf8"
    )
    record = PerformanceCalibration.from_payload(payload)
    audit = record.audit(tmp_path)
    artifact = audit["components"]["gpu_kernel"]["source_artifacts"][0]
    assert artifact["matched"] is True
    assert artifact["artifact_hash_mode"] == "text_lf_utf8"

    measurement.write_bytes(normalized)
    audit = record.audit(tmp_path)
    artifact = audit["components"]["gpu_kernel"]["source_artifacts"][0]
    assert artifact["matched"] is True


def test_task_gate_excludes_host_controls_but_rejects_ineligible_device_task(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    measurement = tmp_path / "measurements/gpu.json"
    measurement.parent.mkdir()
    measurement.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    measurement_digest = hashlib.sha256(measurement.read_bytes()).hexdigest()
    record = PerformanceCalibration.from_payload(_payload(digest, measurement_digest))
    gate = evaluate_performance_gate(
        record,
        [
            {
                "task_id": "request_start",
                "fidelity": {
                    "device_performance_included": False,
                    "performance_eligible": False,
                },
            },
            {
                "task_id": "gpu_kernel",
                "fidelity": {
                    "device_performance_included": True,
                    "performance_eligible": False,
                },
            },
        ],
        tmp_path,
    )
    assert gate["component_calibration_allowed"] is True
    assert gate["performance_claim_allowed"] is False
    assert gate["task_gate"]["excluded_control_task_ids"] == ["request_start"]
    assert gate["task_gate"]["ineligible_task_ids"] == ["gpu_kernel"]


def test_repository_p17_inventory_is_intentionally_blocked() -> None:
    root = Path(__file__).resolve().parents[2]
    record = PerformanceCalibration.load(
        root
        / "configs/hetero/calibration/p17_tinyllama_prefill_layer0_ctx16_incomplete.json"
    )
    audit = record.audit(root)
    assert audit["required_component_count"] == 6
    assert audit["qualified_component_count"] == 0
    assert audit["performance_claim_allowed"] is False
    assert all(item["matched"] for item in audit["configuration_sources"])
    for component_id in ("gpu_kernel", "copy_engine", "runtime_control"):
        artifacts = audit["components"][component_id]["source_artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["matched"] is True


def test_experiment_schema_accepts_an_expanded_calibration_record() -> None:
    root = Path(__file__).resolve().parents[2]
    experiment = json.loads(
        (
            root / "validation/p16/leg1/"
            "p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/"
            "d5066ff9081332bd31ae5699f4f572736cc7f188ae9f4272cf89a4af0a1d6e3a/"
            "resolved_config.yaml"
        ).read_text(encoding="utf-8")
    )
    experiment["calibration"] = json.loads(
        (
            root / "configs/hetero/calibration/"
            "p17_tinyllama_prefill_layer0_ctx16_incomplete.json"
        ).read_text(encoding="utf-8")
    )
    validate_config(experiment)


def test_p17_run_audit_keeps_p16_performance_boundary_closed(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.json"
    measurement = tmp_path / "measurements/gpu.json"
    measurement.parent.mkdir()
    measurement.write_text("{}\n", encoding="utf-8")
    measurement_digest = hashlib.sha256(measurement.read_bytes()).hexdigest()
    calibration.write_text(
        json.dumps(_payload(digest, measurement_digest)), encoding="utf-8"
    )
    runs: list[Path] = []
    for index in range(2):
        run = tmp_path / f"leg{index}"
        run.mkdir()
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "makespan_fs": 1000,
                    "requests": [{"request_id": "R0", "ttft_fs": 900}],
                    "performance_claim_allowed": False,
                    "run_status": "operator_event",
                }
            ),
            encoding="utf-8",
        )
        (run / "provenance.json").write_text(
            json.dumps({"simulation_input_key": "same"}), encoding="utf-8"
        )
        runs.append(run)
    audit = build_audit(calibration, runs, tmp_path)
    assert audit["deterministic_runs"] is True
    assert audit["performance_claim_allowed"] is False
    assert "runs:performance_boundary_still_closed" in audit["blockers"]
