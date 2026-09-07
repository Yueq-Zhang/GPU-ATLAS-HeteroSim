import copy
import json
from pathlib import Path

from frontend.hetero.gpu_execution_identity import (
    EXECUTION_IDENTITY_SCHEMA,
    load_execution_identity_catalog,
    trace_kernel_sequence,
)
from frontend.hetero.gpu_operator_calibration import (
    NATIVE_SCHEMA,
    audit_gpu_operator_pairing,
    build_native_vram_simulator_catalog,
    build_simulator_measurement_catalog,
    file_sha256,
    load_gpu_operator_contracts,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _capabilities() -> Path:
    return (
        _root() / "configs/hetero/operator_capabilities/"
        "tinyllama_prefill_layer0_bs1_ctx16.json"
    )


def _simulator() -> dict[str, object]:
    return build_simulator_measurement_catalog(
        _capabilities(),
        _root(),
        core_frequency_hz=1_132_000_000,
        memory_topology="external_shared_3ddram",
    )


def _native_from_simulator(simulator: dict[str, object]) -> dict[str, object]:
    operators = []
    for item in simulator["operators"]:
        simulator_identity = {
            "schema_version": EXECUTION_IDENTITY_SCHEMA,
            "executable_sha256": item["operator_artifact_sha256"],
            "launch_contract_sha256": item["trace_manifest_sha256"],
            "kernel_sequence_sha256": item["trace_manifest_sha256"],
            "target_sm": 86,
            "kernel_launch_count": 1,
            "native_measurement_observed": False,
            "trace_capture_observed": True,
            "provenance": {"source": "unit_test_trace"},
        }
        item["execution_identity"] = simulator_identity
        native_identity = copy.deepcopy(simulator_identity)
        native_identity["native_measurement_observed"] = True
        native_identity["provenance"] = {"source": "unit_test_native"}
        operators.append(
            {
                "operator_type": item["operator_type"],
                "implementation": item["implementation"],
                "shape_key": item["shape_key"],
                "operator_artifact": item["operator_artifact"],
                "operator_artifact_sha256": item["operator_artifact_sha256"],
                "trace_manifest_sha256": item["trace_manifest_sha256"],
                "execution_identity": native_identity,
                "operator_latency_fs": {
                    "min": item["operator_latency_fs"],
                    "p10": item["operator_latency_fs"],
                    "median": item["operator_latency_fs"],
                    "p90": item["operator_latency_fs"],
                    "max": item["operator_latency_fs"],
                    "mean": item["operator_latency_fs"],
                },
                "repetitions": 20,
                "statistic": "median",
            }
        )
    return {
        "schema_version": NATIVE_SCHEMA,
        "catalog_id": "native.test",
        "measurement_scope": {"memory_topology": "external_shared_3ddram"},
        "operators": operators,
        "operator_count": len(operators),
    }


def test_repository_has_fourteen_exact_gpu_operator_contracts() -> None:
    contracts = load_gpu_operator_contracts(_capabilities(), _root())
    assert len(contracts) == 14
    assert contracts["token_embedding"]["implementation"] == (
        "heterosim_cuda_reference_token_embedding_v1"
    )
    assert contracts["attention_norm"]["implementation"] == (
        "transformers.LlamaRMSNorm"
    )


def test_build_simulator_catalog_seals_artifact_cycles() -> None:
    payload = _simulator()
    assert payload["operator_count"] == 14
    records = {item["operator_type"]: item for item in payload["operators"]}
    assert records["token_embedding"]["cycles"] == 6691
    assert records["residual_add"]["cycles"] == 28772
    assert all(item["operator_artifact_sha256"] for item in records.values())


def test_pairing_passes_only_for_exact_same_topology_and_contract() -> None:
    simulator = _simulator()
    native = _native_from_simulator(simulator)
    audit = audit_gpu_operator_pairing(
        native, simulator, _capabilities(), _root(), max_relative_error=0.01
    )
    assert audit["gpu_operator_calibration_complete"] is True
    assert audit["paired_operator_count"] == 14
    assert audit["blockers"] == []


def test_pairing_rejects_native_vram_vs_external_3ddram() -> None:
    simulator = _simulator()
    native = _native_from_simulator(simulator)
    native["measurement_scope"]["memory_topology"] = "gpu_local_vram"
    audit = audit_gpu_operator_pairing(
        native, simulator, _capabilities(), _root(), max_relative_error=0.01
    )
    assert audit["gpu_operator_calibration_complete"] is False
    assert audit["paired_operator_count"] == 0
    assert (
        "memory_topology_mismatch:gpu_local_vram!=external_shared_3ddram"
        in (audit["blockers"])
    )


def test_pairing_rejects_artifact_hash_drift() -> None:
    simulator = _simulator()
    native = _native_from_simulator(simulator)
    changed = copy.deepcopy(native)
    changed["operators"][0]["operator_artifact_sha256"] = "0" * 64
    audit = audit_gpu_operator_pairing(
        changed, simulator, _capabilities(), _root(), max_relative_error=0.01
    )
    assert audit["performance_claim_allowed"] is False
    assert any("artifact_sha256_mismatch" in item for item in audit["blockers"])


def test_pairing_rejects_same_shape_with_different_executable() -> None:
    simulator = _simulator()
    native = _native_from_simulator(simulator)
    native["operators"][0]["execution_identity"]["executable_sha256"] = "0" * 64
    audit = audit_gpu_operator_pairing(
        native, simulator, _capabilities(), _root(), max_relative_error=0.01
    )
    assert audit["performance_claim_allowed"] is False
    assert any("execution_identity_mismatch" in item for item in audit["blockers"])


def test_pairing_rejects_unobserved_native_execution_identity() -> None:
    simulator = _simulator()
    native = _native_from_simulator(simulator)
    native["operators"][0]["execution_identity"][
        "native_measurement_observed"
    ] = False
    audit = audit_gpu_operator_pairing(
        native, simulator, _capabilities(), _root(), max_relative_error=0.01
    )
    assert audit["performance_claim_allowed"] is False
    assert any(
        "native:measurement_identity_not_observed" in item
        for item in audit["blockers"]
    )


def test_build_native_vram_catalog_requires_exact_double_runs(tmp_path: Path) -> None:
    contracts = load_gpu_operator_contracts(_capabilities(), _root())
    simulated = {item["operator_type"]: item for item in _simulator()["operators"]}
    for operator, contract in contracts.items():
        artifact_path = Path(contract["artifact_path"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        trace_item = next(
            item
            for item in artifact["files"]
            if item["kind"] == "accel_sim_trace_manifest"
        )
        trace_path = Path(trace_item["path"])
        if not trace_path.is_absolute():
            trace_path = artifact_path.parent / trace_path
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        output = tmp_path / f"{operator.replace('_', '-')}-native-vram"
        output.mkdir()
        cycles = simulated[operator]["cycles"]
        (output / "qualification_record.json").write_text(
            json.dumps(
                {
                    "schema_version": "hetero-accel-sim-qualification/v1",
                    "status": "passed",
                    "target_sm": 86,
                    "trace_id": trace["trace_id"],
                    "provenance": {"trace_manifest": str(trace_path)},
                    "comparison": {
                        "gpu_tot_sim_cycle": [cycles, cycles],
                        "gpu_tot_sim_insn": [100, 100],
                    },
                    "timing_ownership": {
                        "gpu_local_dram": "accel_sim",
                        "external_ramulator2": False,
                        "duration_mode": "total",
                    },
                }
            ),
            encoding="utf-8",
        )
    catalog = build_native_vram_simulator_catalog(
        _capabilities(), _root(), tmp_path, core_frequency_hz=1_132_000_000
    )
    assert catalog["operator_count"] == 14
    assert catalog["measurement_scope"]["memory_topology"] == "gpu_local_vram"
    assert all(item["qualification_record_sha256"] for item in catalog["operators"])


def test_native_vram_catalog_seals_the_trace_actually_qualified(tmp_path: Path) -> None:
    contracts = load_gpu_operator_contracts(_capabilities(), _root())
    simulated = {item["operator_type"]: item for item in _simulator()["operators"]}
    replacement_manifest: Path | None = None
    for operator, contract in contracts.items():
        artifact_path = Path(contract["artifact_path"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        trace_item = next(
            item
            for item in artifact["files"]
            if item["kind"] == "accel_sim_trace_manifest"
        )
        trace_path = Path(trace_item["path"])
        if not trace_path.is_absolute():
            trace_path = artifact_path.parent / trace_path
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        qualified_trace_path = trace_path
        if operator == "token_embedding":
            replacement_manifest = tmp_path / "recaptured-token-embedding.json"
            replacement_manifest.write_text(
                json.dumps({**trace, "kernels_list": "/new/capture/kernelslist.g"}),
                encoding="utf-8",
            )
            qualified_trace_path = replacement_manifest
        output = tmp_path / f"{operator.replace('_', '-')}-native-vram"
        output.mkdir()
        cycles = simulated[operator]["cycles"]
        (output / "qualification_record.json").write_text(
            json.dumps(
                {
                    "schema_version": "hetero-accel-sim-qualification/v1",
                    "status": "passed",
                    "target_sm": 86,
                    "trace_id": trace["trace_id"],
                    "provenance": {"trace_manifest": str(qualified_trace_path)},
                    "comparison": {
                        "gpu_tot_sim_cycle": [cycles, cycles],
                        "gpu_tot_sim_insn": [100, 100],
                    },
                    "timing_ownership": {
                        "gpu_local_dram": "accel_sim",
                        "external_ramulator2": False,
                        "duration_mode": "total",
                    },
                }
            ),
            encoding="utf-8",
        )
    catalog = build_native_vram_simulator_catalog(
        _capabilities(), _root(), tmp_path, core_frequency_hz=1_132_000_000
    )
    token = next(
        item
        for item in catalog["operators"]
        if item["operator_type"] == "token_embedding"
    )
    assert replacement_manifest is not None
    assert token["trace_manifest"] == str(replacement_manifest)
    assert token["trace_manifest_sha256"] == file_sha256(replacement_manifest)
    assert token["contract_trace_manifest"] != token["trace_manifest"]


def test_p17_sealed_sm86_recapture_record_matches_repository_manifests() -> None:
    record_path = _root() / "validation/p17/sm86_sealed_recapture/recapture_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["target_binary"]["embedded_cubins"] == ["sm_86", "sm_86"]
    assert record["capture_host"]["gpu"] == "NVIDIA GeForce RTX 3070"
    assert record["claim_boundary"]["trace_target_sm"] == 86
    assert record["claim_boundary"][
        "physical_capture_gpu_is_not_the_simulated_gpu"
    ] is False
    assert record["claim_boundary"][
        "native_rtx3070_binary_identity_verified"
    ] is True
    assert record["claim_boundary"]["performance_pairing_allowed"] is False
    for operator, evidence in record["operators"].items():
        manifest_path = (
            _root()
            / "configs/hetero/operator_artifacts/p17_sealed"
            / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["compilation"]["target_sm"] == 86
        assert evidence["trace_manifest_sha256"] == file_sha256(manifest_path)
        native = record["native_measurements"][operator]
        assert file_sha256(_root() / native["path"]) == native["sha256"]


def test_p17_sealed_operator_execution_identity_is_same_program() -> None:
    catalog_path = (
        _root()
        / "validation/p17/sm86_sealed_recapture/execution_identity_catalog.json"
    )
    catalog_bytes = catalog_path.read_bytes()
    assert catalog_bytes.endswith(b"\n")
    assert b"\r\n" not in catalog_bytes
    identities = load_execution_identity_catalog(catalog_path)
    assert set(identities) == set(
        load_gpu_operator_contracts(_capabilities(), _root())
    )
    for operator, identity in identities.items():
        kernels = (
            _root()
            / "configs/hetero/operator_artifacts/p17_sealed/evidence/"
            "local_rtx3070_same_binary_v1"
            / operator
            / "traces/kernelslist.g"
        )
        sequence_sha256, descriptors = trace_kernel_sequence(kernels)
        assert identity["kernel_sequence_sha256"] == sequence_sha256
        assert identity["kernel_launch_count"] == len(descriptors)
        assert descriptors
        assert identity["trace_capture_observed"] is True
        assert identity["native_measurement_observed"] is True


def test_p17_native_vram_import_manifest_seals_all_qualification_records() -> None:
    root = _root() / "validation/p17/native_vram_qualification"
    manifest = json.loads((root / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["qualified_operator_count"] == 14
    assert manifest["pairing_audit"]["topology_match"] is True
    assert manifest["pairing_audit"]["paired_operator_count"] == 4
    for operator, expected_sha256 in manifest["records"].items():
        record = (
            root
            / f"{operator.replace('_', '-')}-native-vram"
            / "qualification_record.json"
        )
        assert file_sha256(record) == expected_sha256
        payload = json.loads(record.read_text(encoding="utf-8"))
        assert payload["status"] == "passed"
        assert len(set(payload["comparison"]["gpu_tot_sim_cycle"])) == 1
        assert len(set(payload["comparison"]["gpu_tot_sim_insn"])) == 1
        assert payload["timing_ownership"]["gpu_local_dram"] == "accel_sim"
        assert payload["timing_ownership"]["external_ramulator2"] is False
    for artifact in (
        "simulator_catalog",
        "execution_identity_catalog",
        "pairing_audit",
    ):
        evidence = manifest[artifact]
        assert file_sha256(_root() / evidence["path"]) == evidence["sha256"]


def test_p17_measurement_manifest_tracks_current_same_binary_evidence() -> None:
    root = _root()
    manifest = json.loads(
        (
            root
            / "validation/p17/gpu_operator_pairing/measurement_manifest.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "native_catalog_sha256": (
            "validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json"
        ),
        "execution_identity_catalog_sha256": (
            "validation/p17/sm86_sealed_recapture/execution_identity_catalog.json"
        ),
        "simulator_catalog_sha256": (
            "validation/p17/gpu_operator_pairing/simulator_native_vram.json"
        ),
        "pairing_audit_sha256": (
            "validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json"
        ),
    }
    for field, locator in expected.items():
        assert manifest[field] == file_sha256(root / locator)
    assert manifest["native_memory_topology"] == "gpu_local_vram"
    assert manifest["simulator_memory_topology"] == "gpu_local_vram"
    assert manifest["same_binary_native_operator_count"] == 14
    assert manifest["performance_eligible"] is False


def test_imported_native_vram_catalog_rebases_remote_trace_locators() -> None:
    catalog = build_native_vram_simulator_catalog(
        _capabilities(),
        _root(),
        _root() / "validation/p17/native_vram_qualification",
        core_frequency_hz=1_132_000_000,
        execution_identity_catalog=(
            _root()
            / "validation/p17/sm86_sealed_recapture/execution_identity_catalog.json"
        ),
    )
    identities = {
        item["operator_type"]: item["execution_identity"]
        for item in catalog["operators"]
        if "execution_identity" in item
    }
    assert set(identities) == set(
        load_gpu_operator_contracts(_capabilities(), _root())
    )
    assert all(item["trace_capture_observed"] for item in identities.values())
    assert all(item["native_measurement_observed"] for item in identities.values())
