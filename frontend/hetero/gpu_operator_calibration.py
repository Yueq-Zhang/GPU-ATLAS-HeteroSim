"""Exact-contract GPU operator calibration catalogs and pairing audits.

The request-cycle artifacts model a GPU attached to the shared 3D-DRAM path.
Native CUDA events on the physical RTX 3070 instead observe its local VRAM.
Those two numbers are useful evidence, but they are not a calibration pair
until operator identity, implementation, shape *and* memory topology match.
This module makes that boundary machine-readable and fail closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


class GPUOperatorCalibrationError(ValueError):
    """Raised when an operator calibration catalog is incomplete or ambiguous."""


NATIVE_SCHEMA = "hetero-gpu-operator-native-measurements/v1"
SIMULATOR_SCHEMA = "hetero-gpu-operator-simulator-measurements/v1"
PAIRING_SCHEMA = "hetero-gpu-operator-calibration-audit/v1"


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GPUOperatorCalibrationError(f"{path} must be an object")
    return value


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GPUOperatorCalibrationError(f"{path} must be a non-empty string")
    return value


def _positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GPUOperatorCalibrationError(f"{path} must be a positive integer")
    return value


def _positive_number(value: object, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) <= 0.0
    ):
        raise GPUOperatorCalibrationError(f"{path} must be positive")
    return float(value)


def _sha256(value: object, path: str) -> str:
    digest = _nonempty(value, path)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise GPUOperatorCalibrationError(f"{path} must be lowercase SHA-256")
    return digest


def _json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GPUOperatorCalibrationError(f"failed to load {path}: {error}") from error
    return _mapping(payload, str(path))


def file_sha256(path: str | Path) -> str:
    """Return a lowercase content digest for one evidence artifact."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_gpu_operator_contracts(
    capability_path: str | Path,
    repository_root: str | Path,
) -> dict[str, dict[str, object]]:
    """Load the exact 14 Accel-Sim operator contracts from a capability catalog."""

    capability_file = Path(capability_path).resolve()
    root = Path(repository_root).resolve()
    payload = _json(capability_file)
    if payload.get("schema_version") != "hetero-operator-capability-catalog/v1":
        raise GPUOperatorCalibrationError("invalid operator capability schema")
    raw_operators = payload.get("operator_types")
    if not isinstance(raw_operators, list):
        raise GPUOperatorCalibrationError("operator_types must be an array")
    contracts: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_operators):
        item = _mapping(raw, f"operator_types[{index}]")
        if item.get("backend_kind") != "accel_sim":
            continue
        operator = _nonempty(item.get("operator_type"), "operator_type")
        refs = item.get("artifact_refs")
        if not isinstance(refs, list) or len(refs) != 1:
            raise GPUOperatorCalibrationError(
                f"{operator} must bind exactly one operator artifact"
            )
        artifact_locator = _nonempty(refs[0], f"{operator}.artifact_refs[0]")
        artifact_path = _resolve(artifact_locator, root)
        artifact = _json(artifact_path)
        source = _mapping(artifact.get("source_contract"), "source_contract")
        artifact_operator = _nonempty(
            source.get("operator"), "source_contract.operator"
        )
        if artifact_operator != operator:
            raise GPUOperatorCalibrationError(
                f"artifact operator mismatch: {operator} != {artifact_operator}"
            )
        contracts[operator] = {
            "operator_type": operator,
            "implementation": _nonempty(
                source.get("implementation"), "source_contract.implementation"
            ),
            "model_spec_name": _nonempty(
                source.get("model_spec_name"), "source_contract.model_spec_name"
            ),
            "checkpoint_revision": _nonempty(
                source.get("checkpoint_revision"),
                "source_contract.checkpoint_revision",
            ),
            "phase": _nonempty(source.get("phase"), "source_contract.phase"),
            "dtype": _nonempty(source.get("dtype"), "source_contract.dtype"),
            "layer_id": source.get("layer_id"),
            "batch_size": source.get("batch_size"),
            "context_length": source.get("context_length"),
            "q_len": source.get("q_len"),
            "kv_length": source.get("kv_length"),
            "artifact_locator": artifact_locator,
            "artifact_path": str(artifact_path),
            "artifact_sha256": file_sha256(artifact_path),
            "artifact_id": _nonempty(artifact.get("artifact_id"), "artifact_id"),
        }
    if len(contracts) != 14:
        raise GPUOperatorCalibrationError(
            f"expected 14 Accel-Sim operator contracts, found {len(contracts)}"
        )
    return contracts


def exact_shape_key(contract: Mapping[str, object]) -> str:
    """Build a stable shape key from one operator source contract."""

    return (
        f"{contract['model_spec_name']}.{contract['phase']}.layer{contract['layer_id']}."
        f"{contract['dtype']}.bs{contract['batch_size']}.ctx{contract['context_length']}."
        f"q{contract['q_len']}.kv{contract['kv_length']}.{contract['operator_type']}"
    ).lower()


def _trace_manifest_for_contract(
    contract: Mapping[str, object],
) -> tuple[Path, Mapping[str, object]]:
    artifact_path = Path(str(contract["artifact_path"]))
    artifact = _json(artifact_path)
    raw_files = artifact.get("files")
    if not isinstance(raw_files, list):
        raise GPUOperatorCalibrationError("operator artifact files must be an array")
    trace_files = [
        _mapping(item, "files[]")
        for item in raw_files
        if isinstance(item, Mapping) and item.get("kind") == "accel_sim_trace_manifest"
    ]
    if len(trace_files) != 1:
        raise GPUOperatorCalibrationError(
            f"expected one trace manifest for {contract['operator_type']}"
        )
    trace_path = _resolve(
        _nonempty(trace_files[0].get("path"), "trace manifest path"),
        artifact_path.parent,
    )
    return trace_path, _json(trace_path)


def build_simulator_measurement_catalog(
    capability_path: str | Path,
    repository_root: str | Path,
    *,
    core_frequency_hz: int,
    memory_topology: str,
) -> dict[str, object]:
    """Extract cycle evidence already sealed into the qualified artifacts."""

    frequency = _positive_int(core_frequency_hz, "core_frequency_hz")
    topology = _nonempty(memory_topology, "memory_topology")
    contracts = load_gpu_operator_contracts(capability_path, repository_root)
    operators: list[dict[str, object]] = []
    for operator, contract in sorted(contracts.items()):
        artifact = _json(Path(str(contract["artifact_path"])))
        trace_path, trace = _trace_manifest_for_contract(contract)
        qualification = _mapping(artifact.get("qualification"), "qualification")
        cycles = _positive_int(qualification.get("cycles"), "qualification.cycles")
        operators.append(
            {
                "operator_type": operator,
                "implementation": contract["implementation"],
                "shape_key": exact_shape_key(contract),
                "operator_artifact": contract["artifact_locator"],
                "operator_artifact_sha256": contract["artifact_sha256"],
                "trace_id": trace.get("trace_id"),
                "trace_manifest": str(trace_path),
                "trace_manifest_sha256": file_sha256(trace_path),
                "cycles": cycles,
                "operator_latency_fs": round(cycles * 1.0e15 / frequency),
                "source_status": qualification.get("status"),
            }
        )
    first = next(iter(contracts.values()))
    return {
        "schema_version": SIMULATOR_SCHEMA,
        "catalog_id": "p17.tinyllama.layer0.ctx16.shared3d.qualified_artifacts",
        "measurement_scope": {
            "memory_topology": topology,
            "timing_owner": "accel_sim.sm86.compute_cache_plus_memory_path",
        },
        "model_contract": {
            "model_spec_name": first["model_spec_name"],
            "checkpoint_revision": first["checkpoint_revision"],
            "dtype": first["dtype"],
        },
        "protocol": {
            "core_frequency_hz": frequency,
            "statistic": "qualified_deterministic_cycle_count",
            "repetitions": 2,
        },
        "operators": operators,
        "operator_count": len(operators),
        "performance_qualification": {
            "eligible": False,
            "reason": (
                "requires exact native measurement pairing with the same memory topology"
            ),
        },
    }


def build_native_vram_simulator_catalog(
    capability_path: str | Path,
    repository_root: str | Path,
    qualification_root: str | Path,
    *,
    core_frequency_hz: int,
) -> dict[str, object]:
    """Build a catalog from deterministic native-VRAM Accel-Sim double runs."""

    frequency = _positive_int(core_frequency_hz, "core_frequency_hz")
    root = Path(repository_root).resolve()
    qualification_directory = Path(qualification_root).resolve()
    contracts = load_gpu_operator_contracts(capability_path, root)
    operators: list[dict[str, object]] = []
    for operator, contract in sorted(contracts.items()):
        record_path = (
            qualification_directory
            / f"{operator.replace('_', '-')}-native-vram"
            / "qualification_record.json"
        )
        qualification = _json(record_path)
        if (
            qualification.get("schema_version") != "hetero-accel-sim-qualification/v1"
            or qualification.get("status") != "passed"
            or qualification.get("target_sm") != 86
        ):
            raise GPUOperatorCalibrationError(
                f"invalid native-VRAM qualification for {operator}"
            )
        comparison = _mapping(qualification.get("comparison"), "comparison")
        cycles_raw = comparison.get("gpu_tot_sim_cycle")
        instructions_raw = comparison.get("gpu_tot_sim_insn")
        if (
            not isinstance(cycles_raw, list)
            or len(cycles_raw) != 2
            or cycles_raw[0] != cycles_raw[1]
            or not isinstance(cycles_raw[0], int)
            or cycles_raw[0] <= 0
            or not isinstance(instructions_raw, list)
            or len(instructions_raw) != 2
            or instructions_raw[0] != instructions_raw[1]
        ):
            raise GPUOperatorCalibrationError(
                f"non-deterministic native-VRAM qualification for {operator}"
            )
        if "external_memory_stats" in comparison:
            raise GPUOperatorCalibrationError(
                f"native-VRAM qualification unexpectedly has external memory: {operator}"
            )
        ownership = _mapping(qualification.get("timing_ownership"), "timing_ownership")
        if (
            ownership.get("gpu_local_dram") != "accel_sim"
            or ownership.get("external_ramulator2") is not False
            or ownership.get("duration_mode") != "total"
        ):
            raise GPUOperatorCalibrationError(
                f"invalid native-VRAM timing ownership for {operator}"
            )
        trace_path, trace = _trace_manifest_for_contract(contract)
        if qualification.get("trace_id") != trace.get("trace_id"):
            raise GPUOperatorCalibrationError(
                f"trace identity mismatch in qualification for {operator}"
            )
        cycles = cycles_raw[0]
        operators.append(
            {
                "operator_type": operator,
                "implementation": contract["implementation"],
                "shape_key": exact_shape_key(contract),
                "operator_artifact": contract["artifact_locator"],
                "operator_artifact_sha256": contract["artifact_sha256"],
                "qualification_record": str(record_path),
                "qualification_record_sha256": file_sha256(record_path),
                "trace_manifest": str(trace_path),
                "trace_manifest_sha256": file_sha256(trace_path),
                "cycles": cycles,
                "instructions": instructions_raw[0],
                "operator_latency_fs": round(cycles * 1.0e15 / frequency),
                "source_status": "passed_native_vram_double_run",
            }
        )
    first = next(iter(contracts.values()))
    return {
        "schema_version": SIMULATOR_SCHEMA,
        "catalog_id": "p17.tinyllama.layer0.ctx16.native_vram.double_qualified",
        "measurement_scope": {
            "memory_topology": "gpu_local_vram",
            "timing_owner": "accel_sim.sm86.compute_cache_and_native_gpu_memory",
        },
        "model_contract": {
            "model_spec_name": first["model_spec_name"],
            "checkpoint_revision": first["checkpoint_revision"],
            "dtype": first["dtype"],
        },
        "protocol": {
            "core_frequency_hz": frequency,
            "statistic": "deterministic_double_run_cycle_count",
            "repetitions": 2,
        },
        "operators": operators,
        "operator_count": len(operators),
        "performance_qualification": {
            "eligible": False,
            "reason": "requires exact native measurement error audit",
        },
    }


def _operator_records(
    payload: Mapping[str, object], schema: str, path: str
) -> dict[str, Mapping[str, object]]:
    if payload.get("schema_version") != schema:
        raise GPUOperatorCalibrationError(f"invalid {path} schema_version")
    scope = _mapping(payload.get("measurement_scope"), f"{path}.measurement_scope")
    _nonempty(scope.get("memory_topology"), f"{path}.memory_topology")
    raw_records = payload.get("operators")
    if not isinstance(raw_records, list):
        raise GPUOperatorCalibrationError(f"{path}.operators must be an array")
    records: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(raw_records):
        record = _mapping(raw, f"{path}.operators[{index}]")
        operator = _nonempty(record.get("operator_type"), "operator_type")
        if operator in records:
            raise GPUOperatorCalibrationError(f"duplicate {path} operator {operator}")
        _nonempty(record.get("implementation"), f"{operator}.implementation")
        _nonempty(record.get("shape_key"), f"{operator}.shape_key")
        _sha256(
            record.get("operator_artifact_sha256"),
            f"{operator}.operator_artifact_sha256",
        )
        if schema == NATIVE_SCHEMA:
            summary = _mapping(record.get("operator_latency_fs"), "operator_latency_fs")
            _positive_number(summary.get("median"), f"{operator}.median")
            _positive_int(record.get("repetitions"), f"{operator}.repetitions")
        else:
            _positive_int(record.get("cycles"), f"{operator}.cycles")
            _positive_number(
                record.get("operator_latency_fs"), f"{operator}.operator_latency_fs"
            )
        records[operator] = record
    count = payload.get("operator_count")
    if count != len(records):
        raise GPUOperatorCalibrationError(f"{path}.operator_count is inconsistent")
    return records


def audit_gpu_operator_pairing(
    native_payload: Mapping[str, object],
    simulator_payload: Mapping[str, object],
    capability_path: str | Path,
    repository_root: str | Path,
    *,
    max_relative_error: float = 0.15,
) -> dict[str, object]:
    """Pair exact native and simulated operators, rejecting topology drift."""

    if not 0.0 <= max_relative_error < 1.0:
        raise GPUOperatorCalibrationError("max_relative_error must be in [0, 1)")
    contracts = load_gpu_operator_contracts(capability_path, repository_root)
    native = _operator_records(native_payload, NATIVE_SCHEMA, "native")
    simulated = _operator_records(simulator_payload, SIMULATOR_SCHEMA, "simulator")
    native_scope = _mapping(native_payload["measurement_scope"], "measurement_scope")
    simulator_scope = _mapping(
        simulator_payload["measurement_scope"], "measurement_scope"
    )
    native_topology = str(native_scope["memory_topology"])
    simulator_topology = str(simulator_scope["memory_topology"])
    topology_match = native_topology == simulator_topology
    blockers: list[str] = []
    if not topology_match:
        blockers.append(
            f"memory_topology_mismatch:{native_topology}!={simulator_topology}"
        )
    expected = set(contracts)
    for kind, records in (("native", native), ("simulator", simulated)):
        for missing in sorted(expected - set(records)):
            blockers.append(f"{kind}:missing_operator={missing}")
        for extra in sorted(set(records) - expected):
            blockers.append(f"{kind}:unexpected_operator={extra}")
    audited: list[dict[str, object]] = []
    for operator, contract in sorted(contracts.items()):
        native_record = native.get(operator)
        simulator_record = simulated.get(operator)
        record_blockers: list[str] = []
        if native_record is None or simulator_record is None:
            audited.append(
                {
                    "operator_type": operator,
                    "paired": False,
                    "blockers": ["missing_measurement_record"],
                }
            )
            continue
        expected_shape = exact_shape_key(contract)
        for kind, record in (
            ("native", native_record),
            ("simulator", simulator_record),
        ):
            if record["implementation"] != contract["implementation"]:
                record_blockers.append(f"{kind}:implementation_mismatch")
            if record["shape_key"] != expected_shape:
                record_blockers.append(f"{kind}:shape_mismatch")
            if record["operator_artifact_sha256"] != contract["artifact_sha256"]:
                record_blockers.append(f"{kind}:artifact_sha256_mismatch")
        native_trace_sha = native_record.get("trace_manifest_sha256")
        simulator_trace_sha = simulator_record.get("trace_manifest_sha256")
        if not isinstance(native_trace_sha, str):
            record_blockers.append("native:trace_binary_identity_unverified")
        elif native_trace_sha != simulator_trace_sha:
            record_blockers.append("trace_binary_identity_mismatch")
        if not topology_match:
            record_blockers.append("memory_topology_mismatch")
        measured = float(
            _mapping(native_record["operator_latency_fs"], "operator_latency_fs")[
                "median"
            ]
        )
        simulated_latency = float(simulator_record["operator_latency_fs"])
        relative_error = abs(simulated_latency - measured) / measured
        within_tolerance = relative_error <= max_relative_error
        if not within_tolerance:
            record_blockers.append("relative_error_exceeds_tolerance")
        paired = not record_blockers
        blockers.extend(f"operator:{operator}:{item}" for item in record_blockers)
        audited.append(
            {
                "operator_type": operator,
                "implementation": contract["implementation"],
                "shape_key": expected_shape,
                "measured_latency_fs": round(measured),
                "simulated_latency_fs": round(simulated_latency),
                "relative_error": relative_error,
                "max_relative_error": max_relative_error,
                "within_tolerance": within_tolerance,
                "topology_match": topology_match,
                "paired": paired,
                "blockers": record_blockers,
            }
        )
    paired_count = sum(bool(item["paired"]) for item in audited)
    complete = paired_count == len(contracts) and not blockers
    return {
        "schema_version": PAIRING_SCHEMA,
        "native_catalog_id": native_payload.get("catalog_id"),
        "simulator_catalog_id": simulator_payload.get("catalog_id"),
        "native_memory_topology": native_topology,
        "simulator_memory_topology": simulator_topology,
        "topology_match": topology_match,
        "required_operator_count": len(contracts),
        "paired_operator_count": paired_count,
        "operators": audited,
        "blockers": sorted(set(blockers)),
        "gpu_operator_calibration_complete": complete,
        "performance_claim_allowed": complete,
    }
