"""Build a fail-closed GPU native/simulator error-triage baseline."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean, median


ERROR_TRIAGE_SCHEMA = "hetero-gpu-operator-error-triage/v1"


class GPUOperatorErrorTriageError(ValueError):
    """Raised when pairing evidence is incomplete or internally inconsistent."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GPUOperatorErrorTriageError(f"{label} must be an object")
    return value


def _operator_map(payload: Mapping[str, object], label: str) -> dict[str, Mapping[str, object]]:
    operators = payload.get("operators")
    if not isinstance(operators, list):
        raise GPUOperatorErrorTriageError(f"{label}.operators must be an array")
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(operators):
        item = _mapping(raw, f"{label}.operators[{index}]")
        operator = str(item.get("operator_type", ""))
        if not operator or operator in result:
            raise GPUOperatorErrorTriageError(
                f"{label} contains missing or duplicate operator_type={operator!r}"
            )
        result[operator] = item
    return result


def _severity(relative_error: float, tolerance: float) -> str:
    if relative_error <= tolerance:
        return "within_tolerance"
    if relative_error <= 0.25:
        return "near_threshold"
    if relative_error <= 0.50:
        return "material_error"
    return "severe_error"


def build_gpu_operator_error_triage(
    pairing_audit: Mapping[str, object],
    simulator_catalog: Mapping[str, object],
) -> dict[str, object]:
    """Combine the strict pairing audit with simulator counters for calibration triage."""

    audit_operators = _operator_map(pairing_audit, "pairing_audit")
    simulator_operators = _operator_map(simulator_catalog, "simulator_catalog")
    if set(audit_operators) != set(simulator_operators):
        missing_simulator = sorted(set(audit_operators) - set(simulator_operators))
        missing_audit = sorted(set(simulator_operators) - set(audit_operators))
        raise GPUOperatorErrorTriageError(
            "operator coverage mismatch: "
            f"missing_simulator={missing_simulator}, missing_audit={missing_audit}"
        )

    rows: list[dict[str, object]] = []
    for operator in sorted(audit_operators):
        audit = audit_operators[operator]
        simulator = simulator_operators[operator]
        native_fs = int(audit.get("measured_latency_fs", 0))
        simulated_fs = int(audit.get("simulated_latency_fs", 0))
        if native_fs <= 0 or simulated_fs <= 0:
            raise GPUOperatorErrorTriageError(
                f"{operator} must have positive native and simulated latency"
            )
        signed_error = (simulated_fs - native_fs) / native_fs
        relative_error = abs(signed_error)
        recorded_error = float(audit.get("relative_error", -1.0))
        if abs(relative_error - recorded_error) > 1e-12:
            raise GPUOperatorErrorTriageError(
                f"{operator} relative_error drift: {relative_error} != {recorded_error}"
            )
        tolerance = float(audit.get("max_relative_error", 0.0))
        identity = _mapping(
            simulator.get("execution_identity", {}),
            f"simulator_catalog.{operator}.execution_identity",
        )
        direction = (
            "simulator_over_native" if signed_error > 0 else "simulator_under_native"
        )
        rows.append(
            {
                "operator_type": operator,
                "implementation": audit.get("implementation"),
                "shape_key": audit.get("shape_key"),
                "native_latency_fs": native_fs,
                "native_latency_us": native_fs / 1_000_000_000,
                "simulated_latency_fs": simulated_fs,
                "simulated_latency_us": simulated_fs / 1_000_000_000,
                "signed_delta_fs": simulated_fs - native_fs,
                "simulated_to_native_ratio": simulated_fs / native_fs,
                "signed_relative_error": signed_error,
                "absolute_relative_error": relative_error,
                "absolute_error_percent": relative_error * 100.0,
                "direction": direction,
                "max_relative_error": tolerance,
                "within_tolerance": bool(audit.get("within_tolerance", False)),
                "paired": bool(audit.get("paired", False)),
                "severity": _severity(relative_error, tolerance),
                "kernel_launch_count": int(identity.get("kernel_launch_count", 0)),
                "simulated_cycles": int(simulator.get("cycles", 0)),
                "simulated_instructions": int(simulator.get("instructions", 0)),
                "execution_identity_match": bool(
                    audit.get("execution_identity_match", False)
                ),
                "topology_match": bool(audit.get("topology_match", False)),
                "blockers": list(audit.get("blockers", [])),
            }
        )

    errors = [float(row["absolute_relative_error"]) for row in rows]
    under = sum(row["direction"] == "simulator_under_native" for row in rows)
    over = len(rows) - under
    severity_counts = {
        severity: sum(row["severity"] == severity for row in rows)
        for severity in (
            "within_tolerance",
            "near_threshold",
            "material_error",
            "severe_error",
        )
    }
    worst = max(rows, key=lambda row: float(row["absolute_relative_error"]))
    return {
        "schema_version": ERROR_TRIAGE_SCHEMA,
        "source_pairing_schema": pairing_audit.get("schema_version"),
        "native_catalog_id": pairing_audit.get("native_catalog_id"),
        "simulator_catalog_id": pairing_audit.get("simulator_catalog_id"),
        "memory_topology": pairing_audit.get("native_memory_topology"),
        "summary": {
            "operator_count": len(rows),
            "paired_operator_count": sum(bool(row["paired"]) for row in rows),
            "blocked_operator_count": sum(not bool(row["paired"]) for row in rows),
            "simulator_under_native_count": under,
            "simulator_over_native_count": over,
            "kernel_launch_count": sum(int(row["kernel_launch_count"]) for row in rows),
            "mean_absolute_relative_error": mean(errors),
            "median_absolute_relative_error": median(errors),
            "maximum_absolute_relative_error": float(
                worst["absolute_relative_error"]
            ),
            "maximum_error_operator": worst["operator_type"],
            "severity_counts": severity_counts,
            "performance_claim_allowed": bool(
                pairing_audit.get("performance_claim_allowed", False)
            ),
        },
        "operators": rows,
        "claim_boundary": (
            "This report classifies observed native/simulator differences; it does "
            "not assign a causal explanation or authorize performance claims."
        ),
    }
