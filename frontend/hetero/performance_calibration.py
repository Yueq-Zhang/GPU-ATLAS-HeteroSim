"""Fail-closed performance-calibration contracts and audit helpers.

Request-cycle qualification proves causality and request conservation.  It does
not prove that any clock, queue, link, runtime or DRAM parameter represents a
target system.  This module keeps that second gate explicit and machine
readable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class PerformanceCalibrationError(ValueError):
    """Raised when a calibration record is structurally invalid."""


_COMPONENT_STATUS = {
    "missing",
    "specified_only",
    "measured_unvalidated",
    "validated",
    "not_applicable",
}
_EVIDENCE_CLASSES = {
    "hardware_measurement",
    "trusted_reference_simulator",
    "official_specification",
    "implementation_configuration",
}


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PerformanceCalibrationError(f"{path} must be an object")
    return value


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerformanceCalibrationError(f"{path} must be a non-empty string")
    return value


def _positive_number(value: object, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) <= 0.0
    ):
        raise PerformanceCalibrationError(f"{path} must be positive")
    return float(value)


def _unsigned_number(value: object, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) < 0.0
    ):
        raise PerformanceCalibrationError(f"{path} must be unsigned")
    return float(value)


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise PerformanceCalibrationError(f"{path} must be an array of strings")
    if len(set(value)) != len(value):
        raise PerformanceCalibrationError(f"{path} contains duplicates")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    source_id: str
    evidence_class: str
    locator: str
    description: str
    artifact_sha256: str | None
    measurement_scope: str | None


@dataclass(frozen=True, slots=True)
class CalibrationReferencePoint:
    reference_id: str
    source_id: str
    metric: str
    measured_value: float
    simulated_value: float
    unit: str
    repetitions: int
    statistic: str
    max_relative_error: float
    max_absolute_error: float

    @property
    def absolute_error(self) -> float:
        return abs(self.simulated_value - self.measured_value)

    @property
    def relative_error(self) -> float | None:
        if self.measured_value == 0.0:
            return None
        return self.absolute_error / abs(self.measured_value)

    @property
    def allowed_absolute_error(self) -> float:
        relative = abs(self.measured_value) * self.max_relative_error
        return max(self.max_absolute_error, relative)

    @property
    def passed(self) -> bool:
        return self.absolute_error <= self.allowed_absolute_error

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "source_id": self.source_id,
            "metric": self.metric,
            "measured_value": self.measured_value,
            "simulated_value": self.simulated_value,
            "unit": self.unit,
            "repetitions": self.repetitions,
            "statistic": self.statistic,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "allowed_absolute_error": self.allowed_absolute_error,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class CalibrationComponent:
    component_id: str
    required: bool
    status: str
    timing_owner: str
    parameter_bindings: tuple[Mapping[str, object], ...]
    sources: tuple[CalibrationSource, ...]
    reference_points: tuple[CalibrationReferencePoint, ...]
    applicable_shape_keys: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class PerformanceCalibration:
    source_path: Path
    calibration_id: str
    target_system: Mapping[str, object]
    scope: Mapping[str, object]
    required_shape_key: str
    required_components: tuple[str, ...]
    required_metrics: Mapping[str, tuple[str, ...]]
    allowed_validation_evidence: tuple[str, ...]
    minimum_reference_points: int
    configuration_sources: tuple[Mapping[str, object], ...]
    components: Mapping[str, CalibrationComponent]

    @classmethod
    def load(cls, path: str | Path) -> "PerformanceCalibration":
        source_path = Path(path).resolve()
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PerformanceCalibrationError(
                f"failed to load {source_path}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise PerformanceCalibrationError("calibration root must be an object")
        return cls.from_payload(payload, source_path)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object], source_path: str | Path = "<inline>"
    ) -> "PerformanceCalibration":
        if payload.get("schema_version") != "hetero-performance-calibration/v1":
            raise PerformanceCalibrationError("invalid calibration schema_version")
        calibration_id = _nonempty(payload.get("calibration_id"), "calibration_id")
        target = _mapping(payload.get("target_system"), "target_system")
        scope = _mapping(payload.get("scope"), "scope")
        policy = _mapping(payload.get("qualification_policy"), "qualification_policy")
        required_shape_key = _nonempty(
            policy.get("required_shape_key"),
            "qualification_policy.required_shape_key",
        )
        required_components = _string_tuple(
            policy.get("required_components"),
            "qualification_policy.required_components",
        )
        allowed_evidence = _string_tuple(
            policy.get("allowed_validation_evidence"),
            "qualification_policy.allowed_validation_evidence",
        )
        if any(item not in _EVIDENCE_CLASSES for item in allowed_evidence):
            raise PerformanceCalibrationError("unknown allowed validation evidence")
        minimum_points_raw = policy.get("minimum_reference_points_per_component")
        if (
            not isinstance(minimum_points_raw, int)
            or isinstance(minimum_points_raw, bool)
            or minimum_points_raw <= 0
        ):
            raise PerformanceCalibrationError(
                "minimum_reference_points_per_component must be positive"
            )
        required_metrics_raw = _mapping(
            policy.get("required_metrics"), "qualification_policy.required_metrics"
        )
        required_metrics = {
            str(component_id): _string_tuple(
                metrics,
                f"qualification_policy.required_metrics.{component_id}",
            )
            for component_id, metrics in required_metrics_raw.items()
        }
        if set(required_metrics) != set(required_components):
            raise PerformanceCalibrationError(
                "required_metrics must cover exactly the required components"
            )

        configuration_sources_raw = payload.get("configuration_sources")
        if not isinstance(configuration_sources_raw, list):
            raise PerformanceCalibrationError("configuration_sources must be an array")
        configuration_sources: list[Mapping[str, object]] = []
        configuration_ids: set[str] = set()
        for index, raw in enumerate(configuration_sources_raw):
            item = _mapping(raw, f"configuration_sources[{index}]")
            source_id = _nonempty(
                item.get("source_id"), f"configuration_sources[{index}].source_id"
            )
            if source_id in configuration_ids:
                raise PerformanceCalibrationError("duplicate configuration source_id")
            configuration_ids.add(source_id)
            _nonempty(item.get("path"), f"configuration_sources[{index}].path")
            digest = _nonempty(
                item.get("sha256"), f"configuration_sources[{index}].sha256"
            )
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise PerformanceCalibrationError(
                    "configuration sha256 must be lowercase"
                )
            configuration_sources.append(dict(item))

        components_raw = _mapping(payload.get("components"), "components")
        if set(required_components) - set(components_raw):
            raise PerformanceCalibrationError(
                "required calibration components are missing"
            )
        components: dict[str, CalibrationComponent] = {}
        for component_id, raw in components_raw.items():
            component_path = f"components.{component_id}"
            item = _mapping(raw, component_path)
            required = item.get("required")
            if not isinstance(required, bool):
                raise PerformanceCalibrationError(
                    f"{component_path}.required must be boolean"
                )
            if (component_id in required_components) != required:
                raise PerformanceCalibrationError(
                    f"{component_path}.required disagrees with qualification policy"
                )
            status = _nonempty(item.get("status"), f"{component_path}.status")
            if status not in _COMPONENT_STATUS:
                raise PerformanceCalibrationError(f"invalid {component_path}.status")
            if required and status == "not_applicable":
                raise PerformanceCalibrationError(
                    f"required component {component_id} cannot be not_applicable"
                )
            timing_owner = _nonempty(
                item.get("timing_owner"), f"{component_path}.timing_owner"
            )
            bindings_raw = item.get("parameter_bindings")
            if not isinstance(bindings_raw, list):
                raise PerformanceCalibrationError(
                    f"{component_path}.parameter_bindings must be an array"
                )
            bindings: list[Mapping[str, object]] = []
            for binding_index, raw_binding in enumerate(bindings_raw):
                binding = _mapping(
                    raw_binding,
                    f"{component_path}.parameter_bindings[{binding_index}]",
                )
                _nonempty(binding.get("name"), "parameter binding name")
                _nonempty(binding.get("unit"), "parameter binding unit")
                source_id = _nonempty(
                    binding.get("configuration_source_id"),
                    "parameter binding configuration_source_id",
                )
                if source_id not in configuration_ids:
                    raise PerformanceCalibrationError(
                        f"unknown configuration source {source_id}"
                    )
                if "configured_value" not in binding:
                    raise PerformanceCalibrationError(
                        "parameter binding configured_value is required"
                    )
                bindings.append(dict(binding))

            sources_raw = item.get("sources")
            if not isinstance(sources_raw, list):
                raise PerformanceCalibrationError(
                    f"{component_path}.sources must be an array"
                )
            sources: list[CalibrationSource] = []
            source_ids: set[str] = set()
            for source_index, raw_source in enumerate(sources_raw):
                source = _mapping(
                    raw_source, f"{component_path}.sources[{source_index}]"
                )
                source_id = _nonempty(source.get("source_id"), "source_id")
                if source_id in source_ids:
                    raise PerformanceCalibrationError(
                        f"duplicate source_id in {component_id}"
                    )
                source_ids.add(source_id)
                evidence_class = _nonempty(
                    source.get("evidence_class"), "source evidence_class"
                )
                if evidence_class not in _EVIDENCE_CLASSES:
                    raise PerformanceCalibrationError(
                        f"unknown evidence class {evidence_class}"
                    )
                sources.append(
                    CalibrationSource(
                        source_id=source_id,
                        evidence_class=evidence_class,
                        locator=_nonempty(source.get("locator"), "source locator"),
                        description=_nonempty(
                            source.get("description"), "source description"
                        ),
                        artifact_sha256=(
                            _nonempty(source.get("artifact_sha256"), "artifact_sha256")
                            if source.get("artifact_sha256") is not None
                            else None
                        ),
                        measurement_scope=(
                            _nonempty(
                                source.get("measurement_scope"), "measurement_scope"
                            )
                            if source.get("measurement_scope") is not None
                            else None
                        ),
                    )
                )
                digest = sources[-1].artifact_sha256
                if digest is not None and (
                    len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise PerformanceCalibrationError(
                        "source artifact_sha256 must be lowercase sha256"
                    )

            points_raw = item.get("reference_points")
            if not isinstance(points_raw, list):
                raise PerformanceCalibrationError(
                    f"{component_path}.reference_points must be an array"
                )
            points: list[CalibrationReferencePoint] = []
            point_ids: set[str] = set()
            for point_index, raw_point in enumerate(points_raw):
                point = _mapping(
                    raw_point, f"{component_path}.reference_points[{point_index}]"
                )
                point_id = _nonempty(point.get("reference_id"), "reference_id")
                if point_id in point_ids:
                    raise PerformanceCalibrationError(
                        f"duplicate reference_id in {component_id}"
                    )
                point_ids.add(point_id)
                repetitions = point.get("repetitions")
                if (
                    not isinstance(repetitions, int)
                    or isinstance(repetitions, bool)
                    or repetitions <= 0
                ):
                    raise PerformanceCalibrationError("repetitions must be positive")
                points.append(
                    CalibrationReferencePoint(
                        reference_id=point_id,
                        source_id=_nonempty(point.get("source_id"), "source_id"),
                        metric=_nonempty(point.get("metric"), "reference metric"),
                        measured_value=_unsigned_number(
                            point.get("measured_value"), "measured_value"
                        ),
                        simulated_value=_unsigned_number(
                            point.get("simulated_value"), "simulated_value"
                        ),
                        unit=_nonempty(point.get("unit"), "reference unit"),
                        repetitions=repetitions,
                        statistic=_nonempty(point.get("statistic"), "statistic"),
                        max_relative_error=_unsigned_number(
                            point.get("max_relative_error"), "max_relative_error"
                        ),
                        max_absolute_error=_unsigned_number(
                            point.get("max_absolute_error"), "max_absolute_error"
                        ),
                    )
                )
                if points[-1].source_id not in source_ids:
                    raise PerformanceCalibrationError(
                        f"unknown reference source {points[-1].source_id} in {component_id}"
                    )
            components[str(component_id)] = CalibrationComponent(
                component_id=str(component_id),
                required=required,
                status=status,
                timing_owner=timing_owner,
                parameter_bindings=tuple(bindings),
                sources=tuple(sources),
                reference_points=tuple(points),
                applicable_shape_keys=_string_tuple(
                    item.get("applicable_shape_keys"),
                    f"{component_path}.applicable_shape_keys",
                ),
                notes=str(item.get("notes", "")),
            )
        return cls(
            source_path=Path(source_path),
            calibration_id=calibration_id,
            target_system=dict(target),
            scope=dict(scope),
            required_shape_key=required_shape_key,
            required_components=required_components,
            required_metrics=required_metrics,
            allowed_validation_evidence=allowed_evidence,
            minimum_reference_points=minimum_points_raw,
            configuration_sources=tuple(configuration_sources),
            components=components,
        )

    def audit(self, project_root: str | Path | None = None) -> dict[str, object]:
        blockers: list[str] = []
        configuration_audit: list[dict[str, object]] = []
        root = Path(project_root).resolve() if project_root is not None else None
        for raw in self.configuration_sources:
            expected = str(raw["sha256"])
            path = Path(str(raw["path"]))
            if not path.is_absolute() and root is not None:
                path = root / path
            exists = path.is_file() if root is not None or path.is_absolute() else None
            actual: str | None = None
            matched: bool | None = None
            if exists:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                matched = actual == expected
                if not matched:
                    blockers.append(f"configuration:{raw['source_id']}:sha256_mismatch")
            elif exists is False:
                blockers.append(f"configuration:{raw['source_id']}:missing")
            configuration_audit.append(
                {
                    "source_id": raw["source_id"],
                    "path": str(raw["path"]),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "exists": exists,
                    "matched": matched,
                }
            )

        component_audit: dict[str, object] = {}
        for component_id in sorted(self.components):
            component = self.components[component_id]
            component_blockers: list[str] = []
            source_artifacts: list[dict[str, object]] = []
            source_artifact_matches: dict[str, bool] = {}
            for source in component.sources:
                if source.artifact_sha256 is None:
                    continue
                artifact_path = Path(source.locator)
                if not artifact_path.is_absolute() and root is not None:
                    artifact_path = root / artifact_path
                exists = (
                    artifact_path.is_file()
                    if root is not None or artifact_path.is_absolute()
                    else None
                )
                actual: str | None = None
                matched: bool | None = None
                if exists:
                    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    matched = actual == source.artifact_sha256
                    if not matched:
                        component_blockers.append(
                            f"source_artifact_sha256_mismatch={source.source_id}"
                        )
                elif exists is False:
                    component_blockers.append(
                        f"source_artifact_missing={source.source_id}"
                    )
                source_artifact_matches[source.source_id] = matched is True
                source_artifacts.append(
                    {
                        "source_id": source.source_id,
                        "locator": source.locator,
                        "measurement_scope": source.measurement_scope,
                        "expected_sha256": source.artifact_sha256,
                        "actual_sha256": actual,
                        "exists": exists,
                        "matched": matched,
                    }
                )
            validating_sources = [
                source
                for source in component.sources
                if source.evidence_class in self.allowed_validation_evidence
                and source.artifact_sha256 is not None
                and source_artifact_matches.get(source.source_id) is True
            ]
            validating_source_ids = {source.source_id for source in validating_sources}
            invalid_reference_sources = sorted(
                {
                    point.source_id
                    for point in component.reference_points
                    if point.source_id not in validating_source_ids
                }
            )
            metrics = {point.metric for point in component.reference_points}
            missing_metrics = sorted(
                set(self.required_metrics.get(component_id, ())) - metrics
            )
            if component.required:
                if component.status != "validated":
                    component_blockers.append(f"status={component.status}")
                if not validating_sources:
                    component_blockers.append("no_validating_source")
                if len(component.reference_points) < self.minimum_reference_points:
                    component_blockers.append("insufficient_reference_points")
                if invalid_reference_sources:
                    component_blockers.append(
                        "unverified_reference_sources="
                        + ",".join(invalid_reference_sources)
                    )
                if missing_metrics:
                    component_blockers.append(
                        "missing_metrics=" + ",".join(missing_metrics)
                    )
                if self.required_shape_key not in component.applicable_shape_keys:
                    component_blockers.append("shape_not_covered")
                if not component.parameter_bindings:
                    component_blockers.append("no_parameter_bindings")
                if any(not point.passed for point in component.reference_points):
                    component_blockers.append("reference_error_exceeds_tolerance")
            for blocker in component_blockers:
                blockers.append(f"component:{component_id}:{blocker}")
            component_audit[component_id] = {
                "required": component.required,
                "status": component.status,
                "timing_owner": component.timing_owner,
                "parameter_binding_count": len(component.parameter_bindings),
                "validating_source_count": len(validating_sources),
                "validating_sources": [
                    source.source_id for source in validating_sources
                ],
                "source_artifacts": source_artifacts,
                "required_metrics": list(self.required_metrics.get(component_id, ())),
                "observed_metrics": sorted(metrics),
                "missing_metrics": missing_metrics,
                "applicable_shape_keys": list(component.applicable_shape_keys),
                "reference_points": [
                    point.to_audit_dict() for point in component.reference_points
                ],
                "blockers": component_blockers,
                "qualified": not component_blockers,
            }
        return {
            "schema_version": "hetero-performance-calibration-audit/v1",
            "calibration_id": self.calibration_id,
            "required_shape_key": self.required_shape_key,
            "configuration_sources": configuration_audit,
            "components": component_audit,
            "required_component_count": len(self.required_components),
            "qualified_component_count": sum(
                not value["blockers"]
                for value in component_audit.values()
                if value["required"]
            ),
            "blockers": blockers,
            "performance_claim_allowed": not blockers,
        }


def evaluate_performance_gate(
    calibration: PerformanceCalibration,
    tasks: list[Mapping[str, object]],
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Combine component calibration with per-task performance eligibility."""

    audit = calibration.audit(project_root)
    blockers = list(audit["blockers"])
    included: list[str] = []
    excluded: list[str] = []
    ineligible: list[str] = []
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id", f"task[{index}]"))
        fidelity = _mapping(task.get("fidelity"), f"{task_id}.fidelity")
        if fidelity.get("device_performance_included", True) is False:
            excluded.append(task_id)
            continue
        included.append(task_id)
        if fidelity.get("performance_eligible") is not True:
            ineligible.append(task_id)
    if not included:
        blockers.append("tasks:no_device_performance_tasks")
    if ineligible:
        blockers.append("tasks:performance_ineligible=" + ",".join(sorted(ineligible)))
    return {
        **audit,
        "schema_version": "hetero-performance-gate/v1",
        "component_calibration_allowed": audit["performance_claim_allowed"],
        "task_gate": {
            "included_task_count": len(included),
            "excluded_control_task_count": len(excluded),
            "included_task_ids": included,
            "excluded_control_task_ids": excluded,
            "ineligible_task_ids": sorted(ineligible),
            "all_included_tasks_performance_eligible": bool(included)
            and not ineligible,
        },
        "blockers": blockers,
        "performance_claim_allowed": not blockers,
    }
