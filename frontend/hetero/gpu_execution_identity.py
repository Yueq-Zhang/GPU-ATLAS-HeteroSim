"""Fail-closed GPU execution-program and kernel-launch identity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


EXECUTION_IDENTITY_SCHEMA = "hetero-gpu-execution-identity/v1"
EXECUTION_IDENTITY_CATALOG_SCHEMA = "hetero-gpu-execution-identity-catalog/v1"
IDENTITY_DIGEST_FIELDS = (
    "executable_sha256",
    "launch_contract_sha256",
    "kernel_sequence_sha256",
)


class GPUExecutionIdentityError(ValueError):
    """Raised when execution identity evidence is incomplete or ambiguous."""


def canonical_sha256(value: object) -> str:
    """Hash one JSON value using a stable UTF-8 representation."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GPUExecutionIdentityError(f"{path} must be an object")
    return value


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GPUExecutionIdentityError(f"{path} must be a non-empty string")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _nonempty(value, path)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise GPUExecutionIdentityError(f"{path} must be lowercase SHA-256")
    return digest


def validate_execution_identity(
    value: object,
    path: str = "execution_identity",
) -> dict[str, object]:
    """Validate and normalize one same-executable identity record."""

    identity = _mapping(value, path)
    if identity.get("schema_version") != EXECUTION_IDENTITY_SCHEMA:
        raise GPUExecutionIdentityError(f"{path}.schema_version is invalid")
    normalized: dict[str, object] = {
        "schema_version": EXECUTION_IDENTITY_SCHEMA,
    }
    for field in IDENTITY_DIGEST_FIELDS:
        normalized[field] = _sha256(identity.get(field), f"{path}.{field}")
    target_sm = identity.get("target_sm")
    launches = identity.get("kernel_launch_count")
    if not isinstance(target_sm, int) or isinstance(target_sm, bool) or target_sm <= 0:
        raise GPUExecutionIdentityError(f"{path}.target_sm must be positive")
    if not isinstance(launches, int) or isinstance(launches, bool) or launches <= 0:
        raise GPUExecutionIdentityError(
            f"{path}.kernel_launch_count must be positive"
        )
    normalized["target_sm"] = target_sm
    normalized["kernel_launch_count"] = launches
    for field in ("native_measurement_observed", "trace_capture_observed"):
        observed = identity.get(field)
        if not isinstance(observed, bool):
            raise GPUExecutionIdentityError(f"{path}.{field} must be a boolean")
        normalized[field] = observed
    provenance = _mapping(identity.get("provenance"), f"{path}.provenance")
    normalized["provenance"] = dict(provenance)
    return normalized


def load_execution_identity_catalog(
    path: str | Path,
) -> dict[str, dict[str, object]]:
    """Load an operator-indexed catalog of independently sealed identities."""

    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GPUExecutionIdentityError(f"cannot load {source}: {error}") from error
    root = _mapping(payload, str(source))
    if root.get("schema_version") != EXECUTION_IDENTITY_CATALOG_SCHEMA:
        raise GPUExecutionIdentityError("execution identity catalog schema is invalid")
    raw_operators = root.get("operators")
    if not isinstance(raw_operators, list):
        raise GPUExecutionIdentityError("execution identity operators must be an array")
    operators: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_operators):
        record = _mapping(raw, f"operators[{index}]")
        operator = _nonempty(record.get("operator_type"), "operator_type")
        if operator in operators:
            raise GPUExecutionIdentityError(f"duplicate execution identity {operator}")
        operators[operator] = validate_execution_identity(
            record.get("execution_identity"),
            f"operators[{index}].execution_identity",
        )
    count = root.get("operator_count")
    if count != len(operators):
        raise GPUExecutionIdentityError("execution identity operator_count is invalid")
    return operators


def trace_kernel_sequence(
    kernels_list: str | Path,
) -> tuple[str, list[dict[str, str]]]:
    """Hash stable launch headers while excluding runtime virtual addresses."""

    kernels = Path(kernels_list).resolve()
    required = (
        "kernel name",
        "grid dim",
        "block dim",
        "shmem",
        "nregs",
        "binary version",
    )
    descriptors: list[dict[str, str]] = []
    for raw in kernels.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry:
            continue
        trace = (kernels.parent / entry).resolve()
        values: dict[str, str] = {}
        with trace.open("r", encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream):
                if index >= 128:
                    break
                stripped = line.strip()
                if not stripped.startswith("-") or "=" not in stripped:
                    continue
                key, raw_value = stripped[1:].split("=", maxsplit=1)
                key = key.strip()
                if key in required:
                    values[key] = raw_value.strip()
                if all(key in values for key in required):
                    break
        missing = [key for key in required if key not in values]
        if missing:
            raise GPUExecutionIdentityError(
                f"trace launch header is incomplete for {trace}: {missing}"
            )
        descriptors.append({key: values[key] for key in required})
    if not descriptors:
        raise GPUExecutionIdentityError("kernel sequence must contain one launch")
    return canonical_sha256(descriptors), descriptors


def execution_identities_match(
    native: Mapping[str, object],
    simulator: Mapping[str, object],
) -> bool:
    """Compare the immutable fields used by the same-Binary gate."""

    return all(native.get(field) == simulator.get(field) for field in (
        *IDENTITY_DIGEST_FIELDS,
        "target_sm",
        "kernel_launch_count",
    ))
