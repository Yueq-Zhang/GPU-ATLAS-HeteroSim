#!/usr/bin/env python3
"""Replace P17 simple-operator native timings with sealed same-Binary evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.gpu_execution_identity import (  # noqa: E402
    load_execution_identity_catalog,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--execution-identity-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = _json(args.catalog.resolve())
    identities = load_execution_identity_catalog(
        args.execution_identity_catalog.resolve()
    )
    operators = payload.get("operators")
    if not isinstance(operators, list):
        raise ValueError("native catalog operators must be an array")
    records = {
        str(item.get("operator_type")): item
        for item in operators
        if isinstance(item, dict)
    }
    updated: list[str] = []
    for operator, identity in identities.items():
        if not identity.get("native_measurement_observed"):
            continue
        record = records.get(operator)
        provenance = identity.get("provenance")
        if not isinstance(record, dict) or not isinstance(provenance, Mapping):
            raise ValueError(f"native catalog record is absent for {operator}")
        measurement_path = _resolve(provenance.get("native_measurement"))
        expected_measurement_sha256 = str(
            provenance.get("native_measurement_sha256", "")
        )
        if _sha256(measurement_path) != expected_measurement_sha256:
            raise ValueError(f"native measurement hash mismatch for {operator}")
        measurement = _json(measurement_path)
        summary = measurement.get("measurement")
        protocol = measurement.get("protocol")
        software = measurement.get("software")
        if (
            measurement.get("operator_type") != operator
            or not isinstance(summary, Mapping)
            or not isinstance(protocol, Mapping)
            or not isinstance(software, Mapping)
        ):
            raise ValueError(f"native measurement is invalid for {operator}")
        latency = {}
        for target, source in (
            ("min", "min_fs"),
            ("p10", "p10_fs"),
            ("median", "median_fs"),
            ("p90", "p90_fs"),
            ("max", "max_fs"),
            ("mean", "mean_fs"),
        ):
            value = summary.get(source)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"invalid {operator} measurement field {source}")
            latency[target] = round(value)
        artifact = _resolve(record.get("operator_artifact"))
        record.update(
            {
                "measurement_backend": "sealed_same_binary_cuda_event",
                "measurement_software": dict(software),
                "measurement_source_sha256": expected_measurement_sha256,
                "operator_artifact_sha256": _sha256(artifact),
                "operator_latency_fs": latency,
                "repetitions": int(protocol["measured_iterations"]),
                "trace_binary_identity": "verified_same_binary",
                "execution_identity": identity,
            }
        )
        updated.append(operator)
    if not updated:
        raise ValueError("no native-observed execution identities were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"P17 native catalog updated with same-Binary evidence: {args.output} "
        f"({', '.join(sorted(updated))})"
    )


if __name__ == "__main__":
    main()
