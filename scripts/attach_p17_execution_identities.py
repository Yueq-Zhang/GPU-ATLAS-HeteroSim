#!/usr/bin/env python3
"""Attach independently sealed execution identities to a P17 catalog."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--execution-identity-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("operators"), list
    ):
        raise ValueError("P17 measurement catalog is invalid")
    identities = load_execution_identity_catalog(args.execution_identity_catalog)
    observed: set[str] = set()
    operators: list[object] = []
    for raw in payload["operators"]:
        if not isinstance(raw, Mapping):
            raise ValueError("P17 operator record must be an object")
        record = dict(raw)
        operator = str(record.get("operator_type", ""))
        if operator in identities:
            record["execution_identity"] = identities[operator]
            observed.add(operator)
        operators.append(record)
    missing = sorted(set(identities) - observed)
    if missing:
        raise ValueError(f"identity catalog contains unknown operators: {missing}")
    result = dict(payload)
    result["operators"] = operators
    result["execution_identity_evidence"] = {
        "catalog": str(args.execution_identity_catalog),
        "trace_observed_operator_count": len(identities),
        "native_observed_operator_count": sum(
            bool(item["native_measurement_observed"])
            for item in identities.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"P17 execution identities attached: {args.output} "
        f"({len(identities)} operators)"
    )


if __name__ == "__main__":
    main()
