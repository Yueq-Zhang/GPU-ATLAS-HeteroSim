#!/usr/bin/env python3
"""Combine the one-layer and 22-layer P20 qualification records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} root must be an object")
    if (
        payload.get("schema_version")
        != "hetero-p20-decode-loop-qualification/v1"
        or payload.get("status") != "passed"
        or payload.get("double_run_deterministic") is not True
        or payload.get("claim_boundary", {}).get("performance_claim_allowed")
        is not False
    ):
        raise ValueError(f"{path} is not a passing unqualified P20 record")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-layer", type=Path, required=True)
    parser.add_argument("--twenty-two-layer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [_load(args.one_layer), _load(args.twenty_two_layer)]
    if [record["expected_layers"] for record in records] != [1, 22]:
        raise ValueError("P20 summary requires one-layer and 22-layer records")
    cases = []
    for record, source in zip(records, (args.one_layer, args.twenty_two_layer)):
        leg = record["legs"][0]
        cases.append(
            {
                "layers": record["expected_layers"],
                "task_count": leg["task_count"],
                "accepted_parent_requests": leg["accepted_parent_requests"],
                "gpu_cycles_observed_unqualified": leg["gpu_cycles"],
                "ramulator2_cycles": leg["ramulator2_cycles"],
                "makespan_fs_observed_unqualified": leg["makespan_fs"],
                "ttft_fs_observed_unqualified": leg[
                    "ttft_fs_observed_unqualified"
                ],
                "itl_fs_observed_unqualified": leg[
                    "itl_fs_observed_unqualified"
                ],
                "qualification_record": source.as_posix(),
            }
        )
    summary = {
        "schema_version": "hetero-p20-decode-loop-qualification-summary/v1",
        "status": "passed",
        "functional_scope": records[0]["shape_contract"],
        "qualified_invariants": records[0]["invariants"],
        "cases": cases,
        "claim_boundary": records[0]["claim_boundary"],
        "next_replacement": (
            "Replace exact-shape tiled GPU cycle contracts with qualified "
            "Accel-Sim instruction traces before performance calibration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
