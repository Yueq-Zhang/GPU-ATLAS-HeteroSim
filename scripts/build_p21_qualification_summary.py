#!/usr/bin/env python3
"""Combine the one-layer and 22-layer P21 qualification records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "hetero-p21-decode-trace-timeline-qualification/v1"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} root must be an object")
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "passed"
        or payload.get("double_run_deterministic") is not True
        or payload.get("performance_eligible") is not False
    ):
        raise ValueError(f"{path} is not a passing, performance-ineligible P21 record")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-layer", type=Path, required=True)
    parser.add_argument("--twenty-two-layer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [_load(args.one_layer), _load(args.twenty_two_layer)]
    if [int(record["fixed_scope"]["layers"]) for record in records] != [1, 22]:
        raise ValueError("P21 summary requires one-layer and 22-layer records")
    cases = []
    for record, source in zip(records, (args.one_layer, args.twenty_two_layer)):
        leg = record["legs"][0]
        gpu_tasks = leg["gpu_instruction_request_cycle"]
        runtime_tasks = leg["runtime_memory"]
        cases.append(
            {
                "layers": record["fixed_scope"]["layers"],
                "task_count": leg["task_count"],
                "real_trace_task_count": len(gpu_tasks),
                "runtime_memory_task_count": len(runtime_tasks),
                "gpu_parent_requests": sum(item["parents"] for item in gpu_tasks),
                "gpu_child_requests": sum(item["children"] for item in gpu_tasks),
                "gpu_instructions": sum(item["instructions"] for item in gpu_tasks),
                "runtime_parent_requests": sum(
                    item["requests"] for item in runtime_tasks
                ),
                "makespan_fs_observed_unqualified": leg["makespan_fs"],
                "step_completion_fs_observed_unqualified": leg["step_completion_fs"],
                "qualification_record": source.as_posix(),
            }
        )
    summary = {
        "schema_version": "hetero-p21-decode-trace-qualification-summary/v1",
        "status": "passed",
        "functional_scope": records[0]["fixed_scope"],
        "qualified_invariants": records[0]["invariants"],
        "cases": cases,
        "claim_boundary": {
            "performance_claim_allowed": False,
            "reason": records[0]["claim_boundary"],
        },
        "next_replacement": (
            "Calibrate GPU core, cache, interconnect, memory timing and remaining "
            "runtime contracts against matched hardware before performance claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
