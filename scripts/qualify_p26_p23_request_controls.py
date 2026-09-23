#!/usr/bin/env python3
"""Qualify P24 request controls against the sealed real P23 timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.hetero.p23_request_controls import (  # noqa: E402
    run_kv_allocator_pressure_probe,
    run_p23_request_control_timeline,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hetero/experiments/p26_p23_request_control_stream.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/p26/p23_request_controls"),
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.get("schema_version") != "hetero-p23-p24-request-control-experiment/v1":
        raise ValueError("invalid P26 request-control experiment schema")
    qualification = project_root / str(config["qualification_ref"])
    first = run_p23_request_control_timeline(
        project_root, config["requests"], config["allocator"], qualification
    )
    second = run_p23_request_control_timeline(
        project_root, config["requests"], config["allocator"], qualification
    )
    equal = _canonical(first) == _canonical(second)
    pressure = run_kv_allocator_pressure_probe(
        epochs=4096,
        batch_size=2,
        per_request_bytes=int(config["allocator"]["per_member_kv_bytes"]),
        capacity_bytes=int(config["allocator"]["capacity_bytes"]),
        alignment_bytes=int(config["allocator"]["alignment_bytes"]),
        global_pa_base=int(config["allocator"]["global_pa_base"]),
    )
    gates = dict(first["gates"])
    required = {
        "cancellation_only_at_token_barrier",
        "release_after_all_requests_durable",
        "deterministic_reuse_observed",
        "no_active_overlap",
        "no_leak",
    }
    passed = (
        equal
        and all(gates.get(key) is True for key in required)
        and pressure["zero_in_flight"] is True
        and pressure["overlap_count"] == 0
        and pressure["leaked_allocations"] == 0
    )
    if not passed:
        raise RuntimeError("P23/P24 request-control qualification failed")

    output = args.output.resolve()
    _write(output / "leg1.json", first)
    _write(output / "leg2.json", second)
    record = {
        "schema_version": "hetero-p23-p24-request-control-qualification/v1",
        "status": "passed",
        "double_run_signature_equal": equal,
        "signature_sha256": hashlib.sha256(_canonical(first).encode("utf-8")).hexdigest(),
        "scope": {
            "input_requests": first["conservation"]["input_requests"],
            "admitted_requests": first["conservation"]["admitted_requests"],
            "exact_p23_epochs": first["conservation"]["exact_p23_epochs"],
            "exact_batch_size": first["exact_identity"]["batch_size"],
            "kv_length": first["exact_identity"]["final_kv_length"],
        },
        "gates": gates,
        "memory": {
            "capacity_bytes": first["memory"]["capacity_bytes"],
            "peak_bytes": first["memory"]["peak_bytes"],
            "retired_range_reuse_count": first["memory"]["retired_range_reuse_count"],
        },
        "long_capacity_pressure_probe": pressure,
        "conservation": first["conservation"],
        "performance_claim_allowed": False,
        "qualification_boundary": first["qualification_boundary"],
    }
    _write(output / "qualification_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
