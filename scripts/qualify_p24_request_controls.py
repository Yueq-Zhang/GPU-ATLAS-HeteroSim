#!/usr/bin/env python3
"""Run the bounded single-layer P24 request-control qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from frontend.hetero.request_control_runtime import run_request_control_runtime
from frontend.hetero.schema import load_and_validate_config

CASES = {
    "eos_max_active_cancel": (
        "configs/hetero/experiments/"
        "p24_tinyllama_1layer_request_controls_bs3.json"
    ),
    "kv_capacity_release_reuse": (
        "configs/hetero/experiments/"
        "p24_tinyllama_1layer_kv_pressure_bs4.json"
    ),
}


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(config: Mapping[str, object]) -> dict[str, object]:
    return run_request_control_runtime(
        config["workload"]["requests"],  # type: ignore[index]
        config["scheduling"],  # type: ignore[arg-type]
        config["model"],  # type: ignore[arg-type]
        config["address"],  # type: ignore[arg-type]
    )


def _qualify(
    name: str,
    config_ref: str,
    config: Mapping[str, object],
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    memory = dict(first["memory"])  # type: ignore[arg-type]
    conservation = dict(first["conservation"])  # type: ignore[arg-type]
    reasons = dict(first["termination"]["reason_counts"])  # type: ignore[index]
    checks = {
        "double_run_identical": first == second,
        "single_layer_only": bool(first["single_layer_qualification"]),
        "all_requests_terminal": bool(conservation["all_requests_terminal"]),
        "admission_allocation_bijective": bool(
            conservation["admission_allocation_bijective"]
        ),
        "zero_in_flight": bool(conservation["zero_in_flight"]),
        "global_pa_active_ranges_disjoint": (
            int(memory["active_range_overlap_count"]) == 0
        ),
        "allocation_epochs_unique": bool(memory["allocation_epoch_unique"]),
        "zero_kv_bytes_after_retirement": bool(
            memory["zero_bytes_after_retirement"]
        ),
        "kv_peak_within_capacity": (
            int(memory["peak_bytes"]) <= int(memory["capacity_bytes"])
        ),
        "performance_claim_closed": first["performance_claim_allowed"] is False,
    }
    if name == "eos_max_active_cancel":
        requests = {
            str(item["request_id"]): dict(item)
            for item in first["requests"]  # type: ignore[index]
        }
        checks.update(
            {
                "eos_observed": reasons["eos"] == 1,
                "max_length_observed": reasons["max_length"] == 1,
                "active_cancel_observed": (
                    reasons["cancelled"] == 1
                    and requests["R-cancel-active"]["generated_length"] == 1
                    and requests["R-cancel-active"]["cancel_observed_time_fs"]
                    == 1000
                ),
            }
        )
    elif name == "kv_capacity_release_reuse":
        checks.update(
            {
                "all_termination_reasons_observed": reasons
                == {"completed": 1, "eos": 1, "max_length": 1, "cancelled": 1},
                "waiting_cancel_never_allocated": first["termination"][  # type: ignore[index]
                    "cancelled_before_admission"
                ]
                == ["R2-cancel-waiting"],
                "capacity_delayed_admission_observed": bool(
                    memory["capacity_delayed_request_ids"]
                ),
                "retired_global_pa_reused": (
                    int(memory["retired_range_reuse_count"]) >= 2
                ),
                "capacity_reached": (
                    int(memory["peak_bytes"]) == int(memory["capacity_bytes"])
                ),
            }
        )
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return {
        "config_ref": config_ref,
        "resolved_config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "checks": checks,
        "qualification_passed": all(checks.values()),
        "termination": first["termination"],
        "memory": memory,
        "conservation": conservation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("validation/p24"))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root

    records: dict[str, object] = {}
    for name, config_ref in CASES.items():
        config = load_and_validate_config(project_root / config_ref)
        leg1_path = output_root / name / "leg1" / "request_control_runtime.json"
        leg2_path = output_root / name / "leg2" / "request_control_runtime.json"
        if args.reuse_existing:
            first, second = _read(leg1_path), _read(leg2_path)
        else:
            first, second = _run(config), _run(config)
            _write(leg1_path, first)
            _write(leg2_path, second)
        records[name] = _qualify(name, config_ref, config, first, second)

    passed = all(
        bool(dict(record)["qualification_passed"]) for record in records.values()
    )
    summary = {
        "schema_version": "hetero-p24-request-control-qualification/v1",
        "validation_scope": "tinyllama_decode_single_layer_only",
        "qualification_passed": passed,
        "performance_claim_allowed": False,
        "cases": records,
        "boundary": (
            "P24 validates functional-cycle EOS, maximum output length, explicit "
            "barrier cancellation, KV capacity pressure, release and Global PA "
            "reuse for a single decoder layer. Exact batched traces and calibrated "
            "performance remain outside this qualification."
        ),
    }
    _write(output_root / "qualification_record.json", summary)
    print(output_root / "qualification_record.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
