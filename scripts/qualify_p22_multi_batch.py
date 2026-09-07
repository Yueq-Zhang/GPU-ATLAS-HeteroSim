#!/usr/bin/env python3
"""Run and qualify the deterministic P22 functional multi-batch contracts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config

CASES = {
    "one_layer_static_bs2": (
        "configs/hetero/experiments/"
        "p22_tinyllama_decode4_1layer_static_bs2.json"
    ),
    "one_layer_static_ragged_padding_bs2": (
        "configs/hetero/experiments/"
        "p22_tinyllama_decode2_1layer_static_ragged_padding_bs2.json"
    ),
    "one_layer_static_ragged_split_bs2": (
        "configs/hetero/experiments/"
        "p22_tinyllama_decode2_1layer_static_ragged_split_bs2.json"
    ),
    "two_layer_mixed_continuous_bs4": (
        "configs/hetero/experiments/"
        "p22_tinyllama_2layer_mixed_continuous_bs4.json"
    ),
    "twenty_two_layer_continuous_bs4": (
        "configs/hetero/experiments/"
        "p22_tinyllama_decode4_22layer_continuous_bs4.json"
    ),
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_case(
    case_name: str,
    config: Mapping[str, object],
    first: Path,
    second: Path,
    project_root: Path,
) -> dict[str, object]:
    first_runtime = _load(first / "multi_batch_runtime.json")
    second_runtime = _load(second / "multi_batch_runtime.json")
    first_plan = _load(first / "batch_plan.json")
    second_plan = _load(second / "batch_plan.json")
    first_lifecycle = _load(first / "memory_lifecycle.json")
    second_lifecycle = _load(second / "memory_lifecycle.json")
    conservation = dict(first_runtime["conservation"])
    memory = dict(first_runtime["memory_isolation"])
    scheduling = dict(config["scheduling"])  # type: ignore[arg-type]
    resource_ids = {
        str(item["resource_id"])
        for item in first_runtime["resource_intervals"]  # type: ignore[index]
    }
    layer_ids = {
        int(item["layer_id"])
        for item in first_plan["device_subbatches"]  # type: ignore[index]
        if item["layer_id"] is not None
    }
    admissions = {
        str(item["request_id"]): int(item["time_fs"])
        for item in first_runtime["request_events"]  # type: ignore[index]
        if item["event"] == "ADMIT"
    }
    retirements = sorted(
        int(item["time_fs"])
        for item in first_runtime["request_events"]  # type: ignore[index]
        if item["event"] == "RETIRE"
    )
    checks = {
        "double_run_runtime_identical": first_runtime == second_runtime,
        "double_run_batch_plan_identical": first_plan == second_plan,
        "double_run_memory_lifecycle_identical": (
            first_lifecycle == second_lifecycle
        ),
        "all_requests_admitted_once": (
            conservation["input_requests"] == conservation["admitted_requests"]
        ),
        "all_requests_retired_once": (
            conservation["input_requests"] == conservation["retired_requests"]
        ),
        "all_requests_finished": bool(conservation["all_requests_finished"]),
        "zero_in_flight": bool(conservation["zero_in_flight"]),
        "member_fanout_bijective": bool(
            conservation["member_fanout_is_bijective_per_subbatch"]
        ),
        "global_pa_active_ranges_disjoint": (
            int(memory["active_range_overlap_count"]) == 0
        ),
        "global_pa_allocation_epochs_unique": bool(
            memory["allocation_epoch_unique"]
        ),
        "kv_zero_bytes_after_retirement": bool(
            memory["zero_bytes_after_retirement"]
        ),
        "active_sequence_limit_obeyed": (
            int(first_runtime["metrics"]["max_active_requests"])  # type: ignore[index]
            <= int(scheduling["max_num_sequences"])
        ),
        "performance_claim_closed": (
            first_runtime["performance_claim_allowed"] is False
        ),
        "request_cycle_composed": (
            first_runtime["cycle_mode"] == "request_cycle_composed"
        ),
    }
    if case_name == "one_layer_static_bs2":
        checks.update(
            {
                "static_request_count_is_two": len(admissions) == 2,
                "homogeneous_batch_size_two": (
                    int(first_runtime["metrics"]["max_device_subbatch_size"])  # type: ignore[index]
                    == 2
                ),
                "one_layer_materialized": layer_ids == {0},
                "gpu_only_zero_atlas": resource_ids <= {"gpu0"},
            }
        )
    elif case_name == "one_layer_static_ragged_padding_bs2":
        checks.update(
            {
                "gpu_only_zero_atlas": resource_ids <= {"gpu0"},
                "padding_dense_selected": first_plan["batch_policy"]
                == "padding_dense",
                "padding_cost_recorded": int(
                    first_plan["padded_attention_tokens"]
                )
                > int(first_plan["effective_attention_tokens"]),
                "one_layer_materialized": layer_ids == {0},
            }
        )
    elif case_name == "one_layer_static_ragged_split_bs2":
        checks.update(
            {
                "gpu_only_zero_atlas": resource_ids <= {"gpu0"},
                "ragged_split_selected": first_plan["batch_policy"]
                == "ragged_split",
                "incompatible_shapes_split": int(
                    first_runtime["metrics"]["max_device_subbatch_size"]  # type: ignore[index]
                )
                == 1,
                "one_layer_materialized": layer_ids == {0},
            }
        )
    elif case_name == "two_layer_mixed_continuous_bs4":
        phases = {
            str(item["phase"])
            for item in first_plan["device_subbatches"]  # type: ignore[index]
        }
        checks.update(
            {
                "continuous_request_count_is_four": len(admissions) == 4,
                "mixed_prefill_decode_present": phases == {"prefill", "decode"},
                "chunked_prefill_present": len(first_plan["epochs"]) > 4,  # type: ignore[arg-type]
                "gpu_and_atlas_subbatches_present": resource_ids
                == {"gpu0", "atlas0.compute"},
                "two_layers_materialized": layer_ids == {0, 1},
            }
        )
    else:
        checks.update(
            {
                "gpu_only_zero_atlas": resource_ids <= {"gpu0"},
                "continuous_request_count_is_four": len(admissions) == 4,
                "retire_then_readmit_observed": (
                    len(retirements) == 4
                    and any(time >= retirements[0] for time in admissions.values())
                ),
                "twenty_two_layers_materialized": layer_ids == set(range(22)),
            }
        )
    return {
        "config": CASES[case_name],
        "leg1": str(first.relative_to(project_root)),
        "leg2": str(second.relative_to(project_root)),
        "checks": checks,
        "qualification_passed": all(checks.values()),
        "metrics": first_runtime["metrics"],
        "qualification_boundary": first_runtime["qualification_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("validation/p22")
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="rebuild only the qualification summary from existing two-leg runs",
    )
    parser.add_argument(
        "--single-layer-only",
        action="store_true",
        help="run only the three one-layer validation cases",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    records: dict[str, object] = {}
    active_cases = {
        name: ref
        for name, ref in CASES.items()
        if not args.single_layer_only or name.startswith("one_layer_")
    }
    for case_name, config_ref in active_cases.items():
        config = load_and_validate_config(project_root / config_ref)
        if args.reuse_existing:
            experiment_name = str(config["experiment"]["name"])  # type: ignore[index]
            first_candidates = list(
                (output_root / case_name / "leg1" / experiment_name).iterdir()
            )
            second_candidates = list(
                (output_root / case_name / "leg2" / experiment_name).iterdir()
            )
            if len(first_candidates) != 1 or len(second_candidates) != 1:
                raise RuntimeError(
                    f"{case_name} must have exactly one run per qualification leg"
                )
            first, second = first_candidates[0], second_candidates[0]
        else:
            first = execute_run(
                config, project_root, output_root / case_name / "leg1"
            )
            second = execute_run(
                config, project_root, output_root / case_name / "leg2"
            )
        records[case_name] = _check_case(
            case_name, config, first, second, project_root
        )
    passed = all(
        bool(dict(record)["qualification_passed"])
        for record in records.values()
    )
    summary = {
        "schema_version": "hetero-p22-multi-batch-qualification/v1",
        "validation_scope": (
            "single_layer_only" if args.single_layer_only else "all_cases"
        ),
        "qualification_passed": passed,
        "performance_claim_allowed": False,
        "cases": records,
        "boundary": (
            "P22 validates deterministic functional-cycle batching, request/KV "
            "causality, shared-resource serialization and Global PA isolation. "
            "Exact fused batched-kernel traces and calibrated performance remain open."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    record_path = output_root / "qualification_record.json"
    record_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(record_path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
