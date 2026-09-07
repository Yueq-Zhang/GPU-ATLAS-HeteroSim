#!/usr/bin/env python3
"""Qualify the single-layer P25 QoS and liveness micro-contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontend.hetero.qos_watchdog import (
    InterconnectActivationError,
    ProgressWatchdogError,
    activate_interconnect_adapter,
    simulate_qos_micro_stress,
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _watchdog_check(
    requests: list[dict[str, object]],
    policy: dict[str, object],
    fault: str,
) -> dict[str, object]:
    try:
        simulate_qos_micro_stress(requests, policy, fault_injection=fault)
    except ProgressWatchdogError as error:
        return {
            "fault": fault,
            "detected": True,
            "classification": error.kind,
            "detection_cycle": error.cycle,
            "window_cycles": error.window_cycles,
            "passed": error.kind == fault,
        }
    return {"fault": fault, "detected": False, "passed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/hetero/p25/p25_qos_watchdog_micro.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/p25/qualification_record.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile_path = args.profile if args.profile.is_absolute() else root / args.profile
    output = args.output if args.output.is_absolute() else root / args.output
    profile = _load(profile_path)
    requests = [dict(item) for item in profile["requests"]]  # type: ignore[index]
    policy = dict(profile["policy"])  # type: ignore[arg-type]

    first = simulate_qos_micro_stress(requests, policy)
    second = simulate_qos_micro_stress(requests, policy)
    metrics = dict(first["metrics"])
    conservation = dict(first["conservation"])
    prefix = dict(metrics["service_units_by_class_in_fairness_window"])
    normal_checks = {
        "double_run_identical": first == second,
        "single_layer_scope": dict(profile["scope"])["layers"] == [0],
        "all_requests_finished": bool(conservation["all_requests_finished"]),
        "work_conserved": bool(conservation["zero_work_in_flight"]),
        "gpu_and_atlas_serviced": set(metrics["service_units_by_resource"])
        == {"gpu0", "atlas0.compute"},
        "starvation_bound_obeyed": int(metrics["max_observed_wait_cycles"])
        <= int(policy["max_starvation_cycles"]),
        "latency_class_advantaged_in_prefix": int(prefix["latency"])
        > int(prefix["throughput"]),
        "same_class_peer_fairness": float(
            metrics["same_group_prefix_jain_fairness"]["normal-gpu-peers"]
        )
        >= 0.95,
        "healthy_run_forward_progress": bool(
            dict(first["watchdog"])["forward_progress_proven_for_run"]
        ),
        "performance_claim_closed": first["performance_claim_allowed"] is False,
    }
    watchdog_checks = {
        fault: _watchdog_check(requests, policy, fault)
        for fault in ("deadlock", "livelock")
    }

    booksim_config_path = (
        root / "configs/hetero/p25/p25_booksim2_fail_closed.json"
    )
    booksim_config = _load(booksim_config_path)
    booksim_gate: dict[str, object]
    try:
        activate_interconnect_adapter(booksim_config, root)
    except InterconnectActivationError as error:
        reason = str(error).replace(str(root) + "/", "")
        reason = reason.replace(str(root) + "\\", "")
        booksim_gate = {
            "requested": True,
            "activated": False,
            "failed_closed": True,
            "reason": reason,
            "passed": True,
        }
    else:
        booksim_gate = {
            "requested": True,
            "activated": True,
            "failed_closed": False,
            "passed": False,
        }

    passed = (
        all(normal_checks.values())
        and all(bool(record["passed"]) for record in watchdog_checks.values())
        and bool(booksim_gate["passed"])
    )
    record = {
        "schema_version": "hetero-p25-qos-liveness-qualification/v1",
        "qualification_passed": passed,
        "scope": profile["scope"],
        "performance_claim_allowed": False,
        "normal_run_checks": normal_checks,
        "watchdog_fault_checks": watchdog_checks,
        "booksim2_activation_gate": booksim_gate,
        "metrics": metrics,
        "runtime_digest": _load_digest(first),
        "boundary": (
            "P25 proves deterministic functional QoS arbitration, bounded "
            "starvation and deadlock/livelock termination for a one-layer "
            "GPU/ATLAS micro-stress. BookSim2 is not activated and scheduler "
            "micro-cycles are not performance cycles."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime_path = output.parent / "qos_runtime.json"
    runtime_path.write_text(
        json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0 if passed else 1


def _load_digest(value: object) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
