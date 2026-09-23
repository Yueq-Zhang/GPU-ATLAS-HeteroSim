#!/usr/bin/env python3
"""Double-run qualification of the active BookSim2 + Ramulator2 QoS path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from frontend.hetero.booksim2_adapter import probe_booksim2_adapter
from frontend.hetero.booksim2_memory_runtime import run_booksim2_ramulator2_qos
from frontend.hetero.qos_watchdog import activate_interconnect_adapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hetero/p25/p27_booksim2_ramulator2_qos.json"
OUTPUT = ROOT / "validation/p27/booksim2_ramulator2_qos"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    activation = activate_interconnect_adapter(
        {
            "backend": "booksim2",
            "adapter_abi": "gpu-atlas-booksim2-adapter/v1",
            "adapter_library": config["adapter_library"],
            "network_config": config["network_config"],
        },
        ROOT,
        booksim2_probe=probe_booksim2_adapter,
    )
    requests = config.pop("requests")
    first = run_booksim2_ramulator2_qos(ROOT, requests, config)
    second = run_booksim2_ramulator2_qos(ROOT, requests, config)
    deterministic = _canonical(first) == _canonical(second)
    gates = {
        "booksim2_active": bool(activation["booksim2_active"]),
        "deterministic_double_run": deterministic,
        "single_ramulator2_owner": first["timing_owner"]
        == "global_cycle_booksim2_plus_single_ramulator2",
        "packet_conservation": first["conservation"]["packets"]
        == first["conservation"]["packet_completions"],
        "flit_conservation": first["conservation"]["flits"]
        == first["conservation"]["flit_completions"],
        "credit_conservation": first["conservation"]["outstanding_credits"] == 0,
        "parent_conservation": first["conservation"]["parent_requests"]
        == first["conservation"]["durable_parent_completions"],
        "zero_in_flight": bool(first["conservation"]["zero_in_flight"]),
        "gpu_and_atlas_present": {item["initiator"] for item in first["requests"]}
        == {"gpu0", "atlas0.compute"},
        "qos_contention_observed": first["qos"]["max_observed_packet_wait_cycles"]
        > 0,
    }
    record = {
        "schema_version": "hetero-p27-booksim2-ramulator2-qualification/v1",
        "status": "passed" if all(gates.values()) else "failed",
        "performance_claim_allowed": False,
        "config": str(CONFIG.relative_to(ROOT)),
        "activation": activation,
        "gates": gates,
        "leg1_sha256": hashlib.sha256(_canonical(first)).hexdigest(),
        "leg2_sha256": hashlib.sha256(_canonical(second)).hexdigest(),
        "qualification_boundary": first["qualification_boundary"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, payload in (("leg1.json", first), ("leg2.json", second), ("qualification_record.json", record)):
        (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
