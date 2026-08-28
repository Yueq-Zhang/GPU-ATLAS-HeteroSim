#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
COUPLED_ROOT="${ACCEL_SIM_RAMULATOR2_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT-ramulator2}"
SMOKE="$COUPLED_ROOT/ramulator2_bridge/full_chip_scheduler_smoke"
MEMORY_CONFIG="$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbdram_edge_16ch_shared.yaml"
CHIP_CONFIG="$PROJECT_ROOT/configs/hetero/atlas/tinyllama_qproj_edge_16core_chip.yaml"
ARTIFACT_ROOT="$PROJECT_ROOT/configs/hetero/atlas/tinyllama11b_qproj_decode_bs1_ctx1024"
OUTPUT_ROOT="${1:-/opt/gpu-atlas/qualification/full-chip-scheduler-memory-path}"

[[ -x "$SMOKE" ]] || {
  echo "missing full-chip scheduler smoke binary: $SMOKE" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT"
LOG="$OUTPUT_ROOT/full_chip_scheduler.log"

env \
  HETEROSIM_GPU_CLOCK_HZ=1200000000 \
  HETEROSIM_LINK_CLOCK_HZ=400000000 \
  HETEROSIM_GATEWAY_CLOCK_HZ=400000000 \
  HETEROSIM_DRAM_CLOCK_HZ=400000000 \
  HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS=409600000000 \
  HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS=409600000000 \
  HETEROSIM_GATEWAY_ISSUE_WIDTH=64 \
  "$SMOKE" \
    "$MEMORY_CONFIG" \
    "$CHIP_CONFIG" \
    "$ARTIFACT_ROOT/operator_description.yaml" \
    "$ARTIFACT_ROOT/data_placement.yaml" > "$LOG"

python3 - "$LOG" "$OUTPUT_ROOT/qualification_record.json" <<'PY'
import json
import sys
from pathlib import Path


def parse(line: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in line.split()[1:]:
        key, value = field.split("=", 1)
        result[key] = int(value)
    return result


log = Path(sys.argv[1])
lines = log.read_text(encoding="utf-8").splitlines()
summary = parse(next(
    line for line in lines
    if line.startswith("heterosim_full_chip_scheduler_smoke ")
))
bridge = [
    parse(line) for line in lines
    if line.startswith("heterosim_ramulator2_summary ")
]
errors: list[str] = []
expected = {
    "atlas_memory_bytes": 8_925_184,
    "gpu_parents": 4_096,
    "outstanding": 0,
    "contention": 1,
    "conservation": 1,
    "instances": 1,
}
for field, value in expected.items():
    if summary.get(field) != value:
        errors.append(f"{field}={summary.get(field)} expected {value}")
if summary.get("atlas_parents", 0) <= 0:
    errors.append("full atlasim.Chip emitted no shared-DRAM requests")
if summary.get("concurrent_atlas_finish", 0) <= summary.get(
    "atlas_only_finish", 0
):
    errors.append("synthetic GPU traffic did not delay the live ATLAS Chip")
if len(bridge) != 2:
    errors.append(f"bridge_summary_count={len(bridge)} expected 2")
elif bridge[0].get("logical_bytes") != summary.get("atlas_memory_bytes"):
    errors.append(
        "ATLAS-only bridge bytes do not match full-chip transaction bytes"
    )

record = {
    "schema_version": "heterosim-full-chip-scheduler-qualification/v1",
    "status": "failed" if errors else "passed",
    "artifact": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "operator": "layer0.q_proj",
        "phase": "decode",
        "batch": 1,
        "context": 1024,
        "shape": {"M": 1, "K": 2048, "N": 2048},
        "dtype": "FP16",
        "atlas_cores": 16,
    },
    "coverage": {
        "atlas_full_chip_scheduler": not errors,
        "atlas_native_component_inputs": not errors,
        "external_nonblocking_dram_service": not errors,
        "single_shared_ramulator2": not errors,
        "initiator_specific_completion_queues": not errors,
        "per_core_global_pa_projection": not errors,
        "accel_sim_compute_backend": False,
        "gpu_traffic_source": "deterministic synthetic parents",
    },
    "configuration": {
        "gpu_clock_Hz": 1_200_000_000,
        "atlas_clock_Hz": 1_000_000_000,
        "dram_clock_Hz": 400_000_000,
        "channels": 16,
        "transaction_bytes": 64,
        "internal_peak_Bps": 409_600_000_000,
        "qualification_external_link_Bps": 409_600_000_000,
        "core_global_pa_region_bytes": 1_048_576,
    },
    "byte_accounting": {
        "external_transaction_bytes": 8_925_184,
        "legacy_native_component_reported_bytes": 8_916_992,
        "difference_bytes": 8_192,
        "explanation": (
            "The live service records every aligned 64-byte read/write "
            "parent emitted by HBFrontend-equivalent lowering. The prior "
            "native component statistic accounts the eight 32-byte output "
            "stores per core as logical write bytes, so it is not used as "
            "the transaction-byte conservation oracle."
        ),
    },
    "results": summary,
    "bridge_summaries": bridge,
    "invariants": {
        "no_native_atlas_ramulator_instance": not errors,
        "parent_completion_conservation": not errors,
        "both_initiators_observe_one_memory_owner": not errors,
        "zero_inflight_at_exit": not errors,
    },
    "claim_boundary": (
        "This qualifies a complete atlasim.Chip scheduler against the shared "
        "memory service with deterministic synthetic GPU contention. It is "
        "not yet concurrent execution of the Accel-Sim compute backend."
    ),
    "errors": errors,
}
output = Path(sys.argv[2])
output.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(record, sort_keys=True))
if errors:
    raise SystemExit(3)
PY
