#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
COUPLED_ROOT="${ACCEL_SIM_RAMULATOR2_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT-ramulator2}"
SMOKE="$COUPLED_ROOT/ramulator2_bridge/dual_initiator_smoke"
CONFIG="$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbdram_edge_1ch_shared.yaml"
OUTPUT_ROOT="${1:-/opt/gpu-atlas/qualification/dual-initiator-memory-path}"

[[ -x "$SMOKE" ]] || {
  echo "missing dual-initiator smoke binary: $SMOKE" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT"
LOG="$OUTPUT_ROOT/dual_initiator.log"

env \
  HETEROSIM_GPU_CLOCK_HZ=1200000000 \
  HETEROSIM_LINK_CLOCK_HZ=400000000 \
  HETEROSIM_GATEWAY_CLOCK_HZ=400000000 \
  HETEROSIM_DRAM_CLOCK_HZ=400000000 \
  HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS=409600000000 \
  HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS=409600000000 \
  HETEROSIM_GATEWAY_ISSUE_WIDTH=16 \
  "$SMOKE" "$CONFIG" > "$LOG"

python3 - "$LOG" "$OUTPUT_ROOT/qualification_record.json" <<'PY'
import json
import sys
from pathlib import Path


def parse_summary(line: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for field in line.split()[1:]:
        key, value = field.split("=", 1)
        result[key] = int(value)
    return result


log = Path(sys.argv[1])
lines = log.read_text(encoding="utf-8").splitlines()
golden = parse_summary(next(
    line for line in lines if line.startswith("heterosim_dual_initiator_smoke ")
))
summaries = [
    parse_summary(line)
    for line in lines
    if line.startswith("heterosim_ramulator2_summary ")
]
errors: list[str] = []
expected = {
    "gpu_parents": 72,
    "atlas_parents": 80,
    "gpu_children": 144,
    "atlas_children": 80,
    "logical_bytes": 14336,
    "internal_bytes": 14336,
    "outstanding": 0,
    "contention": 1,
    "instances": 1,
}
for field, value in expected.items():
    if golden.get(field) != value:
        errors.append(f"{field}={golden.get(field)} expected {value}")
if len(summaries) != 3:
    errors.append(f"summary_count={len(summaries)} expected 3")
if golden.get("concurrent_cycles", 0) <= max(
    golden.get("gpu_only_cycles", 0), golden.get("atlas_only_cycles", 0)
):
    errors.append("concurrent makespan did not expose shared-DRAM contention")
if golden.get("concurrent_gpu_last", 0) <= golden.get("gpu_only_last", 0):
    errors.append("GPU did not observe contention latency")
if golden.get("concurrent_atlas_last", 0) <= golden.get("atlas_only_last", 0):
    errors.append("ATLAS did not observe contention latency")

record = {
    "schema_version": "heterosim-dual-initiator-qualification/v1",
    "status": "failed" if errors else "passed",
    "configuration": {
        "dram": "HBDRAM_2Gb_512pin/HBDRAM_400Mbps",
        "channels": 1,
        "transaction_bytes": 64,
        "address_mapper": "OneLevelInterleave",
        "gpu_external_link_Bps": 409_600_000_000,
        "gpu_clock_Hz": 1_200_000_000,
        "link_gateway_dram_clock_Hz": 400_000_000,
    },
    "cases": {
        "gpu_only": {
            "cycles": golden["gpu_only_cycles"],
            "completion_gpu_cycle": golden["gpu_only_last"],
        },
        "atlas_only": {
            "cycles": golden["atlas_only_cycles"],
            "completion_gpu_cycle": golden["atlas_only_last"],
        },
        "concurrent": golden,
    },
    "invariants": {
        "native_atlas_component_input_contract": not errors,
        "gpu_uses_external_port": not errors,
        "atlas_uses_internal_hybrid_bond_port": not errors,
        "single_ramulator2_owner": not errors,
        "parent_child_byte_conservation": not errors,
        "durable_completion": not errors,
        "both_initiators_observe_contention": not errors,
        "zero_inflight_at_exit": not errors,
    },
    "errors": errors,
}
output = Path(sys.argv[2])
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
if errors:
    raise SystemExit(3)
PY
