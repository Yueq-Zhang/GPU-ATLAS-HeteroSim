#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
COUPLED_ROOT="${ACCEL_SIM_RAMULATOR2_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT-ramulator2}"
SMOKE="$COUPLED_ROOT/ramulator2_bridge/ramulator_bridge_smoke"
OUTPUT_ROOT="${1:-/opt/gpu-atlas/qualification/gpu-only-layered-memory-path}"
EXTERNAL_CONFIG="$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbdram_edge_16ch_gpu_only.yaml"
INTERNAL_CONFIG="$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbdram_edge_1ch_gpu_only.yaml"

[[ -x "$SMOKE" ]] || {
  echo "missing bridge smoke binary: $SMOKE" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT"

run_case() {
  local config="$1"
  local bandwidth="$2"
  local issue_width="$3"
  local log="$4"
  env \
    HETEROSIM_GPU_CLOCK_HZ=1200000000 \
    HETEROSIM_LINK_CLOCK_HZ=400000000 \
    HETEROSIM_GATEWAY_CLOCK_HZ=400000000 \
    HETEROSIM_DRAM_CLOCK_HZ=400000000 \
    HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS="$bandwidth" \
    HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS="$bandwidth" \
    HETEROSIM_GATEWAY_ISSUE_WIDTH="$issue_width" \
    "$SMOKE" "$config" > "$log"
}

EXTERNAL_LOG="$OUTPUT_ROOT/external_link_bottleneck.log"
INTERNAL_LOG="$OUTPUT_ROOT/internal_dram_bottleneck.log"
run_case "$EXTERNAL_CONFIG" 12800000000 4 "$EXTERNAL_LOG"
run_case "$INTERNAL_CONFIG" 409600000000 16 "$INTERNAL_LOG"

python3 - "$EXTERNAL_LOG" "$INTERNAL_LOG" "$OUTPUT_ROOT/qualification_record.json" <<'PY'
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[str, int]:
    line = next(
        item for item in path.read_text(encoding="utf-8").splitlines()
        if item.startswith("heterosim_ramulator2_smoke ")
    )
    result: dict[str, int] = {}
    for field in line.split()[1:]:
        key, value = field.split("=", 1)
        result[key] = int(value)
    return result


external = load(Path(sys.argv[1]))
internal = load(Path(sys.argv[2]))
expected = {
    "sent": 72,
    "returned": 72,
    "reads": 64,
    "writes": 8,
    "completed": 72,
    "durable_completed": 72,
    "children_sent": 144,
    "children_completed": 144,
    "logical_bytes": 9216,
    "internal_bytes": 9216,
    "request_payload_bytes": 1024,
    "response_payload_bytes": 8192,
    "request_wire_bytes": 3328,
    "response_wire_bytes": 10496,
    "outstanding": 0,
    "instances": 1,
}
errors: list[str] = []
for name, stats in (("external", external), ("internal", internal)):
    for field, value in expected.items():
        if stats.get(field) != value:
            errors.append(f"{name}.{field}={stats.get(field)} expected {value}")
    if stats.get("gpu_cycles") != stats.get("link_cycles", -1) * 3:
        errors.append(f"{name} multi-clock phase ratio failed")
    if stats.get("gateway_cycles") != stats.get("link_cycles"):
        errors.append(f"{name} gateway/link clock ratio failed")
    if stats.get("cycles") != stats.get("link_cycles"):
        errors.append(f"{name} DRAM/link clock ratio failed")
if external.get("cycles", 0) <= internal.get("cycles", 0):
    errors.append("external-link-limited case must take more cycles than internal case")

record = {
    "schema_version": "heterosim-gpu-only-layered-qualification/v1",
    "status": "failed" if errors else "passed",
    "cases": {
        "external_link_bottleneck": {
            "external_payload_bandwidth_Bps": 12_800_000_000,
            "internal_peak_bandwidth_Bps": 409_600_000_000,
            "stats": external,
        },
        "internal_dram_bottleneck": {
            "external_payload_bandwidth_Bps": 409_600_000_000,
            "internal_peak_bandwidth_Bps": 25_600_000_000,
            "stats": internal,
        },
    },
    "invariants": {
        "parent_child_conservation": not errors,
        "payload_wire_byte_conservation": not errors,
        "no_parent_before_all_children": not errors,
        "durable_write_completion": not errors,
        "zero_inflight_at_exit": not errors,
        "single_ramulator2_owner": not errors,
    },
    "errors": errors,
}
output = Path(sys.argv[3])
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
if errors:
    raise SystemExit(3)
PY
