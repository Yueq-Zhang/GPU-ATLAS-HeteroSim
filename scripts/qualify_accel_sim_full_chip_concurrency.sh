#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HETEROSIM_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
BACKEND="$PROJECT_ROOT/configs/hetero/backends/gpu_accelsim_rtx3070_full_atlas_chip_shared_hbdram_edge_16ch.json"
TRACE="$PROJECT_ROOT/configs/hetero/traces/local_rtx3070_tinyllama11b_qproj_decode_v2.json"
OUTPUT="${1:-/opt/gpu-atlas/qualification/accel-sim-v2/rtx3070-tinyllama-qproj-full-atlas-chip-shared-memory}"

[[ -x "$PYTHON" ]] || {
  echo "missing Python environment: $PYTHON" >&2
  exit 2
}

"$PYTHON" -m frontend.hetero.cli qualify-gpu \
  --backend-config "$BACKEND" \
  --trace-manifest "$TRACE" \
  --output "$OUTPUT"

"$PYTHON" - "$OUTPUT/qualification_record.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
if record["status"] != "passed":
    raise SystemExit(f"qualification failed: {path}")
required = {
    "cycle_coupled_request_response",
    "full_atlas_chip_shared_memory_concurrency",
    "single_ramulator2_multi_initiator_conservation",
}
if not required <= set(record["qualified_scopes"]):
    raise SystemExit(f"qualification coverage is incomplete: {path}")
for evidence in record["overlap_evidence"]:
    if not evidence["both_initiators_active"]:
        raise SystemExit("GPU and ATLAS did not both submit requests")
    if not evidence["atlas_finished_before_gpu_run_end"]:
        raise SystemExit("ATLAS execution did not overlap the Accel-Sim run")
print(json.dumps(record, sort_keys=True))
PY
