#!/usr/bin/env bash
set -euo pipefail

PYTHON="${HETEROSIM_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"
BACKEND="${P17_NATIVE_VRAM_BACKEND:-configs/hetero/backends/gpu_accelsim_rtx3070.json}"
CAPABILITIES="${P17_CAPABILITIES:-configs/hetero/operator_capabilities/tinyllama_prefill_layer0_bs1_ctx16.json}"
QUALIFICATION_ROOT="${P17_NATIVE_VRAM_ROOT:-/opt/gpu-atlas/qualification/p17-native-vram}"
NATIVE_CATALOG="${P17_NATIVE_CATALOG:-validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json}"
SIMULATOR_CATALOG="${P17_SIMULATOR_CATALOG:-validation/p17/gpu_operator_pairing/simulator_native_vram.json}"
PAIRING_AUDIT="${P17_PAIRING_AUDIT:-validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json}"

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$QUALIFICATION_ROOT" "$(dirname "$SIMULATOR_CATALOG")"

mapfile -t OPERATOR_TRACES < <(
  "$PYTHON" - "$CAPABILITIES" <<'PY'
import json
import sys
from pathlib import Path

capability_path = Path(sys.argv[1]).resolve()
root = Path.cwd().resolve()
payload = json.loads(capability_path.read_text(encoding="utf-8"))
for capability in payload["operator_types"]:
    if capability.get("backend_kind") != "accel_sim":
        continue
    operator = capability["operator_type"]
    refs = capability.get("artifact_refs", [])
    if len(refs) != 1:
        raise SystemExit(f"{operator} does not have exactly one Artifact")
    artifact_path = (root / refs[0]).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    traces = [
        item for item in artifact["files"]
        if item.get("kind") == "accel_sim_trace_manifest"
    ]
    if len(traces) != 1:
        raise SystemExit(f"{operator} does not have exactly one Trace Manifest")
    trace_path = Path(traces[0]["path"])
    if not trace_path.is_absolute():
        trace_path = (artifact_path.parent / trace_path).resolve()
    if not trace_path.is_file():
        raise SystemExit(f"Trace Manifest is absent for {operator}: {trace_path}")
    print(f"{operator}\t{trace_path}")
PY
)

if [[ ${#OPERATOR_TRACES[@]} -ne 14 ]]; then
  echo "expected 14 GPU operators, found ${#OPERATOR_TRACES[@]}" >&2
  exit 3
fi

for entry in "${OPERATOR_TRACES[@]}"; do
  IFS=$'\t' read -r operator trace_manifest <<<"$entry"
  qualification="$QUALIFICATION_ROOT/${operator//_/-}-native-vram"
  echo "P17 native-VRAM qualification: $operator"
  "$PYTHON" -m frontend.hetero.cli qualify-gpu \
    --resume-completed-runs \
    --backend-config "$BACKEND" \
    --trace-manifest "$trace_manifest" \
    --output "$qualification"
  "$PYTHON" - "$qualification/qualification_record.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
cycles = record.get("comparison", {}).get("gpu_tot_sim_cycle")
instructions = record.get("comparison", {}).get("gpu_tot_sim_insn")
ownership = record.get("timing_ownership", {})
if (
    record.get("status") != "passed"
    or not isinstance(cycles, list)
    or len(cycles) != 2
    or cycles[0] != cycles[1]
    or not isinstance(instructions, list)
    or len(instructions) != 2
    or instructions[0] != instructions[1]
    or "external_memory_stats" in record.get("comparison", {})
    or ownership.get("gpu_local_dram") != "accel_sim"
    or ownership.get("external_ramulator2") is not False
    or ownership.get("duration_mode") != "total"
):
    raise SystemExit(f"invalid native-VRAM qualification: {path}")
PY
done

"$PYTHON" scripts/build_p17_gpu_simulator_catalog.py \
  --capabilities "$CAPABILITIES" \
  --qualification-root "$QUALIFICATION_ROOT" \
  --output "$SIMULATOR_CATALOG"
"$PYTHON" scripts/audit_p17_gpu_operator_pairing.py \
  --native "$NATIVE_CATALOG" \
  --simulator "$SIMULATOR_CATALOG" \
  --output "$PAIRING_AUDIT"

echo "P17 native-VRAM Accel-Sim qualification complete: $QUALIFICATION_ROOT"
