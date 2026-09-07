#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE="${P21_QUALIFY_MODE:-smoke}"
PYTHON="${HETEROSIM_PYTHON:-/home/yueqi/anaconda3/envs/dl/bin/python}"
SOURCE_ROOT="${P21_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p21_sm89_decode}"
COUPLED_ROOT="${P21_COUPLED_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p21_sm89_decode_coupled}"
QUALIFICATION_ROOT="${P21_QUALIFICATION_ROOT:-/opt/gpu-atlas/qualification/p21-sm89-decode-range-rebase}"
BACKEND="${P21_BACKEND:-configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch_range_rebase.json}"
OPERATORS=(
  token_embedding attention_norm qkv_projection rope causal_attention
  output_projection residual_add mlp_norm gate_up_projection silu_multiply
  down_projection final_norm lm_head sampling
)
KV_LENGTHS=(17 18 19 20)

case "$MODE" in smoke|all) ;; *) echo "P21_QUALIFY_MODE must be smoke or all" >&2; exit 2 ;; esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$QUALIFICATION_ROOT" "$COUPLED_ROOT"

if [[ "$MODE" == "smoke" ]]; then
  OPERATORS=(attention_norm)
  KV_LENGTHS=(17)
fi

for kv_length in "${KV_LENGTHS[@]}"; do
  for operator in "${OPERATORS[@]}"; do
    stem="tinyllama_decode_bs1_ctx16_kv${kv_length}_${operator}_sm89"
    source_artifact="$SOURCE_ROOT/${stem}.json"
    trace_manifest="$SOURCE_ROOT/${stem}_trace.json"
    qualification="$QUALIFICATION_ROOT/kv${kv_length}-${operator//_/-}"
    coupled_artifact="$COUPLED_ROOT/${stem}_shared_hbdram_range_rebase.json"
    for path in "$source_artifact" "$trace_manifest" "$BACKEND"; do
      [[ -s "$path" ]] || { echo "required P21 qualification input is absent: $path" >&2; exit 4; }
    done

    echo "P21 remote-captured SM86 functional qualification: KV=$kv_length operator=$operator"
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
record = json.loads(path.read_text())
comparison = record.get("comparison", {})
cycles = comparison.get("gpu_tot_sim_cycle")
instructions = comparison.get("gpu_tot_sim_insn")
memory = comparison.get("external_memory_stats")
ownership = record.get("timing_ownership", {})
if (
    record.get("status") != "passed"
    or not isinstance(cycles, list) or len(cycles) != 2 or cycles[0] != cycles[1]
    or not isinstance(instructions, list) or len(instructions) != 2
    or instructions[0] != instructions[1]
    or not isinstance(memory, list) or len(memory) != 2 or memory[0] != memory[1]
    or ownership.get("duration_mode") != "coupled"
    or ownership.get("external_ramulator2") != "shared3d.ramulator2"
    or ownership.get("gpu_local_dram") is not None
):
    raise SystemExit(f"invalid P21 deterministic qualification: {path}")
stats = memory[0]
required_equal = (
    ("instances", 1), ("outstanding", 0), ("address_unmapped", 0),
    ("atlas_parents", 0), ("atlas_completed", 0),
)
if any(stats.get(name) != expected for name, expected in required_equal):
    raise SystemExit(f"invalid P21 ownership or address counters: {path}")
if (
    stats.get("gpu_parents", 0) <= 0
    or stats.get("address_translated", 0) <= 0
    or stats.get("gpu_parents") != stats.get("gpu_completed")
    or stats.get("gpu_parents") != stats.get("completed")
    or stats.get("children_sent") != stats.get("children_completed")
    or stats.get("completed") != stats.get("durable_completed")
):
    raise SystemExit(f"invalid P21 request conservation: {path}")
PY
    "$PYTHON" scripts/build_coupled_gpu_operator_artifact.py \
      --source-artifact "$source_artifact" \
      --backend-config "$BACKEND" \
      --qualification-record "$qualification/qualification_record.json" \
      --address-mode range_rebase \
      --output "$coupled_artifact"
  done
done

if [[ "$MODE" == "all" ]]; then
  "$PYTHON" scripts/build_p21_decode_ready_catalog.py \
    --capture-catalog validation/p21/remote_sm89/capture_catalog.json \
    --coupled-root "$COUPLED_ROOT" \
    --qualification-root "$QUALIFICATION_ROOT" \
    --artifact-catalog "$COUPLED_ROOT/tinyllama_decode4_bs1_ctx16_p21_catalog.json" \
    --qualification-map "$COUPLED_ROOT/tinyllama_decode4_bs1_ctx16_p21_qualification_map.json" \
    --ready-catalog validation/p21/remote_sm89/ready_catalog.json
fi

echo "P21 remote-captured SM86 functional qualification complete: mode=$MODE"
