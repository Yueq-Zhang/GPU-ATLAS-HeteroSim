#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${HETEROSIM_PYTHON:-/home/yueqi/anaconda3/envs/dl/bin/python}"
SOURCE_ROOT="${P23_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p23_sm89_decode_bs2}"
COUPLED_ROOT="${P23_COUPLED_ROOT:-configs/hetero/operator_artifacts/p23_sm89_decode_bs2_coupled}"
QUALIFICATION_ROOT="${P23_QUALIFICATION_ROOT:-/opt/gpu-atlas/qualification/p23-bs2-decode-range-rebase}"
BACKEND="${P23_BACKEND:-configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch_range_rebase.json}"
OPERATORS=(
  token_embedding attention_norm qkv_projection rope causal_attention
  output_projection residual_add mlp_norm gate_up_projection silu_multiply
  down_projection final_norm lm_head sampling
)

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$QUALIFICATION_ROOT" "$COUPLED_ROOT"

for operator in "${OPERATORS[@]}"; do
  stem="tinyllama_decode_bs2_ctx16_kv17_${operator}_sm89"
  source_artifact="$SOURCE_ROOT/${stem}.json"
  trace_manifest="$SOURCE_ROOT/${stem}_trace.json"
  qualification="$QUALIFICATION_ROOT/${operator//_/-}"
  coupled_artifact="$COUPLED_ROOT/${stem}_shared_hbdram_range_rebase.json"
  for path in "$source_artifact" "$trace_manifest" "$BACKEND"; do
    [[ -s "$path" ]] || { echo "required P23 input is absent: $path" >&2; exit 4; }
  done

  echo "P23 BS2 Range-Rebase qualification: operator=$operator"
  "$PYTHON" -m frontend.hetero.cli qualify-gpu \
    --resume-completed-runs \
    --backend-config "$BACKEND" \
    --trace-manifest "$trace_manifest" \
    --output "$qualification"
  "$PYTHON" scripts/validate_p23_qualification_record.py \
    "$qualification/qualification_record.json"
  "$PYTHON" scripts/build_coupled_gpu_operator_artifact.py \
    --source-artifact "$source_artifact" \
    --backend-config "$BACKEND" \
    --qualification-record "$qualification/qualification_record.json" \
    --address-mode range_rebase \
    --output "$coupled_artifact"
done

"$PYTHON" scripts/build_p23_bs2_decode_ready_catalog.py \
  --capture-catalog validation/p23/capture_catalog.json \
  --coupled-root "$COUPLED_ROOT" \
  --qualification-root "$QUALIFICATION_ROOT" \
  --output validation/p23/ready_catalog.json

echo "P23 BS2 one-layer operator qualification complete"
