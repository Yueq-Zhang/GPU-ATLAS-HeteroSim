#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <operator> <context> [output root] [manifest root]" >&2
  exit 2
fi

OPERATOR="$1"
CONTEXT="$2"
OUTPUT_ROOT="${3:-/opt/gpu-atlas/qualification/p15a}"
MANIFEST_DIR="${4:-configs/hetero/operator_artifacts/p15a}"
MODEL_ROOT="${TINYLLAMA_MODEL_ROOT:-/opt/hf-cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
PYTHON="${TINYLLAMA_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"
RUN_DIR="$OUTPUT_ROOT/tinyllama-prefill-bs1-ctx${CONTEXT}-${OPERATOR//_/-}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

case "$OPERATOR" in
  attention_norm|qkv_projection|rope|kv_append|causal_attention|output_projection|mlp_norm|gate_up_projection|silu_multiply|down_projection|final_norm|lm_head|sampling) ;;
  *)
    echo "unsupported shape-locked Prefill operator: $OPERATOR" >&2
    exit 2
    ;;
esac
if ! [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]]; then
  echo "context must be a positive integer" >&2
  exit 2
fi
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to mix a new capture with existing files: $RUN_DIR" >&2
  exit 3
fi

export ACTIVE_FROM_START=1
unset DYNAMIC_KERNEL_RANGE || true
bash scripts/capture_accel_sim_trace.sh \
  "$PYTHON" \
  "$RUN_DIR" \
  workloads/python/tinyllama_prefill_operator.py \
  --model "$MODEL_ROOT" \
  --operator "$OPERATOR" \
  --context "$CONTEXT" \
  --capture-allocator-history \
  --metadata-output "$RUN_DIR/operator_metadata.json"

mkdir -p "$MANIFEST_DIR"
BUILD_ARGS=(
  --metadata "$RUN_DIR/operator_metadata.json"
  --kernels-list "$RUN_DIR/traces/kernelslist.g"
  --output "$MANIFEST_DIR/tinyllama_prefill_bs1_ctx${CONTEXT}_${OPERATOR}_sm86.json"
)
if [[ -s "$RUN_DIR/traces/kernelslist.g" ]]; then
  BUILD_ARGS+=(
    --trace-manifest-output
    "$MANIFEST_DIR/tinyllama_prefill_bs1_ctx${CONTEXT}_${OPERATOR}_sm86_trace.json"
  )
fi
"${HETEROSIM_PYTHON:-.venv/bin/python}" \
  scripts/build_gpu_operator_artifact.py "${BUILD_ARGS[@]}"

echo "P15-A capture and manifest complete: $RUN_DIR"
