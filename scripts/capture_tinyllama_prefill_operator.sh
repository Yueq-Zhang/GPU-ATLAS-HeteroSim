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
  token_embedding|attention_norm|qkv_projection|rope|kv_append|causal_attention|output_projection|residual_add|mlp_norm|gate_up_projection|silu_multiply|down_projection|final_norm|lm_head|sampling) ;;
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

WORKLOAD_ARGS=(
  --operator "$OPERATOR"
  --context "$CONTEXT"
  --metadata-output "$RUN_DIR/operator_metadata.json"
)
CAPTURE_ALLOCATOR_HISTORY="${HETEROSIM_CAPTURE_ALLOCATOR_HISTORY:-auto}"
if [[ "$CAPTURE_ALLOCATOR_HISTORY" == "auto" ]]; then
  case "$OPERATOR" in
    token_embedding|residual_add) CAPTURE_ALLOCATOR_HISTORY="0" ;;
    *) CAPTURE_ALLOCATOR_HISTORY="1" ;;
  esac
fi
if [[ "$CAPTURE_ALLOCATOR_HISTORY" == "1" ]]; then
  WORKLOAD_ARGS+=(--capture-allocator-history)
elif [[ "$CAPTURE_ALLOCATOR_HISTORY" != "0" ]]; then
  echo "HETEROSIM_CAPTURE_ALLOCATOR_HISTORY must be auto, 0 or 1" >&2
  exit 2
fi
if [[ "$OPERATOR" == "token_embedding" || "$OPERATOR" == "residual_add" ]]; then
  SIMPLE_BINARY="${HETEROSIM_SIMPLE_OPS_BINARY:-/opt/gpu-atlas/build/tinyllama_simple_ops}"
  bash scripts/build_tinyllama_simple_ops.sh "$SIMPLE_BINARY"
  export ACTIVE_FROM_START=1
  CAPTURE_APPLICATION="$SIMPLE_BINARY"
  CAPTURE_ARGS=("${WORKLOAD_ARGS[@]}")
elif [[ "${HETEROSIM_CAPTURE_RANGE:-process}" == "driver_profiler" ]]; then
  export ACTIVE_FROM_START=0
  WORKLOAD_ARGS=(--model "$MODEL_ROOT" "${WORKLOAD_ARGS[@]}" --driver-profiler --warmup "${HETEROSIM_CAPTURE_WARMUP:-1}")
  CAPTURE_APPLICATION="$PYTHON"
  CAPTURE_ARGS=(workloads/python/tinyllama_prefill_operator.py "${WORKLOAD_ARGS[@]}")
else
  export ACTIVE_FROM_START=1
  WORKLOAD_ARGS=(--model "$MODEL_ROOT" "${WORKLOAD_ARGS[@]}")
  CAPTURE_APPLICATION="$PYTHON"
  CAPTURE_ARGS=(workloads/python/tinyllama_prefill_operator.py "${WORKLOAD_ARGS[@]}")
fi
unset DYNAMIC_KERNEL_RANGE || true
bash scripts/capture_accel_sim_trace.sh \
  "$CAPTURE_APPLICATION" \
  "$RUN_DIR" \
  "${CAPTURE_ARGS[@]}"

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
