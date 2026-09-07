#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <operator> <kv-length> [output root] [manifest root]" >&2
  exit 2
fi

OPERATOR="$1"
KV_LENGTH="$2"
OUTPUT_ROOT="${3:-/opt/gpu-atlas/qualification/p21-sm89-decode-capture}"
MANIFEST_DIR="${4:-configs/hetero/operator_artifacts/p21_sm89_decode}"
MODEL_ROOT="${TINYLLAMA_MODEL_ROOT:-/opt/hf-cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
PYTHON="${TINYLLAMA_PYTHON:-/home/yueqi/anaconda3/envs/dl/bin/python}"
CONTEXT="${P21_INITIAL_CONTEXT:-16}"
BATCH_SIZE="${P21_BATCH_SIZE:-1}"
RUN_DIR="$OUTPUT_ROOT/tinyllama-decode-bs${BATCH_SIZE}-ctx${CONTEXT}-kv${KV_LENGTH}-${OPERATOR//_/-}"
WORKLOAD_PROGRAM="workloads/python/tinyllama_decode_operator.py"
if [[ "$OPERATOR" == "rope" ]]; then
  WORKLOAD_PROGRAM="workloads/python/tinyllama_decode_rope_sm86_operator.py"
fi
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

case "$OPERATOR" in
  token_embedding|attention_norm|qkv_projection|rope|causal_attention|output_projection|residual_add|mlp_norm|gate_up_projection|silu_multiply|down_projection|final_norm|lm_head|sampling) ;;
  *) echo "unsupported shape-locked Decode operator: $OPERATOR" >&2; exit 2 ;;
esac
if ! [[ "$KV_LENGTH" =~ ^[1-9][0-9]*$ ]] || (( KV_LENGTH <= CONTEXT )); then
  echo "kv-length must be an integer greater than the initial context" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "P21_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
for path in "$PYTHON" "$MODEL_ROOT" "$WORKLOAD_PROGRAM"; do
  [[ -e "$path" ]] || { echo "required Decode capture input is absent: $path" >&2; exit 2; }
done
REUSE_CAPTURED_TRACE=0
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${P21_REUSE_CAPTURED_TRACE:-0}" == "1" &&
        -s "$RUN_DIR/operator_metadata.json" &&
        -s "$RUN_DIR/traces/kernelslist.g" ]]; then
    REUSE_CAPTURED_TRACE=1
    echo "reusing complete captured Trace for identity sealing: $RUN_DIR"
  else
    echo "refusing to mix a new capture with existing files: $RUN_DIR" >&2
    exit 3
  fi
fi

GPU_ROW=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -n 1)
IFS=',' read -r GPU_NAME GPU_CC GPU_DRIVER <<<"$GPU_ROW"
GPU_NAME=$(echo "$GPU_NAME" | xargs)
GPU_CC=$(echo "$GPU_CC" | xargs)
GPU_DRIVER=$(echo "$GPU_DRIVER" | xargs)
TARGET_SM="${GPU_CC/.}"
if [[ "${P21_REQUIRE_SM89:-1}" == "1" ]] && {
  [[ "$GPU_NAME" != "NVIDIA GeForce RTX 4090" ]] || [[ "$TARGET_SM" != "89" ]];
}; then
  echo "P21 SM89 capture requires an RTX 4090 with compute capability 8.9" >&2
  exit 4
fi

WORKLOAD_ARGS=(
  --model "$MODEL_ROOT"
  --operator "$OPERATOR"
  --context "$CONTEXT"
  --kv-length "$KV_LENGTH"
  --batch-size "$BATCH_SIZE"
  --metadata-output "$RUN_DIR/operator_metadata.json"
  --capture-allocator-history
)
if [[ "${HETEROSIM_CAPTURE_RANGE:-process}" == "driver_profiler" ]]; then
  export ACTIVE_FROM_START=0
  WORKLOAD_ARGS+=(--driver-profiler --warmup "${HETEROSIM_CAPTURE_WARMUP:-1}")
elif [[ "${HETEROSIM_CAPTURE_RANGE:-process}" == "process" ]]; then
  export ACTIVE_FROM_START=1
else
  echo "HETEROSIM_CAPTURE_RANGE must be driver_profiler or process" >&2
  exit 2
fi
unset DYNAMIC_KERNEL_RANGE || true

if [[ "$REUSE_CAPTURED_TRACE" == "0" ]]; then
  bash scripts/capture_accel_sim_trace.sh \
    "$PYTHON" "$RUN_DIR" \
    "$WORKLOAD_PROGRAM" "${WORKLOAD_ARGS[@]}"
fi

SEAL_ARGS=(
  --metadata "$RUN_DIR/operator_metadata.json"
  --kernels-list "$RUN_DIR/traces/kernelslist.g"
  --expected-device-sm "$TARGET_SM"
)
if [[ "${P21_ALLOW_MIXED_AMPERE:-0}" == "1" ]]; then
  SEAL_ARGS+=(
    --allow-binary-sm 80
    --allow-binary-sm 86
    --replay-target-sm "${P21_REQUIRED_BINARY_SM:-86}"
  )
else
  SEAL_ARGS+=(--require-binary-sm "${P21_REQUIRED_BINARY_SM:-86}")
fi
BINARY_SM=$("${HETEROSIM_PYTHON:-$PYTHON}" \
  scripts/seal_gpu_trace_binary_identity.py "${SEAL_ARGS[@]}")

mkdir -p "$MANIFEST_DIR"
ARTIFACT="$MANIFEST_DIR/tinyllama_decode_bs${BATCH_SIZE}_ctx${CONTEXT}_kv${KV_LENGTH}_${OPERATOR}_sm${TARGET_SM}.json"
TRACE_MANIFEST="$MANIFEST_DIR/tinyllama_decode_bs${BATCH_SIZE}_ctx${CONTEXT}_kv${KV_LENGTH}_${OPERATOR}_sm${TARGET_SM}_trace.json"
"${HETEROSIM_PYTHON:-$PYTHON}" scripts/build_gpu_operator_artifact.py \
  --metadata "$RUN_DIR/operator_metadata.json" \
  --kernels-list "$RUN_DIR/traces/kernelslist.g" \
  --output "$ARTIFACT" \
  --trace-manifest-output "$TRACE_MANIFEST" \
  --gpu "$GPU_NAME" \
  --driver "$GPU_DRIVER" \
  --target-sm "$BINARY_SM"

echo "P21 Decode capture complete: device=SM${TARGET_SM} binary=SM${BINARY_SM} $RUN_DIR"
