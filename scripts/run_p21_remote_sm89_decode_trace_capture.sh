#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE="${P21_PHASE:-preflight}"
PYTHON="${P21_PYTHON:-/home/yueqi/anaconda3/envs/dl/bin/python}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-/opt/gpu-atlas/dependencies/accel-sim-framework-64653015f85fb5664c84a10f48527e8897d289d0}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda}"
OUTPUT_ROOT="${P21_OUTPUT_ROOT:-/opt/gpu-atlas/qualification/p21-sm89-decode-capture-process-v1}"
ARTIFACT_ROOT="${P21_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p21_sm89_decode}"
VALIDATION_ROOT="${P21_VALIDATION_ROOT:-validation/p21/remote_sm89}"
MODEL_ROOT="${TINYLLAMA_MODEL_ROOT:-/opt/hf-cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
TRACER="${ACCEL_SIM_TRACER:-$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/tracer_tool_sm89.so}"
POSTPROCESSOR="${ACCEL_SIM_POSTPROCESSOR:-$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing}"
OPERATORS=(
  token_embedding attention_norm qkv_projection rope causal_attention
  output_projection residual_add mlp_norm gate_up_projection silu_multiply
  down_projection final_norm lm_head sampling
)
KV_LENGTHS=(17 18 19 20)

case "$PHASE" in preflight|capture|all) ;; *) echo "P21_PHASE must be preflight, capture or all" >&2; exit 2 ;; esac
mkdir -p "$VALIDATION_ROOT"

GPU_ROW=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | head -n 1)
IFS=',' read -r GPU_NAME GPU_CC GPU_DRIVER GPU_MEMORY <<<"$GPU_ROW"
GPU_NAME=$(echo "$GPU_NAME" | xargs)
GPU_CC=$(echo "$GPU_CC" | xargs)
GPU_DRIVER=$(echo "$GPU_DRIVER" | xargs)
GPU_MEMORY=$(echo "$GPU_MEMORY" | xargs)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 4090" || "$GPU_CC" != "8.9" ]]; then
  echo "P21 remote capture requires RTX 4090 compute capability 8.9" >&2
  exit 3
fi
for path in "$PYTHON" "$ACCEL_ROOT" "$CUDA_ROOT/bin/nvdisasm" "$TRACER" "$POSTPROCESSOR"; do
  [[ -e "$path" ]] || { echo "required SM89 capture dependency is absent: $path" >&2; exit 4; }
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TINYLLAMA_PYTHON="$PYTHON"
export TINYLLAMA_MODEL_ROOT="$MODEL_ROOT"
export HETEROSIM_PYTHON="$PYTHON"
export ACCEL_SIM_ROOT="$ACCEL_ROOT"
export ACCEL_SIM_CUDA_ROOT="$CUDA_ROOT"
export ACCEL_SIM_TRACER="$TRACER"
export ACCEL_SIM_POSTPROCESSOR="$POSTPROCESSOR"
export HETEROSIM_CAPTURE_RANGE="${HETEROSIM_CAPTURE_RANGE:-process}"

"$PYTHON" - "$VALIDATION_ROOT/preflight.json" "$GPU_NAME" "$GPU_CC" "$GPU_DRIVER" "$GPU_MEMORY" "$ACCEL_ROOT" "$CUDA_ROOT" "$TRACER" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

output, gpu, cc, driver, memory, accel_root, cuda_root, tracer = sys.argv[1:]
accel = Path(accel_root)
tested = sorted(
    str(path) for path in accel.glob("gpu-simulator/**/tested-cfgs/*SM89*")
)
payload = {
    "schema_version": "hetero-p21-sm89-preflight/v1",
    "status": "passed_capture_only",
    "host": platform.node(),
    "gpu": gpu,
    "compute_capability": cc,
    "driver_version": driver,
    "memory": memory,
    "python": sys.version.split()[0],
    "pytorch": __import__("torch").__version__,
    "transformers": __import__("transformers").__version__,
    "cuda_root": cuda_root,
    "accel_sim_root": accel_root,
    "tracer": tracer,
    "tracer_sha256": hashlib.sha256(Path(tracer).read_bytes()).hexdigest(),
    "tested_sm89_configs": tested,
    "official_tested_sm89_config_available": bool(tested),
    "capture_allowed": True,
    "accel_sim_double_run_allowed": False,
    "performance_claim_allowed": False,
    "blocker": (
        None if tested else
        "Pinned Accel-Sim 2.0 has no tested SM89/RTX4090 timing configuration."
    ),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ "$PHASE" == "preflight" ]]; then
  echo "P21 SM89 capture preflight complete: $VALIDATION_ROOT/preflight.json"
  exit 0
fi

[[ -d "$MODEL_ROOT" ]] || {
  echo "fixed TinyLlama checkpoint is absent: $MODEL_ROOT" >&2
  exit 5
}

for kv_length in "${KV_LENGTHS[@]}"; do
  for operator in "${OPERATORS[@]}"; do
    stem="tinyllama_decode_bs1_ctx16_kv${kv_length}_${operator}_sm89"
    artifact="$ARTIFACT_ROOT/${stem}.json"
    trace_manifest="$ARTIFACT_ROOT/${stem}_trace.json"
    run_dir="$OUTPUT_ROOT/tinyllama-decode-bs1-ctx16-kv${kv_length}-${operator//_/-}"
    metadata="$run_dir/operator_metadata.json"
    kernels_list="$run_dir/traces/kernelslist.g"
    if [[ -s "$metadata" && -s "$kernels_list" ]]; then
      binary_sm=$("$PYTHON" scripts/seal_gpu_trace_binary_identity.py \
        --metadata "$metadata" \
        --kernels-list "$kernels_list" \
        --expected-device-sm 89 \
        --require-binary-sm 86)
      "$PYTHON" scripts/build_gpu_operator_artifact.py \
        --metadata "$metadata" \
        --kernels-list "$kernels_list" \
        --output "$artifact" \
        --trace-manifest-output "$trace_manifest" \
        --gpu "$GPU_NAME" \
        --driver "$GPU_DRIVER" \
        --target-sm "$binary_sm"
      echo "P21 resume: KV=$kv_length operator=$operator resealed as SM$binary_sm"
      continue
    fi
    if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
      echo "incomplete P21 capture requires inspection before resume: $run_dir" >&2
      exit 6
    fi
    echo "P21 SM89 capture: KV=$kv_length operator=$operator"
    bash scripts/capture_tinyllama_decode_operator.sh \
      "$operator" "$kv_length" "$OUTPUT_ROOT" "$ARTIFACT_ROOT"
  done
done

"$PYTHON" scripts/build_p21_sm89_decode_capture_catalog.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --output "$VALIDATION_ROOT/capture_catalog.json"

echo "P21 all 56 SM89 Decode captures complete: $VALIDATION_ROOT/capture_catalog.json"
