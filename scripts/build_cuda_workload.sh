#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
OUTPUT="${1:-$PROJECT_ROOT/build/workloads/vector_add}"

[[ -x "$CUDA_ROOT/bin/nvcc" ]] || {
  echo "CUDA toolkit not found at $CUDA_ROOT" >&2
  exit 2
}
mkdir -p "$(dirname "$OUTPUT")"
"$CUDA_ROOT/bin/nvcc" -std=c++14 -O2 -lineinfo -arch=sm_86 \
  "$PROJECT_ROOT/workloads/cuda/vector_add.cu" -o "$OUTPUT"
"$OUTPUT"
echo "CUDA workload built and verified: $OUTPUT"
