#!/usr/bin/env bash
set -euo pipefail

CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
OUTPUT="${1:-/opt/gpu-atlas/build/tinyllama_simple_ops}"
mkdir -p "$(dirname "$OUTPUT")"
"$CUDA_ROOT/bin/nvcc" \
  -std=c++17 \
  -O3 \
  -lineinfo \
  -arch=sm_86 \
  workloads/cuda/tinyllama_simple_ops.cu \
  -o "$OUTPUT"
echo "TinyLlama simple operator workload built: $OUTPUT"
