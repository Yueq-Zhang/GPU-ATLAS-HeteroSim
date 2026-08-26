#!/usr/bin/env bash
set -euo pipefail

ACCEL_COMMIT="c5296df152c99a28dd64e5d9560bd58a8fd2e774"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"

if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "CUDA 11.8 toolkit not found at $CUDA_ROOT" >&2
  echo "Install cuda-toolkit-11-8 or set ACCEL_SIM_CUDA_ROOT." >&2
  exit 2
fi
if [[ ! -f "$ACCEL_ROOT/gpu-simulator/setup_environment.sh" ]]; then
  echo "Accel-Sim source not found: $ACCEL_ROOT" >&2
  echo "Run scripts/install_accel_sim.sh first." >&2
  exit 2
fi

export CUDA_INSTALL_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
export ARCH="${ACCEL_SIM_NVBIT_ARCH:-sm_86}"

cd "$ACCEL_ROOT"
# shellcheck disable=SC1091
set +u
source gpu-simulator/setup_environment.sh
set -u
make -j"${ACCEL_SIM_BUILD_JOBS:-8}" -C gpu-simulator
make -j"${ACCEL_SIM_BUILD_JOBS:-8}" -C util/tracer_nvbit/tracer_tool
make -j"${ACCEL_SIM_BUILD_JOBS:-8}" \
  -C util/tracer_nvbit/tracer_tool/traces-processing

SIMULATOR="$ACCEL_ROOT/gpu-simulator/bin/release/accel-sim.out"
TRACER="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POSTPROCESSOR="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
for artifact in "$SIMULATOR" "$TRACER" "$POSTPROCESSOR"; do
  [[ -e "$artifact" ]] || {
    echo "build did not produce $artifact" >&2
    exit 3
  }
done
echo "Accel-Sim build passed: $SIMULATOR"
echo "NVBit tracer build passed: $TRACER"
