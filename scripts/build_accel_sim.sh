#!/usr/bin/env bash
set -euo pipefail

ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
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
if [[ ! -f /usr/include/zstd.h ]]; then
  echo "zstd development headers not found (/usr/include/zstd.h)" >&2
  echo "Install them with: sudo apt-get install libzstd-dev" >&2
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
cmake -S gpu-simulator -B gpu-simulator/build -DCMAKE_BUILD_TYPE=Release
cmake --build gpu-simulator/build -j"${ACCEL_SIM_BUILD_JOBS:-8}"
make -j"${ACCEL_SIM_BUILD_JOBS:-8}" -C util/tracer_nvbit/tracer_tool
make -j"${ACCEL_SIM_BUILD_JOBS:-8}" \
  -C util/tracer_nvbit/tracer_tool/traces-processing

# Accel-Sim 2.0's CMake build emits the simulator in the build directory.
SIMULATOR="$ACCEL_ROOT/gpu-simulator/build/accel-sim.out"
TRACER="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POSTPROCESSOR="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
for artifact in "$SIMULATOR" "$TRACER" "$POSTPROCESSOR"; do
  [[ -e "$artifact" ]] || {
    echo "build did not produce $artifact" >&2
    exit 3
  }
done
echo "Accel-Sim build passed: $SIMULATOR"
echo "Accel-Sim 2.0 NVBit 1.8 tracer build passed: $TRACER"
