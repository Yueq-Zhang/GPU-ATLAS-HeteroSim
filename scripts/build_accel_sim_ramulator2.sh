#!/usr/bin/env bash
set -euo pipefail

ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
BASE_ROOT="${ACCEL_SIM_BASE_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT}"
COUPLED_ROOT="${ACCEL_SIM_RAMULATOR2_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT-ramulator2}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
ATLAS_ROOT="${ATLAS_ROOT:-/opt/atlas/ATLAS-MICRO-2026}"
RAMULATOR_SRC="${RAMULATOR2_SRC:-$ATLAS_ROOT/simulator/src/dram/ramulator2}"
RAMULATOR_LIB="${RAMULATOR2_LIB:-$ATLAS_ROOT/simulator/build/lib}"
YAML_INCLUDE="${RAMULATOR2_YAML_INCLUDE:-$ATLAS_ROOT/simulator/3rd/yaml-cpp/include}"
BRIDGE_SOURCE="$PROJECT_ROOT/integrations/accel_sim_ramulator2"
BRIDGE_OUT="$COUPLED_ROOT/ramulator2_bridge"
PATCH_FILE="$BRIDGE_SOURCE/accel_sim_v2_ramulator2.patch"
MARKER="$COUPLED_ROOT/.heterosim_ramulator2_patch_applied"

for required in \
  "$BASE_ROOT/gpu-simulator/setup_environment.sh" \
  "$CUDA_ROOT/bin/nvcc" \
  "$RAMULATOR_SRC/src/memory_system/memory_system.h" \
  "$RAMULATOR_LIB/libramulator.so" \
  "$YAML_INCLUDE/yaml-cpp/yaml.h" \
  "$PATCH_FILE"; do
  [[ -e "$required" ]] || {
    echo "missing prerequisite: $required" >&2
    exit 2
  }
done

if [[ ! -d "$COUPLED_ROOT" ]]; then
  mkdir -p "$(dirname "$COUPLED_ROOT")"
  cp -a "$BASE_ROOT" "$COUPLED_ROOT"
fi

if [[ ! -f "$MARKER" ]]; then
  cd "$COUPLED_ROOT"
  git apply --check "$PATCH_FILE"
  git apply "$PATCH_FILE"
  printf '%s\n' "$ACCEL_COMMIT" > "$MARKER"
fi

mkdir -p "$BRIDGE_OUT"
cp "$BRIDGE_SOURCE/ramulator_bridge.h" "$BRIDGE_OUT/ramulator_bridge.h"

g++ -std=c++20 -O3 -fPIC -shared \
  -I"$BRIDGE_SOURCE" \
  -I"$RAMULATOR_SRC/src" \
  -I"$ATLAS_ROOT/simulator/src" \
  -I"$RAMULATOR_SRC/ext/spdlog/include" \
  -I"$YAML_INCLUDE" \
  "$BRIDGE_SOURCE/ramulator_bridge.cpp" \
  -L"$RAMULATOR_LIB" -Wl,--no-as-needed -lramulator \
  -Wl,-rpath,"$RAMULATOR_LIB" \
  -o "$BRIDGE_OUT/libramulator_gpgpusim_bridge.so"

g++ -std=c++20 -O2 \
  -I"$BRIDGE_SOURCE" \
  "$BRIDGE_SOURCE/smoke_test.cpp" \
  -L"$BRIDGE_OUT" -lramulator_gpgpusim_bridge \
  -Wl,-rpath,"$BRIDGE_OUT" \
  -o "$BRIDGE_OUT/ramulator_bridge_smoke"

"$BRIDGE_OUT/ramulator_bridge_smoke" \
  "$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbm3_32ch_gpu_only.yaml" \
  | tee "$BRIDGE_OUT/smoke.log"

export CUDA_INSTALL_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
cd "$COUPLED_ROOT"
set +u
source gpu-simulator/setup_environment.sh
set -u
cmake -S gpu-simulator -B gpu-simulator/build-ramulator2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHETEROSIM_ENABLE_RAMULATOR2=ON \
  -DHETEROSIM_RAMULATOR2_BRIDGE_DIR="$BRIDGE_OUT"
cmake --build gpu-simulator/build-ramulator2 \
  -j"${ACCEL_SIM_BUILD_JOBS:-8}"

SIMULATOR="$COUPLED_ROOT/gpu-simulator/build-ramulator2/accel-sim.out"
[[ -x "$SIMULATOR" ]] || {
  echo "coupled build did not produce $SIMULATOR" >&2
  exit 3
}

echo "Cycle-coupled Accel-Sim + Ramulator2 build passed: $SIMULATOR"
echo "Shared Ramulator2 bridge: $BRIDGE_OUT/libramulator_gpgpusim_bridge.so"
