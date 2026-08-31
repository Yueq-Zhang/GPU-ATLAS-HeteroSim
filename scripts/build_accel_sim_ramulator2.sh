#!/usr/bin/env bash
set -euo pipefail

ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
BASE_ROOT="${ACCEL_SIM_BASE_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT}"
COUPLED_ROOT="${ACCEL_SIM_RAMULATOR2_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT-ramulator2}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
ATLAS_ROOT="${ATLAS_ROOT:-/opt/atlas/ATLAS-MICRO-2026}"
SOURCE_RAMULATOR_SRC="${RAMULATOR2_SRC:-$ATLAS_ROOT/simulator/src/dram/ramulator2}"
SOURCE_YAML_INCLUDE="${RAMULATOR2_YAML_INCLUDE:-$ATLAS_ROOT/simulator/3rd/yaml-cpp/include}"
BRIDGE_SOURCE="$PROJECT_ROOT/integrations/accel_sim_ramulator2"
BRIDGE_OUT="$COUPLED_ROOT/ramulator2_bridge"
PATCH_FILE="$BRIDGE_SOURCE/accel_sim_v2_ramulator2.patch"
RAMULATOR_PATCH="$BRIDGE_SOURCE/ramulator2_durable_write_callback.patch"
MARKER="$COUPLED_ROOT/.heterosim_ramulator2_patch_applied"
PATCH_SHA256="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
RAMULATOR_PATCH_SHA256="$(sha256sum "$RAMULATOR_PATCH" | awk '{print $1}')"
DURABLE_RAMULATOR_ROOT="$COUPLED_ROOT/atlas-simulator-durable"
RAMULATOR_SRC="$DURABLE_RAMULATOR_ROOT/src/dram/ramulator2"
RAMULATOR_BUILD="$DURABLE_RAMULATOR_ROOT/build"
RAMULATOR_LIB="$RAMULATOR_BUILD/lib"
YAML_INCLUDE="$DURABLE_RAMULATOR_ROOT/3rd/yaml-cpp/include"

for required in \
  "$BASE_ROOT/gpu-simulator/setup_environment.sh" \
  "$CUDA_ROOT/bin/nvcc" \
  "$SOURCE_RAMULATOR_SRC/src/memory_system/memory_system.h" \
  "$SOURCE_YAML_INCLUDE/yaml-cpp/yaml.h" \
  "$PATCH_FILE" \
  "$RAMULATOR_PATCH"; do
  [[ -e "$required" ]] || {
    echo "missing prerequisite: $required" >&2
    exit 2
  }
done

EXPECTED_MARKER="$ACCEL_COMMIT $PATCH_SHA256 $RAMULATOR_PATCH_SHA256"
if [[ -d "$COUPLED_ROOT" &&
      (! -f "$MARKER" || "$(cat "$MARKER")" != "$EXPECTED_MARKER") ]]; then
  case "$COUPLED_ROOT" in
    "$DEPENDENCIES_ROOT"/accel-sim-framework-*-ramulator2) ;;
    *)
      echo "refusing to replace unexpected coupled root: $COUPLED_ROOT" >&2
      exit 2
      ;;
  esac
  rm -rf -- "$COUPLED_ROOT"
fi

if [[ ! -d "$COUPLED_ROOT" ]]; then
  mkdir -p "$(dirname "$COUPLED_ROOT")"
  cp -a "$BASE_ROOT" "$COUPLED_ROOT"
fi

if [[ ! -f "$MARKER" ]]; then
  cd "$COUPLED_ROOT"
  git apply --check "$PATCH_FILE"
  git apply "$PATCH_FILE"
  printf '%s\n' "$EXPECTED_MARKER" > "$MARKER"
fi

if [[ ! -f "$RAMULATOR_LIB/libramulator.so" ]]; then
  rsync -a --exclude build --exclude .git \
    "$ATLAS_ROOT/simulator/" "$DURABLE_RAMULATOR_ROOT/"
  git -C "$RAMULATOR_SRC" init -q
  git -C "$RAMULATOR_SRC" apply --ignore-space-change --check \
    "$RAMULATOR_PATCH"
  git -C "$RAMULATOR_SRC" apply --ignore-space-change "$RAMULATOR_PATCH"
  cmake -S "$DURABLE_RAMULATOR_ROOT" -B "$RAMULATOR_BUILD" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$RAMULATOR_BUILD" --target ramulator \
    -j"${ACCEL_SIM_BUILD_JOBS:-8}"
fi

mkdir -p "$BRIDGE_OUT"
cp "$BRIDGE_SOURCE/ramulator_bridge.h" "$BRIDGE_OUT/ramulator_bridge.h"
cp "$BRIDGE_SOURCE/atlas_hb_port.h" "$BRIDGE_OUT/atlas_hb_port.h"

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

g++ -std=c++20 -O2 \
  -I"$BRIDGE_SOURCE" \
  "$BRIDGE_SOURCE/address_translation_smoke.cpp" \
  -L"$BRIDGE_OUT" -lramulator_gpgpusim_bridge \
  -Wl,-rpath,"$BRIDGE_OUT" \
  -o "$BRIDGE_OUT/address_translation_smoke"

g++ -std=c++20 -O2 \
  -I"$BRIDGE_SOURCE" \
  -I"$ATLAS_ROOT/simulator/src" \
  -I"$SOURCE_YAML_INCLUDE" \
  "$BRIDGE_SOURCE/atlas_hb_port.cpp" \
  "$BRIDGE_SOURCE/dual_initiator_smoke.cpp" \
  -L"$BRIDGE_OUT" -lramulator_gpgpusim_bridge \
  -Wl,-rpath,"$BRIDGE_OUT" \
  -o "$BRIDGE_OUT/dual_initiator_smoke"

HETEROSIM_GPU_CLOCK_HZ=1200000000 \
HETEROSIM_LINK_CLOCK_HZ=400000000 \
HETEROSIM_GATEWAY_CLOCK_HZ=400000000 \
HETEROSIM_DRAM_CLOCK_HZ=400000000 \
"$BRIDGE_OUT/ramulator_bridge_smoke" \
  "$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbdram_edge_16ch_gpu_only.yaml" \
  | tee "$BRIDGE_OUT/smoke.log"

HETEROSIM_GPU_ADDRESS_BINDINGS=\
"$PROJECT_ROOT/configs/hetero/tests/address_bindings_smoke.tsv" \
"$BRIDGE_OUT/address_translation_smoke" \
  | tee "$BRIDGE_OUT/address_translation_smoke.log"

HETEROSIM_GPU_CLOCK_HZ=1200000000 \
HETEROSIM_LINK_CLOCK_HZ=400000000 \
HETEROSIM_GATEWAY_CLOCK_HZ=400000000 \
HETEROSIM_DRAM_CLOCK_HZ=400000000 \
HETEROSIM_LINK_REQUEST_BANDWIDTH_BPS=409600000000 \
HETEROSIM_LINK_RESPONSE_BANDWIDTH_BPS=409600000000 \
HETEROSIM_GATEWAY_ISSUE_WIDTH=16 \
"$BRIDGE_OUT/dual_initiator_smoke" \
  "$PROJECT_ROOT/configs/hetero/memory/ramulator2_hbdram_edge_1ch_shared.yaml" \
  | tee "$BRIDGE_OUT/dual_initiator_smoke.log"

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
