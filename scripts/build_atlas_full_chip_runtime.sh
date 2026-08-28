#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATLAS_ROOT="${ATLAS_ROOT:-/opt/atlas/ATLAS-MICRO-2026}"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
COUPLED_ROOT="${ACCEL_SIM_RAMULATOR2_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT-ramulator2}"
BRIDGE_OUT="$COUPLED_ROOT/ramulator2_bridge"
ATLAS_PATCH="$PROJECT_ROOT/integrations/atlas/atlas_external_dram_service.patch"
RAMULATOR_PATCH="$PROJECT_ROOT/integrations/accel_sim_ramulator2/ramulator2_durable_write_callback.patch"
ATLAS_COMMIT="$(git -C "$ATLAS_ROOT" rev-parse HEAD)"
PATCH_SHA256="$(sha256sum "$ATLAS_PATCH" | awk '{print $1}')"
RAMULATOR_PATCH_SHA256="$(sha256sum "$RAMULATOR_PATCH" | awk '{print $1}')"
RUNTIME_ROOT="$DEPENDENCIES_ROOT/atlas-full-chip-$ATLAS_COMMIT"
MARKER="$RUNTIME_ROOT/.heterosim_external_dram_patch"
EXPECTED_MARKER="$ATLAS_COMMIT $PATCH_SHA256 $RAMULATOR_PATCH_SHA256"
BUILD_ROOT="$RUNTIME_ROOT/simulator/build"

for required in \
  "$ATLAS_ROOT/simulator/src/chip/chip.cpp" \
  "$ATLAS_PATCH" \
  "$RAMULATOR_PATCH" \
  "$BRIDGE_OUT/libramulator_gpgpusim_bridge.so"; do
  [[ -e "$required" ]] || {
    echo "missing prerequisite: $required" >&2
    exit 2
  }
done

if [[ -d "$RUNTIME_ROOT" &&
      (! -f "$MARKER" || "$(cat "$MARKER")" != "$EXPECTED_MARKER") ]]; then
  case "$RUNTIME_ROOT" in
    "$DEPENDENCIES_ROOT"/atlas-full-chip-*) ;;
    *)
      echo "refusing to replace unexpected ATLAS runtime root: $RUNTIME_ROOT" >&2
      exit 2
      ;;
  esac
  rm -rf -- "$RUNTIME_ROOT"
fi

if [[ ! -d "$RUNTIME_ROOT" ]]; then
  mkdir -p "$RUNTIME_ROOT"
  rsync -a --exclude build --exclude .git "$ATLAS_ROOT/" "$RUNTIME_ROOT/"
  git -C "$RUNTIME_ROOT" init -q
  git -C "$RUNTIME_ROOT" apply --check "$ATLAS_PATCH"
  git -C "$RUNTIME_ROOT" apply "$ATLAS_PATCH"
  git -C "$RUNTIME_ROOT/simulator/src/dram/ramulator2" init -q
  git -C "$RUNTIME_ROOT/simulator/src/dram/ramulator2" apply \
    --ignore-space-change --check "$RAMULATOR_PATCH"
  git -C "$RUNTIME_ROOT/simulator/src/dram/ramulator2" apply \
    --ignore-space-change "$RAMULATOR_PATCH"
  printf '%s\n' "$EXPECTED_MARKER" > "$MARKER"
fi

cmake -S "$RUNTIME_ROOT/simulator" -B "$BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_ROOT" --target atlasim-lib \
  -j"${ATLAS_BUILD_JOBS:-8}"

BRIDGE_RAMULATOR_LIB="$COUPLED_ROOT/atlas-simulator-durable/build/lib"
BRIDGE_RAMULATOR_SRC="$COUPLED_ROOT/atlas-simulator-durable/src/dram/ramulator2"

g++ -std=c++20 -O3 -fPIC -shared \
  -I"$PROJECT_ROOT/integrations/accel_sim_ramulator2" \
  -I"$BRIDGE_RAMULATOR_SRC/src" \
  -I"$BRIDGE_RAMULATOR_SRC/ext/spdlog/include" \
  -I"$COUPLED_ROOT/atlas-simulator-durable/3rd/yaml-cpp/include" \
  -I"$RUNTIME_ROOT/simulator/src" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/include" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/networks" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/power" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/routers" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/allocators" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/arbiters" \
  -I"$RUNTIME_ROOT/simulator/3rd/yaml-cpp/include" \
  "$PROJECT_ROOT/integrations/accel_sim_ramulator2/ramulator_bridge.cpp" \
  "$PROJECT_ROOT/integrations/accel_sim_ramulator2/atlas_hb_port.cpp" \
  "$PROJECT_ROOT/integrations/accel_sim_ramulator2/atlas_full_chip_memory_service.cpp" \
  "$PROJECT_ROOT/integrations/accel_sim_ramulator2/atlas_full_chip_runtime.cpp" \
  -L"$BRIDGE_RAMULATOR_LIB" -Wl,--no-as-needed -lramulator \
  -L"$BUILD_ROOT/lib" -latlasim-lib \
  -Wl,-rpath,"$BRIDGE_RAMULATOR_LIB" \
  -Wl,-rpath,"$BUILD_ROOT/lib" \
  -o "$BRIDGE_OUT/libramulator_gpgpusim_bridge.so"

g++ -std=c++20 -O2 \
  -I"$PROJECT_ROOT/integrations/accel_sim_ramulator2" \
  -I"$RUNTIME_ROOT/simulator/src" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/include" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/networks" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/power" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/routers" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/allocators" \
  -I"$RUNTIME_ROOT/simulator/src/noc/booksim2/src/arbiters" \
  -I"$RUNTIME_ROOT/simulator/3rd/yaml-cpp/include" \
  "$PROJECT_ROOT/integrations/accel_sim_ramulator2/full_chip_scheduler_smoke.cpp" \
  -L"$BRIDGE_OUT" -lramulator_gpgpusim_bridge \
  -L"$BUILD_ROOT/lib" -latlasim-lib \
  -Wl,-rpath,"$BRIDGE_OUT" \
  -Wl,-rpath,"$BUILD_ROOT/lib" \
  -Wl,-rpath,"$BRIDGE_RAMULATOR_LIB" \
  -o "$BRIDGE_OUT/full_chip_scheduler_smoke"

echo "ATLAS full-chip external-DRAM runtime build passed: $RUNTIME_ROOT"
echo "Full-chip scheduler smoke binary: $BRIDGE_OUT/full_chip_scheduler_smoke"
