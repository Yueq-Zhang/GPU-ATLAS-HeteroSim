#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATLAS_ROOT="${ATLAS_ROOT:-$(cd "$PROJECT_ROOT/../ATLAS-MICRO-2026" && pwd)}"
BUILD_ROOT="${HETEROSIM_RAMULATOR_BUILD_ROOT:-$PROJECT_ROOT/build/ramulator2-standalone}"
SOURCE_ROOT="$BUILD_ROOT/atlas-simulator-atlas-patched-durable"
RAMULATOR_SRC="$SOURCE_ROOT/src/dram/ramulator2"
RAMULATOR_BUILD="$SOURCE_ROOT/build-ramulator-only"
RAMULATOR_LIB="$RAMULATOR_BUILD/lib"
BRIDGE_SOURCE="$PROJECT_ROOT/integrations/accel_sim_ramulator2"
BRIDGE_OUT="$BUILD_ROOT/bridge"
PATCH_FILE="$BRIDGE_SOURCE/ramulator2_durable_write_callback.patch"
ATLAS_PATCH_FILE="$ATLAS_ROOT/patches/ramulator2.patch"

for required in \
  "$ATLAS_ROOT/simulator/src/dram/ramulator2/src/memory_system/memory_system.h" \
  "$ATLAS_ROOT/simulator/3rd/yaml-cpp/include/yaml-cpp/yaml.h" \
  "$ATLAS_PATCH_FILE" \
  "$PATCH_FILE"; do
  [[ -e "$required" ]] || {
    echo "missing prerequisite: $required" >&2
    exit 2
  }
done

if [[ ! -f "$SOURCE_ROOT/.heterosim_source_ready" ]]; then
  mkdir -p "$SOURCE_ROOT"
  rsync -a --delete --exclude build --exclude .git \
    "$ATLAS_ROOT/simulator/" "$SOURCE_ROOT/"
  git -C "$RAMULATOR_SRC" init -q
  apply_once() {
    local patch="$1"
    if git -C "$RAMULATOR_SRC" apply --ignore-space-change --check "$patch"; then
      git -C "$RAMULATOR_SRC" apply --ignore-space-change "$patch"
    elif git -C "$RAMULATOR_SRC" apply --ignore-space-change --reverse --check "$patch"; then
      echo "patch already present in ATLAS source: $patch"
    else
      echo "patch is neither applicable nor already present: $patch" >&2
      exit 3
    fi
  }
  apply_once "$ATLAS_PATCH_FILE"
  apply_once "$PATCH_FILE"
  touch "$SOURCE_ROOT/.heterosim_source_ready"
fi

if [[ ! -f "$RAMULATOR_LIB/libramulator.so" ]]; then
  cmake -S "$BRIDGE_SOURCE/standalone_ramulator2_project" \
    -B "$RAMULATOR_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHETEROSIM_SOURCE_ROOT="$SOURCE_ROOT"
  cmake --build "$RAMULATOR_BUILD" --target ramulator \
    --parallel "${HETEROSIM_BUILD_JOBS:-4}"
fi

mkdir -p "$BRIDGE_OUT"
g++ -std=c++20 -O3 -fPIC -shared \
  -I"$BRIDGE_SOURCE" \
  -I"$RAMULATOR_SRC/src" \
  -I"$SOURCE_ROOT/src" \
  -I"$SOURCE_ROOT/3rd/yaml-cpp/include" \
  -I"$RAMULATOR_SRC/ext/spdlog/include" \
  -I"$RAMULATOR_SRC/ext/yaml-cpp/include" \
  "$BRIDGE_SOURCE/ramulator_bridge.cpp" \
  -L"$RAMULATOR_LIB" -Wl,--no-as-needed -lramulator \
  -Wl,-rpath,"$RAMULATOR_LIB" \
  -o "$BRIDGE_OUT/libramulator_gpgpusim_bridge.so"

[[ -f "$BRIDGE_OUT/libramulator_gpgpusim_bridge.so" ]] || {
  echo "standalone Ramulator2 bridge was not produced" >&2
  exit 3
}

echo "$BRIDGE_OUT/libramulator_gpgpusim_bridge.so"
