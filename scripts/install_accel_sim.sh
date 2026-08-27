#!/usr/bin/env bash
set -euo pipefail

# Pinned Accel-Sim 2.0 dependency installer. It never follows a moving branch/tag.
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
GPGPU_COMMIT="e10018b67a4b668e7b43f89280cf67624f1df4ff"
ACCEL_SHA256="1ab59739f00b006cefa9ce1dbe0a3196050bb82a8a820f77d71c45d8574ea711"
GPGPU_SHA256="a85c3c16b2636fca999fab3d2fb1d0475c7e54f7e204a75e5c0244377ddfbb98"
PYBIND_SHA256="c6160321dc98e6e1184cc791fbeadd2907bb4a0ce0e447f2ea4ff8ab56550913"
NVBIT_SHA256="72a2b827f9531dcb86b6be13844f267640fb440929d92944177029da6da2b9e1"
FINAL_ROOT="$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT"

for utility in curl sha256sum tar; do
  command -v "$utility" >/dev/null || {
    echo "missing prerequisite: $utility" >&2
    exit 2
  }
done

if [[ -d "$FINAL_ROOT" ]]; then
  echo "Accel-Sim dependency already exists: $FINAL_ROOT"
  exit 0
fi

mkdir -p "$DEPENDENCIES_ROOT"
INSTALL_TMP="$(mktemp -d "$DEPENDENCIES_ROOT/.accel-sim-install.XXXXXX")"
cleanup() {
  if [[ -n "${INSTALL_TMP:-}" && "$INSTALL_TMP" == "$DEPENDENCIES_ROOT"/.accel-sim-install.* ]]; then
    rm -rf -- "$INSTALL_TMP"
  fi
}
trap cleanup EXIT

download_checked() {
  local url="$1"
  local output="$2"
  local expected="$3"
  curl --fail --location --retry 5 --output "$output" "$url"
  echo "$expected  $output" | sha256sum --check --status || {
    echo "checksum mismatch: $output" >&2
    exit 3
  }
}

download_checked \
  "https://codeload.github.com/accel-sim/accel-sim-framework/tar.gz/$ACCEL_COMMIT" \
  "$INSTALL_TMP/accel-sim.tgz" "$ACCEL_SHA256"
download_checked \
  "https://codeload.github.com/gpgpu-sim/gpgpu-sim_distribution/tar.gz/$GPGPU_COMMIT" \
  "$INSTALL_TMP/gpgpu-sim.tgz" "$GPGPU_SHA256"
download_checked \
  "https://codeload.github.com/pybind/pybind11/tar.gz/v2.9.1" \
  "$INSTALL_TMP/pybind11.tgz" "$PYBIND_SHA256"
download_checked \
  "https://github.com/NVlabs/NVBit/releases/download/v1.8/nvbit-Linux-x86_64-1.8.tar.bz2" \
  "$INSTALL_TMP/nvbit.tar.bz2" "$NVBIT_SHA256"

mkdir -p "$INSTALL_TMP/stage"
tar -xzf "$INSTALL_TMP/accel-sim.tgz" -C "$INSTALL_TMP/stage" --strip-components=1
mkdir -p "$INSTALL_TMP/stage/gpu-simulator/gpgpu-sim"
tar -xzf "$INSTALL_TMP/gpgpu-sim.tgz" \
  -C "$INSTALL_TMP/stage/gpu-simulator/gpgpu-sim" --strip-components=1
mkdir -p "$INSTALL_TMP/stage/gpu-simulator/extern/pybind11"
tar -xzf "$INSTALL_TMP/pybind11.tgz" \
  -C "$INSTALL_TMP/stage/gpu-simulator/extern/pybind11" --strip-components=1
mkdir -p "$INSTALL_TMP/stage/util/tracer_nvbit/nvbit_release"
tar -xjf "$INSTALL_TMP/nvbit.tar.bz2" \
  -C "$INSTALL_TMP/stage/util/tracer_nvbit/nvbit_release" --strip-components=1

mv "$INSTALL_TMP/stage" "$FINAL_ROOT"
echo "Installed pinned Accel-Sim dependency: $FINAL_ROOT"
