#!/usr/bin/env bash
set -euo pipefail

# Pinned M5 Step-1 dependency installer. It never follows a moving branch/tag.
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
ACCEL_COMMIT="c5296df152c99a28dd64e5d9560bd58a8fd2e774"
GPGPU_COMMIT="68e1cd30efaecbd71b496822f9d88a5803b33841"
ACCEL_SHA256="2652061806585d27fe62d6ba2b8dc0226401c2680d7474c56768bde3c5a079d7"
GPGPU_SHA256="f288ddfcc06a873c77c9f8ac8a6d513d628c2713dd6130372a36cabfd3c7e92d"
PYBIND_SHA256="c6160321dc98e6e1184cc791fbeadd2907bb4a0ce0e447f2ea4ff8ab56550913"
NVBIT_SHA256="1eff502430936aeed570d73a62886d9057adffda519ada041f43486a36d4356b"
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
  "https://github.com/NVlabs/NVBit/releases/download/v1.7.3/nvbit-Linux-x86_64-1.7.3.tar.bz2" \
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
