#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-/opt/gpu-atlas/qualification/official-traces}"
BUNDLE="$OUTPUT_ROOT/rodinia_2.0-ft.tgz"
EXPECTED_SHA256="aeb7de478785856e4ac834d12be0e71ab0df297f43fc02650e2ca90dea66d8b1"
URL="https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz"

mkdir -p "$OUTPUT_ROOT"
if [[ ! -f "$BUNDLE" ]]; then
  curl --fail --location --retry 20 --retry-all-errors --continue-at - \
    --output "$BUNDLE" "$URL"
fi
echo "$EXPECTED_SHA256  $BUNDLE" | sha256sum --check
mkdir -p "$OUTPUT_ROOT/extracted"
tar -xzf "$BUNDLE" -C "$OUTPUT_ROOT/extracted"
KERNELS_LIST="$OUTPUT_ROOT/extracted/rodinia_2.0-ft/9.1/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/kernelslist.g"
[[ -f "$KERNELS_LIST" ]] || {
  echo "official trace extraction did not produce $KERNELS_LIST" >&2
  exit 3
}
echo "Official Accel-Sim trace ready: $KERNELS_LIST"
