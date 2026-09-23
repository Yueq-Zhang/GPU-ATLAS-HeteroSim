#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATLAS_ROOT="${ATLAS_ROOT:-$(cd "$PROJECT_ROOT/../ATLAS-MICRO-2026" && pwd)}"
SOURCE_ROOT="${BOOKSIM2_PATCHED_SOURCE_ROOT:-$PROJECT_ROOT/build/booksim2-patched-source}"
BUILD_ROOT="${BOOKSIM2_ADAPTER_BUILD_ROOT:-$PROJECT_ROOT/build/booksim2-adapter}"
UPSTREAM="$ATLAS_ROOT/simulator/src/noc/booksim2"
PATCH_FILE="$ATLAS_ROOT/patches/booksim2.patch"

if [[ ! -e "$UPSTREAM/.git" || ! -f "$PATCH_FILE" ]]; then
  echo "ATLAS BookSim2 source or patch is missing" >&2
  exit 2
fi

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$SOURCE_ROOT")"
  git clone --shared "$UPSTREAM" "$SOURCE_ROOT"
fi

if ! git -C "$SOURCE_ROOT" log -1 --format=%s | grep -qx "Initial commit"; then
  if ! git -C "$SOURCE_ROOT" diff --quiet || ! git -C "$SOURCE_ROOT" diff --cached --quiet; then
    echo "isolated BookSim2 source contains local changes; refusing to overwrite" >&2
    exit 3
  fi
  git -C "$SOURCE_ROOT" am --keep-cr --whitespace=nowarn --3way "$PATCH_FILE"
fi

cmake -S "$PROJECT_ROOT/integrations/booksim2" -B "$BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOKSIM2_SOURCE_DIR="$SOURCE_ROOT" \
  -DATLAS_SIMULATOR_SOURCE_DIR="$ATLAS_ROOT/simulator"
cmake --build "$BUILD_ROOT" --parallel "${BOOKSIM2_BUILD_JOBS:-2}"

LIBRARY="$BUILD_ROOT/libheterosim_booksim2_adapter.so"
if [[ ! -f "$LIBRARY" ]]; then
  echo "BookSim2 adapter library was not produced: $LIBRARY" >&2
  exit 4
fi
printf '%s\n' "$LIBRARY"
