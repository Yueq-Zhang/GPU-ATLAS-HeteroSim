#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <CUDA executable> <trace output directory> [application args...]" >&2
  exit 2
fi

APPLICATION="$(realpath "$1")"
TRACE_OUTPUT="$(realpath -m "$2")"
shift 2
ACCEL_COMMIT="c5296df152c99a28dd64e5d9560bd58a8fd2e774"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT}"
TRACER="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POSTPROCESSOR="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"

[[ -x "$APPLICATION" ]] || { echo "application is not executable: $APPLICATION" >&2; exit 2; }
[[ -f "$TRACER" ]] || { echo "tracer not built: $TRACER" >&2; exit 2; }
[[ -x "$POSTPROCESSOR" ]] || { echo "postprocessor not built: $POSTPROCESSOR" >&2; exit 2; }

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR > 575 )) \
   && [[ "${ALLOW_UNSUPPORTED_NVBIT_DRIVER:-0}" != "1" ]]; then
  echo "NVBit 1.7.3 trace capture is blocked on driver $DRIVER_VERSION (>575)." >&2
  echo "Use a supported capture host/driver, then copy kernelslist.g and traces here." >&2
  echo "Set ALLOW_UNSUPPORTED_NVBIT_DRIVER=1 only for diagnostic attempts." >&2
  exit 4
fi

mkdir -p "$TRACE_OUTPUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USER_DEFINED_FOLDERS=1
export TRACES_FOLDER="$TRACE_OUTPUT"
export LD_PRELOAD="$TRACER${LD_PRELOAD:+:$LD_PRELOAD}"
"$APPLICATION" "$@"
unset LD_PRELOAD
"$POSTPROCESSOR" "$TRACE_OUTPUT/kernelslist"
[[ -f "$TRACE_OUTPUT/kernelslist.g" ]] || {
  echo "capture completed without kernelslist.g" >&2
  exit 5
}
echo "Accel-Sim trace captured: $TRACE_OUTPUT/kernelslist.g"
