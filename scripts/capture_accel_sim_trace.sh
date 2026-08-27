#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <CUDA executable> <trace output directory> [application args...]" >&2
  exit 2
fi

APPLICATION="$(realpath "$1")"
TRACE_OUTPUT="$(realpath -m "$2")"
shift 2
ACCEL_COMMIT="64653015f85fb5664c84a10f48527e8897d289d0"
DEPENDENCIES_ROOT="${ACCEL_SIM_DEPS_ROOT:-/opt/gpu-atlas/dependencies}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-$DEPENDENCIES_ROOT/accel-sim-framework-$ACCEL_COMMIT}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
TRACER="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POSTPROCESSOR="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"

[[ -x "$APPLICATION" ]] || { echo "application is not executable: $APPLICATION" >&2; exit 2; }
[[ -f "$TRACER" ]] || { echo "tracer not built: $TRACER" >&2; exit 2; }
[[ -x "$POSTPROCESSOR" ]] || { echo "postprocessor not built: $POSTPROCESSOR" >&2; exit 2; }
[[ -x "$CUDA_ROOT/bin/nvdisasm" ]] || {
  echo "nvdisasm not found at $CUDA_ROOT/bin/nvdisasm" >&2
  echo "Install CUDA 11.8 or set ACCEL_SIM_CUDA_ROOT." >&2
  exit 2
}

mkdir -p "$TRACE_OUTPUT"
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export USER_DEFINED_FOLDERS=1
export TRACES_FOLDER="$TRACE_OUTPUT"
export NVBIT_INSTRUMENTATION_ENABLED="${NVBIT_INSTRUMENTATION_ENABLED:-1}"
export LD_PRELOAD="$TRACER${LD_PRELOAD:+:$LD_PRELOAD}"
"$APPLICATION" "$@"
unset LD_PRELOAD
RAW_TRACE_OUTPUT="$TRACE_OUTPUT/traces"
"$POSTPROCESSOR" "$RAW_TRACE_OUTPUT" -j "${ACCEL_SIM_TRACE_JOBS:-8}"
[[ -f "$RAW_TRACE_OUTPUT/kernelslist.g" ]] || {
  echo "capture completed without kernelslist.g" >&2
  exit 5
}
echo "Accel-Sim 2.0 compressed trace captured: $RAW_TRACE_OUTPUT/kernelslist.g"
