#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PHASE="${P30_P31_PHASE:-all}"
OUTPUT_ROOT="${P30_P31_OUTPUT_ROOT:-/home/yueqi/gpu-atlas/framework-runtimes/validation/p30-p31-remote}"
HF_PYTHON="${P31_HF_PYTHON:-/home/yueqi/gpu-atlas/framework-runtimes/hf-live/bin/python}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-/opt/gpu-atlas/dependencies/accel-sim-framework-64653015f85fb5664c84a10f48527e8897d289d0}"
TRACER="${ACCEL_SIM_TRACER:-$ACCEL_ROOT/util/tracer_nvbit/tracer_tool/tracer_tool_sm89.so}"
BACKEND="configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch_range_rebase.json"

case "$PHASE" in
  all|p30|capture|artifacts|atlas|replay|qualify) ;;
  *) echo "P30_P31_PHASE must be all, p30, capture, artifacts, atlas, replay or qualify" >&2; exit 2 ;;
esac

GPU=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 | tr -d ' ')
if [[ "$GPU" != "NVIDIAGeForceRTX4090,8.9" ]]; then
  echo "P30/P31 must run on the remote RTX 4090/SM89 host" >&2
  exit 3
fi
for required in "$HF_PYTHON" "$TRACER" "$ACCEL_ROOT"; do
  [[ -e "$required" ]] || { echo "missing remote dependency: $required" >&2; exit 4; }
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/yueqi/gpu-atlas/framework-runtimes/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$OUTPUT_ROOT"

run_p30() {
  mkdir -p "$OUTPUT_ROOT/p30/run1" "$OUTPUT_ROOT/p30/run2"
  for run in run1 run2; do
    if [[ ! -s "$OUTPUT_ROOT/p30/$run/bundle.json" ]]; then
      "$HF_PYTHON" scripts/run_p30_huggingface_online.py \
        --capture-allocator-history \
        --output "$OUTPUT_ROOT/p30/$run/bundle.json"
    fi
  done
  python3 scripts/qualify_p30_huggingface_online.py \
    --run-a "$OUTPUT_ROOT/p30/run1/bundle.json" \
    --run-b "$OUTPUT_ROOT/p30/run2/bundle.json" \
    --output "$OUTPUT_ROOT/p30/qualification_record.json"
}

capture_gpu() {
  local capture_root="$OUTPUT_ROOT/p31/hf_layer0_capture"
  local kernels="$capture_root/traces/kernelslist.g"
  if [[ -s "$kernels" && -s "$capture_root/p30_bundle.json" ]]; then
    echo "P31 resume: framework-selected Layer-0 SASS already captured"
    return
  fi
  if [[ -d "$capture_root" ]] && find "$capture_root" -mindepth 1 -print -quit | grep -q .; then
    echo "partial P31 capture requires inspection: $capture_root" >&2
    exit 5
  fi
  mkdir -p "$capture_root"
  export ACTIVE_FROM_START=1
  export NVBIT_INSTRUMENTATION_ENABLED=0
  export ACCEL_SIM_INJECTION_MODE=1
  export ACCEL_SIM_TRACER="$TRACER"
  export ACCEL_SIM_CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda}"
  export ACCEL_SIM_TRACE_JOBS="${ACCEL_SIM_TRACE_JOBS:-8}"
  chmod +x scripts/p31_hf_runtime_entrypoint.sh
  bash scripts/capture_accel_sim_trace.sh \
    scripts/p31_hf_runtime_entrypoint.sh "$capture_root" \
    scripts/run_p30_huggingface_online.py \
    --capture-allocator-history \
    --nvbit-instrumentation-phase decode_step \
    --output "$capture_root/p30_bundle.json"
}

build_artifacts() {
  local capture_root="$OUTPUT_ROOT/p31/hf_layer0_capture"
  local artifact_root="$OUTPUT_ROOT/p31/hf_layer0_artifact"
  python3 scripts/build_p31_huggingface_trace_artifact.py \
    --bundle "$capture_root/p30_bundle.json" \
    --kernels-list "$capture_root/traces/kernelslist.g" \
    --output "$artifact_root/source_artifact.json" \
    --trace-manifest-output "$artifact_root/trace_manifest.json" \
    --binding-directory "$artifact_root/binding"
}

build_atlas() {
  local bundle="$OUTPUT_ROOT/p31/hf_layer0_capture/p30_bundle.json"
  python3 scripts/build_p31_atlas_executable.py \
    --p30-bundle "$bundle" --output-directory "$OUTPUT_ROOT/p31/atlas_run1"
  python3 scripts/build_p31_atlas_executable.py \
    --p30-bundle "$bundle" --output-directory "$OUTPUT_ROOT/p31/atlas_run2"
  cmp "$OUTPUT_ROOT/p31/atlas_run1/atlas_executable_artifact.json" \
    "$OUTPUT_ROOT/p31/atlas_run2/atlas_executable_artifact.json"
  cmp "$OUTPUT_ROOT/p31/atlas_run1/atlas_memory_trace.jsonl.gz" \
    "$OUTPUT_ROOT/p31/atlas_run2/atlas_memory_trace.jsonl.gz"
}

replay_gpu() {
  local artifact_root="$OUTPUT_ROOT/p31/hf_layer0_artifact"
  local replay="$OUTPUT_ROOT/p31/hf_layer0_replay"
  if [[ -s "$replay/stats.json" ]]; then
    echo "P31 resume: single GPU address-audit replay already passed"
    return
  fi
  python3 scripts/run_accel_sim_single.py \
    --backend-config "$BACKEND" \
    --trace-manifest "$artifact_root/trace_manifest.json" \
    --output "$replay"
}

qualify() {
  python3 scripts/qualify_p31_framework_artifacts.py \
    --p30-qualification "$OUTPUT_ROOT/p30/qualification_record.json" \
    --gpu-artifact "$OUTPUT_ROOT/p31/hf_layer0_artifact/source_artifact.json" \
    --gpu-trace-manifest "$OUTPUT_ROOT/p31/hf_layer0_artifact/trace_manifest.json" \
    --gpu-binding "$OUTPUT_ROOT/p31/hf_layer0_artifact/binding/online_address_binding.json" \
    --gpu-replay-stats "$OUTPUT_ROOT/p31/hf_layer0_replay/stats.json" \
    --atlas-run-a "$OUTPUT_ROOT/p31/atlas_run1/atlas_memory_trace_summary.json" \
    --atlas-run-b "$OUTPUT_ROOT/p31/atlas_run2/atlas_memory_trace_summary.json" \
    --output "$OUTPUT_ROOT/p31/qualification_record.json"
}

if [[ "$PHASE" == "all" || "$PHASE" == "p30" ]]; then run_p30; fi
if [[ "$PHASE" == "all" || "$PHASE" == "capture" ]]; then capture_gpu; fi
if [[ "$PHASE" == "all" || "$PHASE" == "artifacts" ]]; then build_artifacts; fi
if [[ "$PHASE" == "all" || "$PHASE" == "atlas" ]]; then build_atlas; fi
if [[ "$PHASE" == "all" || "$PHASE" == "replay" ]]; then replay_gpu; fi
if [[ "$PHASE" == "all" || "$PHASE" == "qualify" ]]; then qualify; fi

echo "P30/P31 remote phase complete: $PHASE"
