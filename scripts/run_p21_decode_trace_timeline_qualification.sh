#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${HETEROSIM_PYTHON:-python3}"
OUTPUT_ROOT="${P21_TIMELINE_OUTPUT_ROOT:-${ROOT}/validation/p21/timeline}"
SOURCE_ROOT="${P21_SOURCE_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p21_sm89_decode}"
COUPLED_ROOT="${P21_COUPLED_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p21_sm89_decode_coupled}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

generate_case() {
  local layers="$1"
  local experiment="configs/hetero/experiments/p21_tinyllama_decode4_${layers}layer_bs1_ctx16_real_trace.json"
  local capability="configs/hetero/operator_capabilities/p21_tinyllama_decode4_${layers}layer_bs1_ctx16_real_trace.json"
  "$PYTHON_BIN" scripts/build_p21_decode_trace_timeline.py \
    --layers "$layers" \
    --source-root "$SOURCE_ROOT" \
    --coupled-root "$COUPLED_ROOT" \
    --experiment-output "$experiment" \
    --capability-output "$capability"
}

run_case() {
  local label="$1"
  local layers="$2"
  local config="configs/hetero/experiments/p21_tinyllama_decode4_${layers}layer_bs1_ctx16_real_trace.json"
  local experiment="p21_tinyllama_decode4_${layers}layer_bs1_ctx16_real_trace"
  local key
  key="$($PYTHON_BIN -c \
    "from frontend.hetero.runner import simulation_input_key; from frontend.hetero.schema import load_and_validate_config; print(simulation_input_key(load_and_validate_config('${config}')))" \
  )"
  mkdir -p "$OUTPUT_ROOT/logs"
  for leg in 1 2; do
    "$PYTHON_BIN" -m frontend.hetero.cli run \
      --config "$config" \
      --runs-root "$OUTPUT_ROOT/$label/leg${leg}" \
      > "$OUTPUT_ROOT/logs/${label}_leg${leg}.log" 2>&1
  done
  "$PYTHON_BIN" scripts/qualify_p21_decode_trace_timeline.py \
    --leg1 "$OUTPUT_ROOT/$label/leg1/$experiment/$key" \
    --leg2 "$OUTPUT_ROOT/$label/leg2/$experiment/$key" \
    --layers "$layers" \
    --output "$OUTPUT_ROOT/$label/qualification_record.json"
}

cd "$ROOT"
generate_case 1
generate_case 22
run_case one_layer 1
run_case twenty_two_layer 22

"$PYTHON_BIN" scripts/build_p21_qualification_summary.py \
  --one-layer "$OUTPUT_ROOT/one_layer/qualification_record.json" \
  --twenty-two-layer "$OUTPUT_ROOT/twenty_two_layer/qualification_record.json" \
  --output "$OUTPUT_ROOT/qualification_summary.json"

echo "P21 Decode real-Trace timeline qualification passed: $OUTPUT_ROOT/qualification_summary.json"
