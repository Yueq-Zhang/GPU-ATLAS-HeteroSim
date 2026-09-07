#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${P20_OUTPUT_ROOT:-${ROOT}/validation/p20}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_case() {
  local label="$1"
  local config="$2"
  local experiment="$3"
  local layers="$4"
  local case_root="${OUTPUT_ROOT}/${label}"
  local key
  key="$(${PYTHON_BIN} -c \
    "from frontend.hetero.runner import simulation_input_key; from frontend.hetero.schema import load_and_validate_config; print(simulation_input_key(load_and_validate_config('${config}')))" \
  )"
  mkdir -p "${OUTPUT_ROOT}/logs"
  for leg in 1 2; do
    ${PYTHON_BIN} -m frontend.hetero.cli run \
      --config "${config}" \
      --runs-root "${case_root}/leg${leg}" \
      > "${OUTPUT_ROOT}/logs/${label}_leg${leg}.log" 2>&1
  done
  ${PYTHON_BIN} "${ROOT}/scripts/qualify_p20_decode_loop.py" \
    --leg1 "${case_root}/leg1/${experiment}/${key}" \
    --leg2 "${case_root}/leg2/${experiment}/${key}" \
    --layers "${layers}" \
    --output "${case_root}/qualification_record.json"
}

cd "${ROOT}"
run_case \
  one_layer \
  configs/hetero/experiments/p20_tinyllama_decode4_1layer_bs1_ctx16_request_cycle.json \
  p20_tinyllama_decode4_1layer_bs1_ctx16_request_cycle \
  1
run_case \
  twenty_two_layer \
  configs/hetero/experiments/p20_tinyllama_decode4_22layer_bs1_ctx16_request_cycle.json \
  p20_tinyllama_decode4_22layer_bs1_ctx16_request_cycle \
  22

${PYTHON_BIN} "${ROOT}/scripts/build_p20_qualification_summary.py" \
  --one-layer "${OUTPUT_ROOT}/one_layer/qualification_record.json" \
  --twenty-two-layer "${OUTPUT_ROOT}/twenty_two_layer/qualification_record.json" \
  --output "${OUTPUT_ROOT}/qualification_summary.json"

echo "P20 Decode-loop qualification passed: ${OUTPUT_ROOT}/qualification_summary.json"
