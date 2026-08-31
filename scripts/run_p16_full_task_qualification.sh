#!/usr/bin/env bash
set -euo pipefail

PYTHON="${HETEROSIM_PYTHON:-python3}"
CONFIG="${P16_CONFIG:-configs/hetero/experiments/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu.json}"
RUN_ROOT="${P16_RUN_ROOT:-validation/p16}"
RUN_NAME="p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu"

run_leg() {
  local leg="$1"
  local output
  output="$($PYTHON -m frontend.hetero.cli run \
    --config "$CONFIG" \
    --runs-root "$RUN_ROOT/$leg")"
  printf '%s\n' "$output" >&2
  local key
  key="$(printf '%s\n' "$output" \
    | sed -n 's/.*simulation_input_key=\([0-9a-f]\{64\}\).*/\1/p' \
    | tail -n 1)"
  if [[ -z "$key" ]]; then
    echo "P16 $leg did not report a simulation_input_key" >&2
    return 1
  fi
  printf '%s\n' "$key"
}

leg1_key="$(run_leg leg1)"
leg2_key="$(run_leg leg2)"
if [[ "$leg1_key" != "$leg2_key" ]]; then
  echo "P16 legs used different Simulation Keys" >&2
  exit 1
fi

leg1_dir="$RUN_ROOT/leg1/$RUN_NAME/$leg1_key"
leg2_dir="$RUN_ROOT/leg2/$RUN_NAME/$leg2_key"
output="$RUN_ROOT/p16_full_task_qualification.json"

"$PYTHON" scripts/summarize_p16_full_task_timeline.py \
  "$leg1_dir" \
  "$leg2_dir" \
  --output "$output"

echo "P16 full-task qualification passed: $output"
