#!/usr/bin/env bash
set -euo pipefail

PHASE="${P16_PHASE:-all}"
PYTHON="${HETEROSIM_PYTHON:-.venv/bin/python}"
CAPTURE_ROOT="${P16_CAPTURE_ROOT:-/opt/gpu-atlas/qualification/p16-simple-ops-capture}"
QUALIFICATION_ROOT="${P16_QUALIFICATION_ROOT:-/opt/gpu-atlas/qualification/p16-simple-ops-range-rebase}"
ARTIFACT_ROOT="${P16_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p16}"
BACKEND="configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch_range_rebase.json"
OPERATORS=(token_embedding residual_add)

if [[ "$PHASE" != "capture" && "$PHASE" != "qualify" && "$PHASE" != "all" ]]; then
  echo "P16_PHASE must be capture, qualify or all" >&2
  exit 2
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CAPTURE_ROOT" "$QUALIFICATION_ROOT" "$ARTIFACT_ROOT/qualification_records"

for operator in "${OPERATORS[@]}"; do
  source_artifact="$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86.json"
  trace_manifest="$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_trace.json"
  qualification="$QUALIFICATION_ROOT/${operator//_/-}-range-rebase"
  repository_record="$ARTIFACT_ROOT/qualification_records/${operator}_range_rebase.json"
  coupled_artifact="$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_shared_hbdram_range_rebase.json"

  if [[ "$PHASE" == "capture" || "$PHASE" == "all" ]]; then
    if [[ ! -s "$source_artifact" || ! -s "$trace_manifest" ]]; then
      bash scripts/capture_tinyllama_prefill_operator.sh \
        "$operator" 16 "$CAPTURE_ROOT" "$ARTIFACT_ROOT"
    else
      echo "P16 resume: $operator capture already exists"
    fi
  fi

  if [[ "$PHASE" == "qualify" || "$PHASE" == "all" ]]; then
    if [[ ! -s "$source_artifact" || ! -s "$trace_manifest" ]]; then
      echo "missing captured SM86 Artifact for $operator" >&2
      exit 3
    fi
    "$PYTHON" -m frontend.hetero.cli qualify-gpu \
      --resume-completed-runs \
      --backend-config "$BACKEND" \
      --trace-manifest "$trace_manifest" \
      --output "$qualification"
    cp "$qualification/qualification_record.json" "$repository_record"
    "$PYTHON" scripts/build_coupled_gpu_operator_artifact.py \
      --source-artifact "$source_artifact" \
      --backend-config "$BACKEND" \
      --qualification-record "$repository_record" \
      --address-mode range_rebase \
      --output "$coupled_artifact"
  fi
done

echo "P16 simple-operator phase complete: $PHASE (${OPERATORS[*]})"
