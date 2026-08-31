#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <passed p15g_qualification.json>" >&2
  exit 2
fi

P15G_RECORD="$1"
PHASE="${P15H_PHASE:-all}"
PYTHON="${HETEROSIM_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"
CAPTURE_ROOT="${P15H_CAPTURE_ROOT:-/opt/gpu-atlas/qualification/p15h-address-capture}"
QUALIFICATION_ROOT="${P15H_QUALIFICATION_ROOT:-/opt/gpu-atlas/qualification/p15h-address}"
ARTIFACT_ROOT="${P15H_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p15h}"
BACKEND="configs/hetero/backends/gpu_accelsim_rtx3070_ramulator2_hbdram_edge_16ch_range_rebase.json"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
DEFAULT_OPERATORS=(
  rope
  causal_attention
  output_projection
  mlp_norm
  gate_up_projection
  silu_multiply
  down_projection
  final_norm
  lm_head
  sampling
)

if [[ -n "${P15H_OPERATORS:-}" ]]; then
  operator_text="${P15H_OPERATORS//,/ }"
  read -r -a OPERATORS <<<"$operator_text"
else
  OPERATORS=("${DEFAULT_OPERATORS[@]}")
fi

if [[ ${#OPERATORS[@]} -eq 0 ]]; then
  echo "P15H_OPERATORS selected no operators" >&2
  exit 2
fi

for operator in "${OPERATORS[@]}"; do
  known=false
  for candidate in "${DEFAULT_OPERATORS[@]}"; do
    if [[ "$operator" == "$candidate" ]]; then
      known=true
      break
    fi
  done
  if [[ "$known" != true ]]; then
    echo "unknown P15h operator: $operator" >&2
    exit 2
  fi
done

if [[ "$PHASE" != "capture" && "$PHASE" != "qualify" && "$PHASE" != "all" ]]; then
  echo "P15H_PHASE must be capture, qualify or all" >&2
  exit 2
fi

"$PYTHON" - "$P15G_RECORD" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if (
    payload.get("schema_version")
    != "hetero-p15g-prefill-timeline-qualification/v1"
    or payload.get("status") != "passed"
):
    raise SystemExit("P15h requires a passed P15g timeline qualification")
PY

mkdir -p "$CAPTURE_ROOT" "$QUALIFICATION_ROOT" "$ARTIFACT_ROOT"
export HETEROSIM_PYTHON="$PYTHON"
export TINYLLAMA_PYTHON="${TINYLLAMA_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"

for operator in "${OPERATORS[@]}"; do
  source_artifact="$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86.json"
  trace_manifest="$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_trace.json"
  qualification="$QUALIFICATION_ROOT/${operator//_/-}-range-rebase"
  coupled_artifact="$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_shared_hbdram_range_rebase.json"

  if [[ "$PHASE" != "capture" && -s "$coupled_artifact" && -s "$qualification/qualification_record.json" ]]; then
    "$PYTHON" -c \
      "from pathlib import Path; from frontend.hetero.operator_artifact import OperatorArtifactManifest; assert OperatorArtifactManifest.load(Path('$coupled_artifact')).request_cycle_ready"
    echo "P15h resume: $operator already qualified"
    continue
  fi

  if [[ "$PHASE" == "capture" || "$PHASE" == "all" ]]; then
    if [[ ! -s "$source_artifact" || ! -s "$trace_manifest" ]]; then
      bash scripts/capture_tinyllama_prefill_operator.sh \
        "$operator" 16 "$CAPTURE_ROOT" "$ARTIFACT_ROOT"
    else
      echo "P15h resume: $operator capture already exists"
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
    "$PYTHON" scripts/build_coupled_gpu_operator_artifact.py \
      --source-artifact "$source_artifact" \
      --backend-config "$BACKEND" \
      --qualification-record "$qualification/qualification_record.json" \
      --address-mode range_rebase \
      --output "$coupled_artifact"
  fi
done

if [[ "$PHASE" == "qualify" || "$PHASE" == "all" ]] && [[ ${#OPERATORS[@]} -eq ${#DEFAULT_OPERATORS[@]} ]]; then
  "$PYTHON" scripts/build_request_cycle_ready_catalog.py \
  --artifact-dir "$ARTIFACT_ROOT" \
  --qualification-root "$QUALIFICATION_ROOT" \
  --output-prefix "$ARTIFACT_ROOT/tinyllama_prefill_bs1_ctx16_twelve_gpu_range_rebase" \
  --existing-artifact "attention_norm=configs/hetero/operator_artifacts/p15e/tinyllama_prefill_bs1_ctx16_attention_norm_sm86_shared_hbdram_range_rebase.json" \
  --existing-qualification "attention_norm=/opt/gpu-atlas/qualification/p15e-address/attention-norm-range-rebase-qualified/qualification_record.json" \
  --existing-artifact "qkv_projection=configs/hetero/operator_artifacts/p15f/tinyllama_prefill_bs1_ctx16_qkv_projection_sm86_shared_hbdram_range_rebase.json" \
  --existing-qualification "qkv_projection=/opt/gpu-atlas/qualification/p15f-address/qkv-range-rebase-qualified-v3/qualification_record.json" \
    "${OPERATORS[@]}"
fi

echo "P15h range-rebase phase complete: $PHASE (${OPERATORS[*]})"
