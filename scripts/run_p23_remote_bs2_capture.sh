#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export P21_BATCH_SIZE="${P23_BATCH_SIZE:-2}"
export P21_INITIAL_CONTEXT="${P23_INITIAL_CONTEXT:-16}"
export P21_REQUIRE_SM89="${P23_REQUIRE_SM89:-1}"
export P21_REQUIRED_BINARY_SM="${P23_REQUIRED_BINARY_SM:-86}"
export P21_ALLOW_MIXED_AMPERE="${P23_ALLOW_MIXED_AMPERE:-1}"
export P21_REUSE_CAPTURED_TRACE="${P23_REUSE_CAPTURED_TRACE:-1}"
export HETEROSIM_CAPTURE_RANGE="${P23_CAPTURE_RANGE:-process}"

KV_LENGTH="${P23_KV_LENGTH:-17}"
OUTPUT_ROOT="${P23_CAPTURE_ROOT:-/opt/gpu-atlas/qualification/p23-bs2-decode-capture}"
MANIFEST_ROOT="${P23_ARTIFACT_ROOT:-configs/hetero/operator_artifacts/p23_sm89_decode_bs2}"
OPERATORS=(
  token_embedding attention_norm qkv_projection rope causal_attention
  output_projection residual_add mlp_norm gate_up_projection silu_multiply
  down_projection final_norm lm_head sampling
)

if [[ -n "${P23_OPERATORS:-}" ]]; then
  IFS=',' read -r -a OPERATORS <<<"$P23_OPERATORS"
fi

mkdir -p "$OUTPUT_ROOT" "$MANIFEST_ROOT"
for operator in "${OPERATORS[@]}"; do
  stem="tinyllama_decode_bs${P21_BATCH_SIZE}_ctx${P21_INITIAL_CONTEXT}_kv${KV_LENGTH}_${operator}_sm89"
  artifact="$MANIFEST_ROOT/${stem}.json"
  trace_manifest="$MANIFEST_ROOT/${stem}_trace.json"
  if [[ -s "$artifact" && -s "$trace_manifest" ]]; then
    echo "P23 capture already complete: $operator"
    continue
  fi
  echo "P23 capture begin: operator=$operator batch=$P21_BATCH_SIZE kv=$KV_LENGTH"
  bash scripts/capture_tinyllama_decode_operator.sh \
    "$operator" "$KV_LENGTH" "$OUTPUT_ROOT" "$MANIFEST_ROOT"
  [[ -s "$artifact" && -s "$trace_manifest" ]] || {
    echo "P23 capture outputs are incomplete: $operator" >&2
    exit 4
  }
  echo "P23 capture complete: operator=$operator"
done

"${HETEROSIM_PYTHON:-${TINYLLAMA_PYTHON:-python3}}" \
  scripts/build_p23_bs2_decode_capture_catalog.py \
  --artifact-root "$MANIFEST_ROOT" \
  --output "${P23_CAPTURE_CATALOG:-validation/p23/capture_catalog.json}"

echo "P23 one-layer BS2 capture set complete: operators=${#OPERATORS[@]}"
