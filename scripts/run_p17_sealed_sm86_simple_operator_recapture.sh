#!/usr/bin/env bash
set -euo pipefail

PYTHON="${HETEROSIM_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-/opt/gpu-atlas/dependencies/accel-sim-framework-64653015f85fb5664c84a10f48527e8897d289d0}"
CAPTURE_ROOT="${P17_SIMPLE_CAPTURE_ROOT:-/opt/gpu-atlas/qualification/p17-simple-sm86-sealed}"
MANIFEST_ROOT="${P17_SIMPLE_MANIFEST_ROOT:-configs/hetero/operator_artifacts/p17_sealed}"
BINARY="${P17_SIMPLE_BINARY:-/opt/gpu-atlas/build/p17_tinyllama_simple_ops_sm86_sealed}"
RECORD="${P17_SIMPLE_RECAPTURE_RECORD:-validation/p17/sm86_sealed_recapture/recapture_record.json}"
OPERATORS=(token_embedding residual_add)

for tool in "$PYTHON" "$CUDA_ROOT/bin/nvcc" "$CUDA_ROOT/bin/cuobjdump"; do
  [[ -x "$tool" ]] || { echo "required executable is absent: $tool" >&2; exit 2; }
done

TRACER_DIR="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool"
[[ -d "$TRACER_DIR" ]] || { echo "NVBit tracer source is absent: $TRACER_DIR" >&2; exit 2; }
if [[ -d "$CAPTURE_ROOT" ]] && find "$CAPTURE_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "capture root is not empty; choose a new P17_SIMPLE_CAPTURE_ROOT: $CAPTURE_ROOT" >&2
  exit 3
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export ACCEL_SIM_CUDA_ROOT="$CUDA_ROOT"
mkdir -p "$CAPTURE_ROOT" "$MANIFEST_ROOT" "$(dirname "$BINARY")" "$(dirname "$RECORD")"

bash scripts/build_tinyllama_simple_ops.sh "$BINARY"
mapfile -t CUBINS < <("$CUDA_ROOT/bin/cuobjdump" --list-elf "$BINARY" | sed -n 's/.*\.\(sm_[0-9][0-9]*\)\.cubin.*/\1/p')
if [[ ${#CUBINS[@]} -eq 0 ]] || printf '%s\n' "${CUBINS[@]}" | grep -qvx 'sm_86'; then
  echo "sealed workload must contain only SM86 cubins" >&2
  exit 4
fi
BINARY_SHA256=$(sha256sum "$BINARY")
BINARY_SHA256=${BINARY_SHA256%% *}

HOST_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
HOST_ARCH="sm_${HOST_CC/./}"
export CUDA_INSTALL_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
export ARCH="$HOST_ARCH"
make -C "$TRACER_DIR" clean >/dev/null 2>&1 || true
make -C "$TRACER_DIR" -j"${ACCEL_SIM_BUILD_JOBS:-8}"
TRACER="$TRACER_DIR/tracer_tool.so"
[[ -s "$TRACER" ]] || { echo "tracer build did not produce $TRACER" >&2; exit 4; }
TRACER_SHA256=$(sha256sum "$TRACER")
TRACER_SHA256=${TRACER_SHA256%% *}

for operator in "${OPERATORS[@]}"; do
  run="$CAPTURE_ROOT/tinyllama-prefill-bs1-ctx16-${operator//_/-}"
  mkdir -p "$run"
  bash scripts/capture_accel_sim_trace.sh \
    "$BINARY" "$run" \
    --operator "$operator" \
    --context 16 \
    --metadata-output "$run/operator_metadata.json"
  "$PYTHON" scripts/build_gpu_operator_artifact.py \
    --metadata "$run/operator_metadata.json" \
    --kernels-list "$run/traces/kernelslist.g" \
    --output "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86.json" \
    --trace-manifest-output \
    "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_trace.json"
  "$PYTHON" - \
    "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_trace.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["capture"]["source"] = "P17 sealed SM86 cubin capture"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
done

"$PYTHON" - \
  "$CAPTURE_ROOT" "$MANIFEST_ROOT" "$BINARY" "$BINARY_SHA256" \
  "$TRACER_SHA256" "$HOST_CC" "$HOST_ARCH" "$RECORD" "${CUBINS[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

capture_root = Path(sys.argv[1]).resolve()
manifest_root = Path(sys.argv[2]).resolve()
binary = Path(sys.argv[3]).resolve()
binary_sha256 = sys.argv[4]
tracer_sha256 = sys.argv[5]
host_cc = sys.argv[6]
host_arch = sys.argv[7]
record = Path(sys.argv[8]).resolve()
cubins = sys.argv[9:]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


gpu_fields = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ],
    text=True,
).splitlines()[0].split(",", maxsplit=1)
operators = {}
for operator in ("token_embedding", "residual_add"):
    run = capture_root / f"tinyllama-prefill-bs1-ctx16-{operator.replace('_', '-')}"
    trace_files = list((run / "traces").glob("*.tracez"))
    if len(trace_files) != 1:
        raise SystemExit(f"expected one trace for {operator}, found {len(trace_files)}")
    manifest = manifest_root / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json"
    operators[operator] = {
        "metadata_sha256": digest(run / "operator_metadata.json"),
        "kernels_list_sha256": digest(run / "traces/kernelslist.g"),
        "trace_sha256": digest(trace_files[0]),
        "trace_manifest_sha256": digest(manifest),
    }
payload = {
    "schema_version": "hetero-p17-sm86-sealed-recapture/v1",
    "capture_host": {
        "gpu": gpu_fields[0].strip(),
        "compute_capability": host_cc,
        "driver_version": gpu_fields[1].strip(),
    },
    "target_binary": {
        "path": str(binary),
        "sha256": binary_sha256,
        "cuda_toolkit": "11.8",
        "embedded_cubins": cubins,
    },
    "tracer": {
        "name": "NVBit",
        "version": "1.8",
        "instrumentation_arch": host_arch,
        "sha256": tracer_sha256,
    },
    "operators": operators,
    "claim_boundary": {
        "trace_target_sm": 86,
        "physical_capture_gpu_is_not_the_simulated_gpu": True,
        "native_rtx3070_binary_identity_verified": False,
        "performance_pairing_allowed": False,
    },
}
record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"P17 sealed SM86 recapture complete: {record}")
PY
