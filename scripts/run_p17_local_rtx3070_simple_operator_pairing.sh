#!/usr/bin/env bash
set -euo pipefail

PYTHON="${HETEROSIM_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-/opt/gpu-atlas/dependencies/accel-sim-framework-64653015f85fb5664c84a10f48527e8897d289d0}"
WORK_ROOT="${P17_LOCAL_WORK_ROOT:-/tmp/p17-local-rtx3070-same-binary-v1}"
BINARY="${P17_LOCAL_BINARY:-/opt/gpu-atlas/build/p17_tinyllama_simple_ops_sm86_same_binary_v1}"
EVIDENCE_ROOT="${P17_LOCAL_EVIDENCE_ROOT:-configs/hetero/operator_artifacts/p17_sealed/evidence/local_rtx3070_same_binary_v1}"
MANIFEST_ROOT="${P17_SIMPLE_MANIFEST_ROOT:-configs/hetero/operator_artifacts/p17_sealed}"
NATIVE_ROOT="${P17_LOCAL_NATIVE_ROOT:-validation/p17/native_rtx3070/sealed_same_binary_v1}"
RECORD="${P17_SIMPLE_RECAPTURE_RECORD:-validation/p17/sm86_sealed_recapture/recapture_record.json}"
IDENTITY_CATALOG="${P17_EXECUTION_IDENTITY_CATALOG:-validation/p17/sm86_sealed_recapture/execution_identity_catalog.json}"
NATIVE_CATALOG="${P17_NATIVE_CATALOG:-validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json}"
QUALIFICATION_STAGE="${P17_LOCAL_QUALIFICATION_STAGE:-$WORK_ROOT/native-vram-qualification}"
QUALIFICATION_ROOT="${P17_NATIVE_VRAM_ROOT:-validation/p17/native_vram_qualification}"
SIMULATOR_CATALOG="${P17_SIMULATOR_CATALOG:-validation/p17/gpu_operator_pairing/simulator_native_vram.json}"
PAIRING_AUDIT="${P17_PAIRING_AUDIT:-validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json}"
MEASUREMENT_MANIFEST="${P17_MEASUREMENT_MANIFEST:-validation/p17/gpu_operator_pairing/measurement_manifest.json}"
IMPORT_MANIFEST="${P17_IMPORT_MANIFEST:-validation/p17/native_vram_qualification/import_manifest.json}"
TRACE_OVERRIDES="${P17_TRACE_MANIFEST_OVERRIDES:-configs/hetero/calibration/p17_native_vram_trace_overrides.json}"
OPERATORS=(token_embedding residual_add)

for tool in "$PYTHON" "$CUDA_ROOT/bin/nvcc" "$CUDA_ROOT/bin/cuobjdump"; do
  [[ -x "$tool" ]] || { echo "required executable is absent: $tool" >&2; exit 2; }
done
[[ -d "$ACCEL_ROOT/util/tracer_nvbit/tracer_tool" ]] || {
  echo "NVBit tracer source is absent under $ACCEL_ROOT" >&2
  exit 2
}
if [[ -d "$WORK_ROOT" ]] && find "$WORK_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "work root is not empty; choose a new P17_LOCAL_WORK_ROOT: $WORK_ROOT" >&2
  exit 3
fi
if [[ -d "$EVIDENCE_ROOT" ]] && find "$EVIDENCE_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "evidence root is not empty; choose a new P17_LOCAL_EVIDENCE_ROOT" >&2
  exit 3
fi
if [[ -d "$NATIVE_ROOT" ]] && find "$NATIVE_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "native root is not empty; choose a new P17_LOCAL_NATIVE_ROOT" >&2
  exit 3
fi

GPU_ROW=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | head -n 1)
IFS=',' read -r GPU_NAME GPU_CC GPU_DRIVER GPU_MEMORY <<<"$GPU_ROW"
GPU_NAME=$(echo "$GPU_NAME" | xargs)
GPU_CC=$(echo "$GPU_CC" | xargs)
GPU_DRIVER=$(echo "$GPU_DRIVER" | xargs)
GPU_MEMORY=$(echo "$GPU_MEMORY" | xargs)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 3070" || "$GPU_CC" != "8.6" ]]; then
  echo "P17 local pairing requires RTX 3070 compute capability 8.6" >&2
  exit 4
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export ACCEL_SIM_CUDA_ROOT="$CUDA_ROOT"
mkdir -p "$WORK_ROOT" "$(dirname "$BINARY")"
bash scripts/build_tinyllama_simple_ops.sh "$BINARY"
mapfile -t CUBINS < <("$CUDA_ROOT/bin/cuobjdump" --list-elf "$BINARY" | sed -n 's/.*\.\(sm_[0-9][0-9]*\)\.cubin.*/\1/p')
if [[ ${#CUBINS[@]} -eq 0 ]] || printf '%s\n' "${CUBINS[@]}" | grep -qvx 'sm_86'; then
  echo "same-Binary workload must contain only SM86 cubins" >&2
  exit 4
fi
BINARY_SHA256=$(sha256sum "$BINARY")
BINARY_SHA256=${BINARY_SHA256%% *}

TRACER_DIR="$ACCEL_ROOT/util/tracer_nvbit/tracer_tool"
export CUDA_INSTALL_PATH="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
export ARCH=sm_86
make -C "$TRACER_DIR" clean >/dev/null 2>&1 || true
make -C "$TRACER_DIR" -j"${ACCEL_SIM_BUILD_JOBS:-8}"
TRACER="$TRACER_DIR/tracer_tool.so"
[[ -s "$TRACER" ]] || { echo "tracer build failed" >&2; exit 4; }
TRACER_SHA256=$(sha256sum "$TRACER")
TRACER_SHA256=${TRACER_SHA256%% *}

for operator in "${OPERATORS[@]}"; do
  native="$WORK_ROOT/native/$operator.json"
  capture="$WORK_ROOT/capture/${operator//_/-}"
  mkdir -p "$(dirname "$native")" "$capture"
  "$BINARY" --operator "$operator" --context 16 \
    --native-measurement-output "$native" --warmup 50 --iterations 500
  "$PYTHON" -m json.tool "$native" >/dev/null
  bash scripts/capture_accel_sim_trace.sh \
    "$BINARY" "$capture" \
    --operator "$operator" --context 16 \
    --metadata-output "$capture/operator_metadata.json"
done

# Promote only after both native measurements and both captures completed.
for operator in "${OPERATORS[@]}"; do
  capture="$WORK_ROOT/capture/${operator//_/-}"
  final="$EVIDENCE_ROOT/$operator"
  mkdir -p "$final/traces" "$NATIVE_ROOT"
  cp "$capture/operator_metadata.json" "$final/operator_metadata.json"
  cp "$capture/traces/kernelslist.g" "$final/traces/kernelslist.g"
  mapfile -t traces < <(find "$capture/traces" -maxdepth 1 -type f -name '*.tracez' -print)
  [[ ${#traces[@]} -eq 1 ]] || {
    echo "expected one compressed Trace for $operator" >&2
    exit 5
  }
  cp "${traces[0]}" "$final/traces/$(basename "${traces[0]}")"
  cp "$WORK_ROOT/native/$operator.json" "$NATIVE_ROOT/$operator.json"
  "$PYTHON" scripts/build_gpu_operator_artifact.py \
    --metadata "$final/operator_metadata.json" \
    --kernels-list "$final/traces/kernelslist.g" \
    --output "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86.json" \
    --trace-manifest-output \
    "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_trace.json"
done

"$PYTHON" - \
  "$EVIDENCE_ROOT" "$MANIFEST_ROOT" "$NATIVE_ROOT" "$BINARY" \
  "$BINARY_SHA256" "$TRACER_SHA256" "$GPU_NAME" "$GPU_CC" \
  "$GPU_DRIVER" "$GPU_MEMORY" "$RECORD" "${CUBINS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd().resolve()
evidence_root = Path(sys.argv[1]).resolve()
manifest_root = Path(sys.argv[2]).resolve()
native_root = Path(sys.argv[3]).resolve()
binary = Path(sys.argv[4]).resolve()
binary_sha256 = sys.argv[5]
tracer_sha256 = sys.argv[6]
gpu_name, gpu_cc, gpu_driver, gpu_memory = sys.argv[7:11]
record = Path(sys.argv[11]).resolve()
cubins = sys.argv[12:]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


payload = json.loads(record.read_text()) if record.is_file() else {}
operators = payload.setdefault("operators", {})
native_measurements = payload.setdefault("native_measurements", {})
for operator in ("token_embedding", "residual_add"):
    evidence = evidence_root / operator
    kernels = evidence / "traces/kernelslist.g"
    trace_names = [line.strip() for line in kernels.read_text().splitlines() if line.strip()]
    if len(trace_names) != 1:
        raise SystemExit(f"expected one Trace for {operator}")
    trace = (kernels.parent / trace_names[0]).resolve()
    manifest = manifest_root / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json"
    native = native_root / f"{operator}.json"
    operators[operator] = {
        "metadata_sha256": digest(evidence / "operator_metadata.json"),
        "kernels_list_sha256": digest(kernels),
        "trace_sha256": digest(trace),
        "trace_manifest_sha256": digest(manifest),
    }
    native_measurements[operator] = {
        "path": relative(native),
        "sha256": digest(native),
    }
payload.update({
    "schema_version": "hetero-p17-sm86-sealed-recapture/v1",
    "capture_host": {
        "gpu": gpu_name,
        "compute_capability": gpu_cc,
        "driver_version": gpu_driver,
        "memory": gpu_memory,
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
        "instrumentation_arch": "sm_86",
        "sha256": tracer_sha256,
    },
    "operators": operators,
    "native_measurements": native_measurements,
    "claim_boundary": {
        "trace_target_sm": 86,
        "physical_capture_gpu_is_not_the_simulated_gpu": False,
        "native_rtx3070_binary_identity_verified": True,
        "performance_pairing_allowed": False,
        "same_program_operator_count": len(operators),
        "reason": "same-Binary is verified; pairing still requires native-VRAM Accel-Sim qualification and error audit",
    },
})
record.parent.mkdir(parents=True, exist_ok=True)
record.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
print(f"local RTX 3070 sealed record written: {record}")
PY

"$PYTHON" scripts/build_p17_execution_identity_catalog.py \
  --recapture-record "$RECORD" \
  --sealed-manifest-root "$MANIFEST_ROOT" \
  --portable-evidence-root "$EVIDENCE_ROOT" \
  --output "$IDENTITY_CATALOG"
"$PYTHON" scripts/update_p17_native_simple_operator_catalog.py \
  --catalog "$NATIVE_CATALOG" \
  --execution-identity-catalog "$IDENTITY_CATALOG" \
  --output "$NATIVE_CATALOG"

P17_OPERATOR_FILTER=token_embedding,residual_add \
P17_EXPECTED_OPERATOR_COUNT=2 \
P17_FINALIZE_CATALOG=0 \
P17_NATIVE_VRAM_ROOT="$QUALIFICATION_STAGE" \
P17_TRACE_MANIFEST_OVERRIDES="$TRACE_OVERRIDES" \
HETEROSIM_PYTHON="$PYTHON" \
ACCEL_SIM_ROOT="$ACCEL_ROOT" \
bash scripts/run_p17_native_vram_accelsim_qualification.sh

for operator in "${OPERATORS[@]}"; do
  source="$QUALIFICATION_STAGE/${operator//_/-}-native-vram/qualification_record.json"
  destination="$QUALIFICATION_ROOT/${operator//_/-}-native-vram/qualification_record.json"
  [[ -s "$source" ]] || { echo "qualification record is absent: $source" >&2; exit 6; }
  mkdir -p "$(dirname "$destination")"
  cp "$source" "$destination"
done

"$PYTHON" scripts/build_p17_gpu_simulator_catalog.py \
  --qualification-root "$QUALIFICATION_ROOT" \
  --execution-identity-catalog "$IDENTITY_CATALOG" \
  --output "$SIMULATOR_CATALOG"
"$PYTHON" scripts/audit_p17_gpu_operator_pairing.py \
  --native "$NATIVE_CATALOG" \
  --simulator "$SIMULATOR_CATALOG" \
  --output "$PAIRING_AUDIT"

"$PYTHON" - \
  "$IMPORT_MANIFEST" "$QUALIFICATION_ROOT" "$SIMULATOR_CATALOG" \
  "$IDENTITY_CATALOG" "$PAIRING_AUDIT" "$MEASUREMENT_MANIFEST" \
  "$NATIVE_CATALOG" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
qualification_root = Path(sys.argv[2])
simulator_catalog = Path(sys.argv[3])
identity_catalog = Path(sys.argv[4])
pairing_audit = Path(sys.argv[5])
measurement_manifest = Path(sys.argv[6])
native_catalog = Path(sys.argv[7])
payload = json.loads(manifest_path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for operator in ("token_embedding", "residual_add"):
    record = qualification_root / f"{operator.replace('_', '-')}-native-vram/qualification_record.json"
    payload["records"][operator] = digest(record)
identity = json.loads(identity_catalog.read_text())
audit = json.loads(pairing_audit.read_text())
payload["source"].update({
    "host": "local-wsl-rtx3070",
    "repository_state": "working_tree_same_binary_recapture",
})
payload["simulator_catalog"]["sha256"] = digest(simulator_catalog)
payload["execution_identity_catalog"].update({
    "sha256": digest(identity_catalog),
    "trace_observed_operator_count": sum(
        item["execution_identity"]["trace_capture_observed"]
        for item in identity["operators"]
    ),
    "native_observed_operator_count": sum(
        item["execution_identity"]["native_measurement_observed"]
        for item in identity["operators"]
    ),
})
payload["pairing_audit"].update({
    "sha256": digest(pairing_audit),
    "topology_match": audit["topology_match"],
    "paired_operator_count": audit["paired_operator_count"],
    "performance_claim_allowed": audit["performance_claim_allowed"],
})
manifest_path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
print(f"P17 import manifest updated: {manifest_path}")

measurement = json.loads(measurement_manifest.read_text())
measurement.update({
    "native_catalog_sha256": digest(native_catalog),
    "execution_identity_catalog_sha256": digest(identity_catalog),
    "simulator_catalog_sha256": digest(simulator_catalog),
    "pairing_audit_sha256": digest(pairing_audit),
    "simulator_memory_topology": "gpu_local_vram",
    "same_binary_native_operator_count": sum(
        item["execution_identity"]["native_measurement_observed"]
        and item["execution_identity"]["trace_capture_observed"]
        for item in identity["operators"]
    ),
    "performance_eligible": audit["performance_claim_allowed"],
    "blocking_reason": (
        "12 operators lack same-Binary native/Trace identity and "
        "10 operators exceed relative-error tolerance"
    ),
})
measurement_manifest.write_bytes(
    (json.dumps(measurement, indent=2, sort_keys=True) + "\n").encode()
)
print(f"P17 measurement manifest updated: {measurement_manifest}")
PY

echo "P17 local RTX 3070 same-Binary simple-operator pairing complete"
