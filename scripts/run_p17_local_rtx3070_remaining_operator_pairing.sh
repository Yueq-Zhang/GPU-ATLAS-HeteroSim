#!/usr/bin/env bash
set -euo pipefail

PYTHON="${HETEROSIM_PYTHON:-/opt/conda/envs/qserve-local/bin/python}"
CUDA_ROOT="${ACCEL_SIM_CUDA_ROOT:-/usr/local/cuda-11.8}"
ACCEL_ROOT="${ACCEL_SIM_ROOT:-/opt/gpu-atlas/dependencies/accel-sim-framework-64653015f85fb5664c84a10f48527e8897d289d0}"
MODEL_ROOT="${TINYLLAMA_MODEL_ROOT:-/opt/hf-cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}"
CAPTURE_ROOT="${P17_REMAINING_CAPTURE_ROOT:-/opt/gpu-atlas/qualification/p17-local-remaining-same-binary-v1}"
EVIDENCE_ROOT="${P17_LOCAL_EVIDENCE_ROOT:-configs/hetero/operator_artifacts/p17_sealed/evidence/local_rtx3070_same_binary_v1}"
MANIFEST_ROOT="${P17_SEALED_MANIFEST_ROOT:-configs/hetero/operator_artifacts/p17_sealed}"
NATIVE_ROOT="${P17_LOCAL_NATIVE_ROOT:-validation/p17/native_rtx3070/sealed_same_binary_v1}"
RECAPTURE_RECORD="${P17_RECAPTURE_RECORD:-validation/p17/sm86_sealed_recapture/recapture_record.json}"
IDENTITY_CATALOG="${P17_EXECUTION_IDENTITY_CATALOG:-validation/p17/sm86_sealed_recapture/execution_identity_catalog.json}"
NATIVE_CATALOG="${P17_NATIVE_CATALOG:-validation/p17/gpu_operator_pairing/native_rtx3070_local_vram.json}"
TRACE_OVERRIDES="${P17_TRACE_MANIFEST_OVERRIDES:-configs/hetero/calibration/p17_native_vram_trace_overrides.json}"
QUALIFICATION_STAGE="${P17_REMAINING_QUALIFICATION_STAGE:-/opt/gpu-atlas/qualification/p17-local-remaining-native-vram-v1}"
QUALIFICATION_ROOT="${P17_NATIVE_VRAM_ROOT:-validation/p17/native_vram_qualification}"
SIMULATOR_CATALOG="${P17_SIMULATOR_CATALOG:-validation/p17/gpu_operator_pairing/simulator_native_vram.json}"
PAIRING_AUDIT="${P17_PAIRING_AUDIT:-validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json}"
IMPORT_MANIFEST="${P17_IMPORT_MANIFEST:-validation/p17/native_vram_qualification/import_manifest.json}"
MEASUREMENT_MANIFEST="${P17_MEASUREMENT_MANIFEST:-validation/p17/gpu_operator_pairing/measurement_manifest.json}"
CALIBRATION_RECORD="${P17_CALIBRATION_RECORD:-configs/hetero/calibration/p17_tinyllama_prefill_layer0_ctx16_incomplete.json}"
WORKLOAD="workloads/python/tinyllama_prefill_operator.py"
OPERATORS=(
  attention_norm
  qkv_projection
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

for tool in "$PYTHON" "$CUDA_ROOT/bin/nvcc"; do
  [[ -x "$tool" ]] || { echo "required executable is absent: $tool" >&2; exit 2; }
done
for path in "$ACCEL_ROOT" "$MODEL_ROOT" "$WORKLOAD" "$RECAPTURE_RECORD"; do
  [[ -e "$path" ]] || { echo "required P17 input is absent: $path" >&2; exit 2; }
done

GPU_ROW=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | head -n 1)
IFS=',' read -r GPU_NAME GPU_CC GPU_DRIVER GPU_MEMORY <<<"$GPU_ROW"
GPU_NAME=$(echo "$GPU_NAME" | xargs)
GPU_CC=$(echo "$GPU_CC" | xargs)
GPU_DRIVER=$(echo "$GPU_DRIVER" | xargs)
GPU_MEMORY=$(echo "$GPU_MEMORY" | xargs)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 3070" || "$GPU_CC" != "8.6" ]]; then
  echo "P17 remaining-operator pairing requires RTX 3070 compute capability 8.6" >&2
  exit 3
fi

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export HETEROSIM_PYTHON="$PYTHON"
export TINYLLAMA_PYTHON="$PYTHON"
export TINYLLAMA_MODEL_ROOT="$MODEL_ROOT"
export ACCEL_SIM_CUDA_ROOT="$CUDA_ROOT"
export ACCEL_SIM_ROOT="$ACCEL_ROOT"
mkdir -p "$CAPTURE_ROOT" "$EVIDENCE_ROOT" "$NATIVE_ROOT" "$QUALIFICATION_STAGE"

for operator in "${OPERATORS[@]}"; do
  native="$NATIVE_ROOT/$operator.json"
  if [[ ! -s "$native" ]]; then
    echo "P17 local native measurement: $operator"
    "$PYTHON" "$WORKLOAD" \
      --model "$MODEL_ROOT" \
      --operator "$operator" \
      --context 16 \
      --batch-size 1 \
      --warmup 50 \
      --iterations 500 \
      --native-measurement-output "$native"
  else
    echo "P17 resume: native measurement exists for $operator"
  fi
  "$PYTHON" - "$native" "$operator" "$PYTHON" "$WORKLOAD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

native, operator, python_path, workload_path = map(Path, sys.argv[1:])
operator = operator.as_posix()
payload = json.loads(native.read_text(encoding="utf-8"))
launch = payload.get("launch", {})
components = launch.get("program_components", {})


def digest(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


if (
    payload.get("schema_version")
    != "hetero-p17-sealed-native-pytorch-operator/v1"
    or payload.get("operator_type") != operator
    or payload.get("checkpoint_revision")
    != "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    or payload.get("batch_size") != 1
    or payload.get("context_length") != 16
    or payload.get("dtype") != "fp16"
    or payload.get("device", {}).get("name") != "NVIDIA GeForce RTX 3070"
    or payload.get("device", {}).get("compute_capability") != "8.6"
    or payload.get("protocol", {}).get("warmup_iterations") != 50
    or payload.get("protocol", {}).get("measured_iterations") != 500
    or payload.get("protocol", {}).get("timer") != "cuda_event_per_iteration"
    or launch.get("program_kind") != "python_pytorch_launch_program"
    or launch.get("target_sm") != 86
    or digest(python_path) != components.get("python_executable_sha256")
    or digest(workload_path) != components.get("workload_source_sha256")
):
    raise SystemExit(f"invalid native same-program measurement: {native}")
PY

  capture="$CAPTURE_ROOT/tinyllama-prefill-bs1-ctx16-${operator//_/-}"
  evidence="$EVIDENCE_ROOT/$operator"
  kernels="$evidence/traces/kernelslist.g"
  if [[ ! -s "$evidence/operator_metadata.json" || ! -s "$kernels" ]]; then
    if [[ -d "$capture" ]] && find "$capture" -mindepth 1 -print -quit | grep -q .; then
      if [[ ! -s "$capture/operator_metadata.json" || ! -s "$capture/traces/kernelslist.g" ]]; then
        echo "incomplete capture requires inspection before resume: $capture" >&2
        exit 4
      fi
      echo "P17 resume: promoting completed capture for $operator"
    else
      echo "P17 local NVBit capture: $operator"
      HETEROSIM_CAPTURE_RANGE=process \
      HETEROSIM_CAPTURE_WARMUP=0 \
      HETEROSIM_CAPTURE_ALLOCATOR_HISTORY=1 \
      bash scripts/capture_tinyllama_prefill_operator.sh \
        "$operator" 16 "$CAPTURE_ROOT" "$CAPTURE_ROOT/staging-artifacts"
    fi
    "$PYTHON" - "$capture" "$evidence" <<'PY'
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
kernels = source / "traces/kernelslist.g"
entries = [line.strip() for line in kernels.read_text().splitlines() if line.strip()]
if not entries or any(Path(entry).is_absolute() or ".." in Path(entry).parts for entry in entries):
    raise SystemExit(f"unsafe or empty kernels list: {kernels}")
if destination.exists() and any(destination.iterdir()):
    raise SystemExit(f"refusing to overwrite partial evidence: {destination}")
(destination / "traces").mkdir(parents=True, exist_ok=True)
shutil.copy2(source / "operator_metadata.json", destination / "operator_metadata.json")
shutil.copy2(kernels, destination / "traces/kernelslist.g")
for entry in entries:
    trace = source / "traces" / entry
    if not trace.is_file() or trace.stat().st_size == 0:
        raise SystemExit(f"captured Trace is absent: {trace}")
    target = destination / "traces" / entry
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trace, target)
PY
  else
    echo "P17 resume: repository evidence exists for $operator"
  fi

  "$PYTHON" scripts/build_gpu_operator_artifact.py \
    --metadata "$evidence/operator_metadata.json" \
    --kernels-list "$kernels" \
    --output "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86.json" \
    --trace-manifest-output \
    "$MANIFEST_ROOT/tinyllama_prefill_bs1_ctx16_${operator}_sm86_trace.json"
done

"$PYTHON" - \
  "$RECAPTURE_RECORD" "$EVIDENCE_ROOT" "$MANIFEST_ROOT" "$NATIVE_ROOT" \
  "$GPU_NAME" "$GPU_CC" "$GPU_DRIVER" "$GPU_MEMORY" "${OPERATORS[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd().resolve()
record_path = Path(sys.argv[1]).resolve()
evidence_root = Path(sys.argv[2]).resolve()
manifest_root = Path(sys.argv[3]).resolve()
native_root = Path(sys.argv[4]).resolve()
gpu_name, gpu_cc, gpu_driver, gpu_memory = sys.argv[5:9]
operators = sys.argv[9:]
payload = json.loads(record_path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


for operator in operators:
    evidence = evidence_root / operator
    kernels = evidence / "traces/kernelslist.g"
    entries = [line.strip() for line in kernels.read_text().splitlines() if line.strip()]
    trace_files = []
    for entry in entries:
        trace = (kernels.parent / entry).resolve()
        trace_files.append({"name": entry, "sha256": digest(trace)})
    manifest = manifest_root / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json"
    native = native_root / f"{operator}.json"
    native_payload = json.loads(native.read_text(encoding="utf-8"))
    launch = native_payload["launch"]
    payload["operators"][operator] = {
        "metadata_sha256": digest(evidence / "operator_metadata.json"),
        "kernels_list_sha256": digest(kernels),
        "trace_files": trace_files,
        "trace_manifest_sha256": digest(manifest),
        "target_binary": {
            "kind": "python_pytorch_launch_program",
            "path": launch["program_components"]["python_executable"],
            "sha256": launch["launch_program_sha256"],
            "components": launch["program_components"],
        },
    }
    payload.setdefault("native_measurements", {})[operator] = {
        "path": relative(native),
        "sha256": digest(native),
    }
payload["capture_host"] = {
    "gpu": gpu_name,
    "compute_capability": gpu_cc,
    "driver_version": gpu_driver,
    "memory": gpu_memory,
}
payload["claim_boundary"] = {
    "trace_target_sm": 86,
    "physical_capture_gpu_is_not_the_simulated_gpu": False,
    "native_rtx3070_binary_identity_verified": True,
    "same_program_operator_count": len(payload["operators"]),
    "performance_pairing_allowed": False,
    "reason": "all operators have same-program Native/Trace evidence; error audit remains mandatory",
}
record_path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
print(f"P17 full recapture record written: {record_path}")
PY

"$PYTHON" scripts/build_p17_execution_identity_catalog.py \
  --recapture-record "$RECAPTURE_RECORD" \
  --sealed-manifest-root "$MANIFEST_ROOT" \
  --portable-evidence-root "$EVIDENCE_ROOT" \
  --output "$IDENTITY_CATALOG"
"$PYTHON" scripts/update_p17_native_simple_operator_catalog.py \
  --catalog "$NATIVE_CATALOG" \
  --execution-identity-catalog "$IDENTITY_CATALOG" \
  --output "$NATIVE_CATALOG"

"$PYTHON" - "$TRACE_OVERRIDES" "$MANIFEST_ROOT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
manifest_root = Path(sys.argv[2])
operators = (
    "token_embedding", "attention_norm", "qkv_projection", "rope",
    "causal_attention", "output_projection", "mlp_norm", "gate_up_projection",
    "silu_multiply", "down_projection", "residual_add", "final_norm",
    "lm_head", "sampling",
)
payload = {
    operator: (manifest_root / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json").as_posix()
    for operator in operators
}
if not all(Path(path).is_file() for path in payload.values()):
    raise SystemExit("P17 full Trace override set is incomplete")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

operator_filter=$(IFS=,; echo "${OPERATORS[*]}")
P17_OPERATOR_FILTER="$operator_filter" \
P17_EXPECTED_OPERATOR_COUNT=12 \
P17_FINALIZE_CATALOG=0 \
P17_NATIVE_VRAM_ROOT="$QUALIFICATION_STAGE" \
P17_TRACE_MANIFEST_OVERRIDES="$TRACE_OVERRIDES" \
P17_EXECUTION_IDENTITY_CATALOG="$IDENTITY_CATALOG" \
bash scripts/run_p17_native_vram_accelsim_qualification.sh

for operator in "${OPERATORS[@]}"; do
  source="$QUALIFICATION_STAGE/${operator//_/-}-native-vram/qualification_record.json"
  destination="$QUALIFICATION_ROOT/${operator//_/-}-native-vram/qualification_record.json"
  [[ -s "$source" ]] || { echo "qualification record is absent: $source" >&2; exit 5; }
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
  "$NATIVE_CATALOG" "$CALIBRATION_RECORD" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import_manifest = Path(sys.argv[1])
qualification_root = Path(sys.argv[2])
simulator_catalog = Path(sys.argv[3])
identity_catalog = Path(sys.argv[4])
pairing_audit = Path(sys.argv[5])
measurement_manifest = Path(sys.argv[6])
native_catalog = Path(sys.argv[7])
calibration_record = Path(sys.argv[8])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


identity = json.loads(identity_catalog.read_text())
audit = json.loads(pairing_audit.read_text())
operators = [item["operator_type"] for item in identity["operators"]]
if len(operators) != 14:
    raise SystemExit("P17 finalization requires fourteen execution identities")

manifest = json.loads(import_manifest.read_text())
manifest["qualified_operator_count"] = 14
manifest["records"] = {}
for operator in operators:
    record = qualification_root / f"{operator.replace('_', '-')}-native-vram/qualification_record.json"
    manifest["records"][operator] = digest(record)
manifest["source"].update({
    "host": "local-wsl-rtx3070",
    "repository_state": "working_tree_full_same_program_recapture",
})
manifest["simulator_catalog"]["sha256"] = digest(simulator_catalog)
manifest["execution_identity_catalog"].update({
    "sha256": digest(identity_catalog),
    "trace_observed_operator_count": 14,
    "native_observed_operator_count": 14,
})
manifest["pairing_audit"].update({
    "sha256": digest(pairing_audit),
    "topology_match": audit["topology_match"],
    "paired_operator_count": audit["paired_operator_count"],
    "performance_claim_allowed": audit["performance_claim_allowed"],
})
import_manifest.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())

identity_blockers = sum("execution_identity" in item for item in audit["blockers"])
error_blockers = sum("relative_error" in item for item in audit["blockers"])
measurement = json.loads(measurement_manifest.read_text())
measurement.update({
    "native_catalog_sha256": digest(native_catalog),
    "execution_identity_catalog_sha256": digest(identity_catalog),
    "simulator_catalog_sha256": digest(simulator_catalog),
    "pairing_audit_sha256": digest(pairing_audit),
    "native_memory_topology": "gpu_local_vram",
    "simulator_memory_topology": "gpu_local_vram",
    "same_binary_native_operator_count": 14,
    "performance_eligible": False,
    "blocking_reason": (
        f"paired={audit['paired_operator_count']}/14; "
        f"identity_blockers={identity_blockers}; error_blockers={error_blockers}; "
        "global six-component calibration remains incomplete"
    ),
})
measurement_manifest.write_bytes((json.dumps(measurement, indent=2, sort_keys=True) + "\n").encode())

calibration = json.loads(calibration_record.read_text())
for source in calibration["components"]["gpu_kernel"]["sources"]:
    if source.get("source_id") == "rtx3070_native_fourteen_operators":
        source["artifact_sha256"] = digest(native_catalog)
        source["description"] = (
            "Fourteen Context-16 RTX 3070 measurements produced by the exact "
            "launch program used for each sealed NVBit Trace"
        )
        break
else:
    raise SystemExit("P17 native calibration source is absent")
calibration["components"]["gpu_kernel"]["notes"] = (
    "All fourteen Native and Trace identities are sealed; numerical error "
    "and the remaining global components still gate performance claims."
)
calibration_record.write_bytes((json.dumps(calibration, indent=2, sort_keys=True) + "\n").encode())
print("P17 full manifests and calibration source refreshed")
PY

P16_KEY=d5066ff9081332bd31ae5699f4f572736cc7f188ae9f4272cf89a4af0a1d6e3a
"$PYTHON" scripts/audit_p17_performance_calibration.py \
  "$CALIBRATION_RECORD" \
  "validation/p16/leg1/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/$P16_KEY" \
  "validation/p16/leg2/p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu/$P16_KEY" \
  --project-root . \
  --output validation/p17/p16_layer0_ctx16/performance_calibration_audit.json

echo "P17 local RTX 3070 remaining-operator same-program pairing complete"
