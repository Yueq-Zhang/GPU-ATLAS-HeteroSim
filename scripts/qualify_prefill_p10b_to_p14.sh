#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/opt/gpu-atlas/GPU-ATLAS-HeteroSim}"
OUTPUT_ROOT="${2:-/opt/gpu-atlas/qualification/prefill-p10b-to-p14}"
PYTHON="${HETEROSIM_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"

CONFIGS=(
  configs/hetero/experiments/p10b_b_tinyllama_prefill_1layer_mixed_live_ramulator2.json
  configs/hetero/experiments/p12_tinyllama_prefill_1layer_gpu_live_ramulator2.json
  configs/hetero/experiments/p13_tinyllama_prefill_22layer_ctx16_gpu_live_ramulator2.json
  configs/hetero/experiments/p14_tinyllama_prefill_bs1_ctx1024_gpu_live_ramulator2.json
)

for config in "${CONFIGS[@]}"; do
  "$PYTHON" -m frontend.hetero.cli validate --config "$config"
  name="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"]["name"])' "$config")"
  key="$($PYTHON -c 'import sys; from frontend.hetero.runner import simulation_input_key; from frontend.hetero.schema import load_and_validate_config; print(simulation_input_key(load_and_validate_config(sys.argv[1])))' "$config")"
  for run in run1 run2; do
    "$PYTHON" -m frontend.hetero.cli run \
      --config "$config" \
      --runs-root "$OUTPUT_ROOT/$run"
  done
  left="$OUTPUT_ROOT/run1/$name/$key"
  right="$OUTPUT_ROOT/run2/$name/$key"
  for artifact in \
    metrics.json \
    memory_statistics.json \
    request_cycle_trace.json \
    global_memory_map.json \
    prefill_artifact_coverage.json \
    execution_graph.json \
    residency.json; do
    cmp "$left/$artifact" "$right/$artifact"
  done
  "$PYTHON" - "$left" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
metrics = json.loads((root / "metrics.json").read_text())
memory = json.loads((root / "memory_statistics.json").read_text())
coverage = json.loads((root / "prefill_artifact_coverage.json").read_text())
execution = json.loads((root / "execution_graph.json").read_text())
address_map = json.loads((root / "global_memory_map.json").read_text())
assert metrics["performance_claim_allowed"] is False
assert memory["instances"] == 1
assert memory["accepted_parent_ids"] == memory["completed"]
assert memory["accepted_parent_ids"] == memory["observed_completion_ids"]
assert memory["outstanding"] == 0
assert coverage["all_tasks_covered"] is True
assert coverage["analytical_fallback_tasks"] == 0
assert coverage["covered_tasks"] == len(execution["tasks"])
assert address_map["non_overlapping"] is True
assert address_map["allocated_bytes"] <= address_map["capacity_bytes"]
if metrics.get("gpu_logic_die_competition", {}).get("mode") == "gpu_only":
    assert memory["initiators"]["atlas0.compute"]["parents"] == 0
print(f"qualified {root}")
PY
done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
artifacts = [
    "metrics.json",
    "memory_statistics.json",
    "request_cycle_trace.json",
    "global_memory_map.json",
    "prefill_artifact_coverage.json",
    "execution_graph.json",
    "residency.json",
]
stages = []
for experiment in sorted((root / "run1").iterdir()):
    run1 = next(experiment.iterdir())
    run2 = root / "run2" / experiment.name / run1.name
    metrics = json.loads((run1 / "metrics.json").read_text())
    memory = json.loads((run1 / "memory_statistics.json").read_text())
    execution = json.loads((run1 / "execution_graph.json").read_text())
    address_map = json.loads((run1 / "global_memory_map.json").read_text())
    hashes = {
        name: hashlib.sha256((run1 / name).read_bytes()).hexdigest()
        for name in artifacts
    }
    assert all((run1 / name).read_bytes() == (run2 / name).read_bytes() for name in artifacts)
    stages.append(
        {
            "experiment": experiment.name,
            "simulation_input_key": run1.name,
            "device_tasks": len(execution["tasks"]),
            "routes": len(execution["routes"]),
            "makespan_fs_unqualified": metrics["makespan_fs"],
            "accepted_parents": memory["accepted_parent_ids"],
            "completed_parents": memory["completed"],
            "outstanding": memory["outstanding"],
            "gpu_parents": memory["initiators"]["gpu0"]["parents"],
            "atlas_parents": memory["initiators"]["atlas0.compute"]["parents"],
            "allocated_global_pa_bytes": address_map["allocated_bytes"],
            "allocation_count": address_map["allocation_count"],
            "run1_run2_byte_identical": True,
            "artifact_sha256": hashes,
        }
    )
record = {
    "schema_version": "hetero-prefill-qualification/v1",
    "date": "2026-08-28",
    "stages": stages,
    "invariants": {
        "one_live_ramulator2": True,
        "all_parents_completed": True,
        "zero_outstanding": True,
        "full_cycle_contract_coverage": True,
        "analytical_fallback_tasks": 0,
        "double_run_byte_identical": True,
    },
    "fidelity": {
        "compute": "tiled_cycle_contract_unqualified",
        "memory": "live_ramulator2_sampled_requests",
        "trace_coverage": 0.0,
        "performance_eligible": False,
    },
}
(root / "qualification_record.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n"
)
PY

echo "P10b-B through P14 Prefill qualification passed: $OUTPUT_ROOT"
