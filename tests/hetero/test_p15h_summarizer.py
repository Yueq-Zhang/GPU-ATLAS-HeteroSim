import json
from pathlib import Path

import pytest

from scripts.summarize_p15h_prefill_timeline import (
    P15hQualificationError,
    READY_OPERATORS,
    summarize,
)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _external() -> dict[str, int]:
    return {
        "instances": 1,
        "gpu_parents": 3,
        "gpu_completed": 3,
        "gpu_children": 4,
        "children_sent": 4,
        "children_completed": 4,
        "durable_completed": 3,
        "address_translated": 8,
        "address_unmapped": 0,
        "atlas_parents": 0,
        "atlas_children": 0,
        "atlas_completed": 0,
        "outstanding": 0,
    }


def _run(tmp_path: Path) -> Path:
    operators = sorted(READY_OPERATORS)
    tasks = []
    commits = []
    ranges = []
    bindings = []
    previous_task = None
    previous_value = "model.input"
    previous_version = 0
    for index, operator in enumerate(operators):
        task_id = f"task.{index}.{operator}"
        output_value = f"value.{index}.{operator}"
        start = index * 20
        completion = start + 10
        tasks.append(
            {
                "task_id": task_id,
                "op": operator,
                "resource_id": "gpu0",
                "dependencies": [previous_task] if previous_task else [],
                "timing": {
                    "start_time_fs": start,
                    "completion_time_fs": completion,
                },
                "compiled_artifact": {"request_cycle_ready": True},
                "backend_statistics": {
                    "cycles": 100 + index,
                    "external_memory_stats": _external(),
                },
                "input_values": [
                    {"value_id": previous_value, "version": previous_version}
                ],
                "validated_input_versions": [
                    {"value_id": previous_value, "version": previous_version}
                ],
            }
        )
        commits.append(
            {
                "task_id": task_id,
                "value_id": output_value,
                "version": 1,
                "commit_time_fs": completion,
                "cause": "backend_completion",
            }
        )
        base = 4096 + index * 4096
        ranges.append(
            {
                "value_id": output_value,
                "base_address": base,
                "end_address_exclusive": base + 1024,
            }
        )
        bindings.append(
            {
                "semantic_bindings": [
                    {
                        "tensor_id": f"tensor.{operator}.output",
                        "value_id": output_value,
                        "global_pa_base": base + 64,
                        "value_offset_bytes": 64,
                    }
                ]
            }
        )
        previous_task = task_id
        previous_value = output_value
        previous_version = 1
    _write(tmp_path / "execution_graph.json", {"tasks": tasks})
    _write(tmp_path / "online_dispatch.json", {"version_commits": commits})
    _write(
        tmp_path / "global_memory_map.json",
        {"ranges": ranges, "request_cycle_bindings": bindings},
    )
    _write(tmp_path / "provenance.json", {"simulator_revision": "abc-dirty"})
    return tmp_path


def test_p15h_summarizer_accepts_twelve_operator_causality(tmp_path: Path) -> None:
    summary = summarize(_run(tmp_path))
    assert summary["status"] == "passed"
    assert summary["ready_operator_count"] == 12
    assert summary["resource_causality"]["all_gpu_tasks_non_overlapping"] is True
    assert summary["global_pa_causality"]["request_binding_count"] == 12
    assert summary["version_causality"]["ready_operator_inputs_validated"] is True


def test_p15h_summarizer_rejects_global_pa_overlap(tmp_path: Path) -> None:
    run = _run(tmp_path)
    payload = json.loads((run / "global_memory_map.json").read_text())
    payload["ranges"][1]["base_address"] = payload["ranges"][0]["base_address"]
    _write(run / "global_memory_map.json", payload)
    with pytest.raises(P15hQualificationError, match="Global PA overlap"):
        summarize(run)
