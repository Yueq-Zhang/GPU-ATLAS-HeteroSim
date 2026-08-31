import json
from pathlib import Path

import pytest

from scripts.summarize_p15g_prefill_timeline import (
    P15gQualificationError,
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
        "outstanding": 0,
    }


def _run(tmp_path: Path) -> Path:
    norm_id = "task.norm"
    qkv_id = "task.qkv"
    value_id = "R0.norm.out"
    _write(
        tmp_path / "execution_graph.json",
        {
            "tasks": [
                {
                    "task_id": norm_id,
                    "op": "attention_norm",
                    "resource_id": "gpu0",
                    "dependencies": ["task.embedding"],
                    "timing": {"start_time_fs": 10, "completion_time_fs": 20},
                    "compiled_artifact": {"request_cycle_ready": True},
                    "backend_statistics": {"external_memory_stats": _external()},
                    "validated_input_versions": [],
                },
                {
                    "task_id": qkv_id,
                    "op": "qkv_projection",
                    "resource_id": "gpu0",
                    "dependencies": [norm_id],
                    "timing": {"start_time_fs": 20, "completion_time_fs": 40},
                    "compiled_artifact": {"request_cycle_ready": True},
                    "backend_statistics": {"external_memory_stats": _external()},
                    "validated_input_versions": [{"value_id": value_id, "version": 1}],
                },
            ]
        },
    )
    _write(
        tmp_path / "online_dispatch.json",
        {
            "version_commits": [
                {
                    "task_id": norm_id,
                    "value_id": value_id,
                    "version": 1,
                    "commit_time_fs": 20,
                    "cause": "backend_completion",
                },
                {
                    "task_id": qkv_id,
                    "value_id": "R0.qkv.out",
                    "version": 1,
                    "commit_time_fs": 40,
                    "cause": "backend_completion",
                },
            ]
        },
    )
    _write(
        tmp_path / "global_memory_map.json",
        {
            "request_cycle_bindings": [
                {
                    "semantic_bindings": [
                        {
                            "tensor_id": "tinyllama.layer0.attention_norm.output",
                            "value_id": value_id,
                            "global_pa_base": 4096,
                        }
                    ]
                },
                {
                    "semantic_bindings": [
                        {
                            "tensor_id": "tinyllama.layer0.qkv_projection.input",
                            "value_id": value_id,
                            "global_pa_base": 4096,
                        }
                    ]
                },
            ]
        },
    )
    _write(tmp_path / "provenance.json", {"simulator_revision": "abc-dirty"})
    return tmp_path


def test_p15g_summarizer_accepts_complete_causality(tmp_path: Path) -> None:
    summary = summarize(_run(tmp_path))
    assert summary["status"] == "passed"
    assert summary["global_pa_causality"]["global_pa_base"] == 4096
    assert summary["version_causality"]["qkv_validated_attention_output_v1"] is True


def test_p15g_summarizer_rejects_early_qkv_launch(tmp_path: Path) -> None:
    run = _run(tmp_path)
    payload = json.loads((run / "execution_graph.json").read_text())
    payload["tasks"][1]["timing"]["start_time_fs"] = 19
    _write(run / "execution_graph.json", payload)
    with pytest.raises(P15gQualificationError, match="starts before"):
        summarize(run)
