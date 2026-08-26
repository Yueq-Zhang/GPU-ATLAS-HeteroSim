import json
from pathlib import Path

from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config


def test_analytical_preview_emits_scheduled_dag(tmp_path: Path) -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m2_model1_analytical_preview.json"
    )
    run_dir = execute_run(config, Path.cwd(), tmp_path)
    execution = json.loads((run_dir / "execution_graph.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())

    assert metrics["run_status"] == "analytical_preview"
    assert metrics["performance_claim_allowed"] is False
    assert metrics["makespan_fs"] > 0
    assert metrics["requests"][0]["ttft_fs"] > 0
    assert all(task["duration_fs"] > 0 for task in execution["tasks"])
    assert all(route["payload_bytes"] > 0 for route in execution["routes"])
    assert all("timing" in task for task in execution["tasks"])
    assert provenance["runtime_owner"] == "cpp.GlobalEventRuntime"
