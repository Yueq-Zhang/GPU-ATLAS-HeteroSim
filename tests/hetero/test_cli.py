from pathlib import Path
import re

from frontend.hetero import __version__
from frontend.hetero.cli import main
from frontend.hetero.trace_manifest import TraceManifest
import json


def test_package_version_matches_project_metadata() -> None:
    project = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^version = "([^"]+)"$', project, flags=re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)


def test_validate_command(capsys) -> None:
    result = main(
        [
            "validate",
            "--config",
            "configs/hetero/experiments/m0_smoke.yaml",
        ]
    )
    assert result == 0
    assert "simulation_input_key=" in capsys.readouterr().out


def test_run_command_creates_required_artifacts(tmp_path, capsys) -> None:
    result = main(
        [
            "run",
            "--config",
            "configs/hetero/experiments/m0_smoke.yaml",
            "--runs-root",
            str(tmp_path),
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "scheduler-validation run completed" in output
    run_dirs = list(tmp_path.glob("m0_smoke/*"))
    assert len(run_dirs) == 1
    required = {
        "resolved_config.yaml",
        "dependency_lock.yaml",
        "provenance.json",
        "model_graph.json",
        "execution_graph.json",
        "buffer_bindings.json",
        "trace_manifest.json",
        "metrics.json",
        "event_log.jsonl",
    }
    assert required <= {path.name for path in run_dirs[0].iterdir()}
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
    assert metrics["run_status"] == "scheduler_validation"
    assert metrics["performance_claim_allowed"] is False
    trace_manifest = TraceManifest.load(run_dirs[0] / "trace_manifest.json")
    assert trace_manifest.trace_semantics == "none"
    assert trace_manifest.kernels_list is None
