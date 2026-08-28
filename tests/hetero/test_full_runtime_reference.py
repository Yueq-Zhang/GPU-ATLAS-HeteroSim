import json
from pathlib import Path

import pytest

from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config


@pytest.mark.parametrize(
    "config_name",
    [
        "m8_model1_full_runtime_reference.json",
        "m8_model2_full_runtime_reference.json",
        "m8_model3_full_runtime_reference.json",
        "m8_model4_full_runtime_reference.json",
    ],
)
def test_full_runtime_reference_runs_all_profiles(
    tmp_path: Path, config_name: str
) -> None:
    config = load_and_validate_config(
        f"configs/hetero/experiments/{config_name}"
    )
    run_dir = execute_run(config, Path.cwd(), tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    graph = json.loads((run_dir / "execution_graph.json").read_text())
    residency = json.loads((run_dir / "residency.json").read_text())
    batch = json.loads((run_dir / "batch_plan.json").read_text())
    lifecycle = json.loads((run_dir / "memory_lifecycle.json").read_text())
    links = json.loads((run_dir / "link_statistics.json").read_text())

    assert metrics["run_status"] == "full_runtime_reference"
    assert metrics["performance_claim_allowed"] is False
    assert metrics["implementation_status"] == "implemented_unqualified"
    assert len(batch["epochs"]) > 1
    assert batch["device_subbatches"]
    assert lifecycle["memory_spaces"][0]["peak_bytes"] > 0
    assert links["submitted_payload_bytes"] == links["completed_payload_bytes"]
    assert links["coupling_iterations"] >= 1
    assert graph["placement_contract"]["each_logical_node_exactly_once"] is True
    assert graph["placement_contract"]["logical_node_count"] == len(graph["tasks"])
    assert all("value_version" in route for route in graph["routes"])
    assert residency["schema_version"] == "hetero-residency/v2"


def test_model3_has_one_shared_memory_owner_and_conservation(tmp_path: Path) -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model3_full_runtime_reference.json"
    )
    run_dir = execute_run(config, Path.cwd(), tmp_path)
    memory = json.loads((run_dir / "memory_statistics.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())
    graph = json.loads((run_dir / "execution_graph.json").read_text())
    residency = json.loads((run_dir / "residency.json").read_text())

    assert provenance["memory_timing_owner"] == "shared3d.memory_service"
    assert memory["timing_owner"] == "shared3d.memory_service"
    assert memory["parent_requests_submitted"] == memory["parent_requests_completed"]
    assert memory["child_requests_submitted"] == memory["child_requests_completed"]
    assert memory["submitted_bytes"] == memory["completed_bytes"]
    assert set(memory["requests_by_initiator"]) == {"gpu0", "atlas0.compute"}
    assert memory["coupling_iterations"] >= 2
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert memory["last_completion_fs"] <= metrics["makespan_fs"]
    assert len(graph["tasks"]) == 228
    assert len(graph["routes"]) == 28
    assert len(residency["events"]) == 571


def test_model3_gpu_only_disables_logic_die_and_has_no_cross_device_routes(
    tmp_path: Path,
) -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model3_gpu_only_no_logic_die_reference.json"
    )
    run_dir = execute_run(config, Path.cwd(), tmp_path)
    graph = json.loads((run_dir / "execution_graph.json").read_text())
    memory = json.loads((run_dir / "memory_statistics.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert {task["device_id"] for task in graph["tasks"]} == {"gpu0"}
    assert graph["routes"] == []
    assert memory["access_mode"] == "gpu_only"
    assert set(memory["requests_by_initiator"]) == {"gpu0"}
    assert memory["parent_requests_submitted"] == memory["parent_requests_completed"]
    assert memory["submitted_bytes"] == memory["completed_bytes"]
    competition = metrics["gpu_logic_die_competition"]
    assert competition["mode"] == "gpu_only"
    assert competition["enabled"] is False
    assert competition["gpu_tasks"] == len(graph["tasks"])
    assert competition["logic_die_tasks"] == 0
    assert competition["gpu_memory_requests"] > 0
    assert competition["logic_die_memory_requests"] == 0


def test_full_runtime_does_not_silently_replace_ramulator2(tmp_path: Path) -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model3_full_runtime_reference.json"
    )
    service = config["system"]["memory_services"]["shared0.dram3d"]
    service.clear()
    service.update(
        {
            "kind": "ramulator2",
            "timing_owner": "shared3d.memory_service",
            "config_ref": "configs/hetero/backends/example_ramulator2.json",
            "parameter_source": "test",
        }
    )
    with pytest.raises(ValueError, match="live Ramulator2 coupling is not qualified"):
        execute_run(config, Path.cwd(), tmp_path)
