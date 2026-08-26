import json
from pathlib import Path

import pytest

from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config


@pytest.mark.parametrize(
    ("config_name", "profile", "expected_route_kind"),
    [
        ("m1_model1_atlas_native.json", "model1_atlas_native", "transfer"),
        ("m1_model2_host_memory_pcie.json", "model2_host_memory_pcie", "transfer"),
        ("m1_model3_gpu_native_3ddram.json", "model3_gpu_native_3ddram", "synchronization"),
        ("m1_model4_cxl_memory_tier.json", "model4_cxl_memory_tier", "transfer"),
    ],
)
def test_same_logical_workload_runs_across_four_profiles(
    tmp_path: Path,
    config_name: str,
    profile: str,
    expected_route_kind: str,
) -> None:
    config = load_and_validate_config(f"configs/hetero/experiments/{config_name}")
    assert config["system"]["profile"] == profile
    run_dir = execute_run(config, Path.cwd(), tmp_path)

    graph = json.loads((run_dir / "model_graph.json").read_text())
    counters = graph["requests"][0]["counters"]
    assert counters == {
        "decode_forwards": 2,
        "final_committed_kv_len": 18,
        "kv_append_pairs": 36,
        "kv_range_writes": 72,
        "lm_head": 3,
        "prefill_forwards": 1,
        "sampling": 3,
    }
    execution = json.loads((run_dir / "execution_graph.json").read_text())
    assert execution["routes"]
    assert {route["kind"] for route in execution["routes"]} == {
        expected_route_kind
    }
    assert all(route["dependencies"] for route in execution["routes"])
