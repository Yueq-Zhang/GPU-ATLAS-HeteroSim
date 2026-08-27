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


def test_opt67b_rtx3070_prefill_roofline_scope_and_result(tmp_path: Path) -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_opt67b_rtx3070_prefill_roofline.json"
    )
    run_dir = execute_run(config, Path.cwd(), tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    graphs = json.loads((run_dir / "model_graph.json").read_text())
    counters = graphs["requests"][0]["counters"]

    assert counters["prefill_forwards"] == 1
    assert counters["decode_forwards"] == 0
    assert counters["final_committed_kv_len"] == 1024
    assert 170_000_000_000_000 <= metrics["requests"][0]["ttft_fs"] <= 190_000_000_000_000
    assert metrics["performance_claim_allowed"] is False


def test_opt67b_single_decode_gpu_vs_3ddram_comparison(tmp_path: Path) -> None:
    experiments = {
        "rtx3070": "m8_opt67b_rtx3070_decode_roofline.json",
        "rtx4090": "m8_opt67b_rtx4090_roofline.json",
        "atlas": "m8_opt67b_atlas_3ddram_decode_roofline.json",
    }
    latency_fs = {}
    for name, filename in experiments.items():
        config = load_and_validate_config(
            f"configs/hetero/experiments/{filename}"
        )
        run_dir = execute_run(config, Path.cwd(), tmp_path / name)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        graphs = json.loads((run_dir / "model_graph.json").read_text())
        counters = graphs["requests"][0]["counters"]
        assert counters["prefill_forwards"] == 0
        assert counters["decode_forwards"] == 1
        assert counters["final_committed_kv_len"] == 1025
        assert metrics["performance_claim_allowed"] is False
        latency_fs[name] = metrics["requests"][0]["e2e_user_fs"]

    assert latency_fs == {
        "rtx3070": 30_898_972_303_758,
        "rtx4090": 13_732_876_579_628,
        "atlas": 33_795_750_957_034,
    }
    assert latency_fs["rtx3070"] < latency_fs["atlas"]
    assert latency_fs["rtx4090"] < latency_fs["atlas"]
