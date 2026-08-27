import json
from pathlib import Path

from frontend.hetero.dse import enumerate_candidates, run_dse
from frontend.hetero.schema import load_and_validate_config


def test_dse_enumerates_cartesian_product() -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model1_full_runtime_reference.json"
    )
    search = json.loads(
        Path("configs/hetero/dse/tiny_roofline_search.json").read_text()
    )
    candidates = enumerate_candidates(config, search)
    assert len(candidates) == 8
    assert len({candidate["experiment"]["name"] for candidate in candidates}) == 8


def test_dse_report_is_ranked_and_unqualified(tmp_path: Path) -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m2_model1_analytical_preview.json"
    )
    search = {
        "axes": {"backends.gpu.effective_memory_bandwidth_Bps": [500000000000]},
        "max_candidates": 1,
        "objective": "makespan_fs",
    }
    report_path = run_dse(config, search, Path.cwd(), tmp_path)
    report = json.loads(report_path.read_text())
    assert report["candidate_count"] == 1
    assert report["ranking"][0]["makespan_fs"] > 0
    assert report["qualification_status"].startswith("unqualified")
