import json
from pathlib import Path

import pytest

from frontend.hetero.p23_request_controls import (
    P23RequestControlError,
    load_p23_service_contract,
    run_kv_allocator_pressure_probe,
    run_p23_request_control_timeline,
)


def _config() -> dict[str, object]:
    return json.loads(
        Path("configs/hetero/experiments/p26_p23_request_control_stream.json").read_text(
            encoding="utf-8"
        )
    )


def test_sealed_p23_contract_is_exact_bs2_kv17() -> None:
    contract = load_p23_service_contract(Path.cwd())
    assert contract.batch_size == 2
    assert contract.initial_kv_length == 16
    assert contract.final_kv_length == 17
    assert contract.task_count == 20
    assert contract.gpu_task_count == 15
    assert contract.runtime_task_count == 5
    assert contract.parent_requests > 0
    assert contract.child_requests >= contract.parent_requests


def test_p24_controls_use_real_p23_barriers_and_reuse_global_pa() -> None:
    config = _config()
    result = run_p23_request_control_timeline(
        Path.cwd(), config["requests"], config["allocator"]
    )
    assert len(result["epochs"]) == 3
    assert all(len(epoch["tasks"]) == 20 for epoch in result["epochs"])
    assert result["memory"]["peak_bytes"] == result["memory"]["capacity_bytes"]
    assert result["memory"]["retired_range_reuse_count"] == 4
    assert result["conservation"]["admitted_requests"] == 6
    assert result["conservation"]["cancelled_before_admission"] == 2
    assert result["conservation"]["zero_in_flight"]
    assert result["gates"]["release_after_all_requests_durable"]


def test_p23_request_controls_are_deterministic() -> None:
    config = _config()
    first = run_p23_request_control_timeline(
        Path.cwd(), config["requests"], config["allocator"]
    )
    second = run_p23_request_control_timeline(
        Path.cwd(), config["requests"], config["allocator"]
    )
    assert first == second


def test_kv18_extrapolation_fails_closed() -> None:
    config = _config()
    request = dict(config["requests"][0])
    request["output_length"] = 2
    request.pop("cancel_time_fs", None)
    with pytest.raises(P23RequestControlError, match="KV18"):
        run_p23_request_control_timeline(
            Path.cwd(), [request, dict(config["requests"][1])], config["allocator"]
        )


def test_long_capacity_pressure_has_deterministic_reuse_and_no_leak() -> None:
    result = run_kv_allocator_pressure_probe(
        epochs=1024,
        batch_size=2,
        per_request_bytes=17408,
        capacity_bytes=34816,
    )
    assert result["allocations"] == result["releases"] == 2048
    assert result["retired_range_reuses"] == 2046
    assert result["overlap_count"] == 0
    assert result["leaked_allocations"] == 0
    assert result["zero_in_flight"] is True
    assert result["exact_gpu_trace_long_generation"] is False
