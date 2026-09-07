import json
from pathlib import Path

import pytest

from frontend.hetero.request_control_runtime import (
    RequestControlRuntimeError,
    run_request_control_runtime,
)
from frontend.hetero.runner import execute_run
from frontend.hetero.schema import ConfigError, load_and_validate_config, validate_config


def _load(name: str) -> dict[str, object]:
    return load_and_validate_config(f"configs/hetero/experiments/{name}")


def _run(config: dict[str, object]) -> dict[str, object]:
    return run_request_control_runtime(
        config["workload"]["requests"],  # type: ignore[index]
        config["scheduling"],  # type: ignore[arg-type]
        config["model"],  # type: ignore[arg-type]
        config["address"],  # type: ignore[arg-type]
    )


def test_p24_eos_max_length_and_active_cancellation_are_terminal() -> None:
    result = _run(_load("p24_tinyllama_1layer_request_controls_bs3.json"))
    assert result["schema_version"] == "hetero-p24-request-control-runtime/v1"
    assert result["performance_claim_allowed"] is False
    assert result["termination"]["reason_counts"] == {  # type: ignore[index]
        "completed": 0,
        "eos": 1,
        "max_length": 1,
        "cancelled": 1,
    }
    by_id = {item["request_id"]: item for item in result["requests"]}  # type: ignore[index]
    assert by_id["R-eos"]["generated_length"] == 2
    assert by_id["R-max"]["generated_length"] == 3
    assert by_id["R-cancel-active"]["generated_length"] == 1
    assert by_id["R-cancel-active"]["cancel_observed_time_fs"] == 1000
    assert by_id["R-cancel-active"]["final_committed_kv_length"] == 17
    assert result["conservation"]["all_requests_terminal"]  # type: ignore[index]
    assert result["conservation"]["zero_in_flight"]  # type: ignore[index]


def test_p24_capacity_pressure_reuses_retired_global_pa() -> None:
    result = _run(_load("p24_tinyllama_1layer_kv_pressure_bs4.json"))
    memory = result["memory"]
    assert memory["peak_bytes"] == memory["capacity_bytes"] == 32768  # type: ignore[index]
    assert memory["retired_range_reuse_count"] == 2  # type: ignore[index]
    assert memory["capacity_delayed_request_ids"] == [  # type: ignore[index]
        "R1-max",
        "R3-complete",
    ]
    assert result["termination"]["cancelled_before_admission"] == [  # type: ignore[index]
        "R2-cancel-waiting"
    ]
    assert result["conservation"]["admitted_requests"] == 3  # type: ignore[index]
    assert result["conservation"]["allocated_requests"] == 3  # type: ignore[index]
    assert result["conservation"]["released_requests"] == 3  # type: ignore[index]
    assert result["memory"]["zero_bytes_after_retirement"]  # type: ignore[index]


@pytest.mark.parametrize(
    "name",
    [
        "p24_tinyllama_1layer_request_controls_bs3.json",
        "p24_tinyllama_1layer_kv_pressure_bs4.json",
    ],
)
def test_p24_single_layer_contract_is_deterministic(name: str) -> None:
    config = _load(name)
    assert _run(config) == _run(config)


def test_p24_experiment_runner_emits_request_control_artifact(
    tmp_path: Path,
) -> None:
    config = _load("p24_tinyllama_1layer_kv_pressure_bs4.json")
    run_dir = execute_run(config, Path.cwd(), tmp_path)
    payload = json.loads((run_dir / "request_control_runtime.json").read_text())
    assert payload["conservation"]["zero_in_flight"]
    assert payload["memory"]["retired_range_reuse_count"] == 2
    assert payload["performance_claim_allowed"] is False


def test_p24_rejects_multilayer_qualification() -> None:
    config = _load("p24_tinyllama_1layer_request_controls_bs3.json")
    model = dict(config["model"])  # type: ignore[arg-type]
    model["num_layers"] = 2
    with pytest.raises(RequestControlRuntimeError, match="restricted to one layer"):
        run_request_control_runtime(
            config["workload"]["requests"],  # type: ignore[index]
            config["scheduling"],  # type: ignore[arg-type]
            model,
            config["address"],  # type: ignore[arg-type]
        )


def test_p24_schema_rejects_cancellation_before_arrival(tmp_path: Path) -> None:
    config = _load("p24_tinyllama_1layer_request_controls_bs3.json")
    config["workload"]["requests"][0]["arrival_time_fs"] = 1000  # type: ignore[index]
    config["workload"]["requests"][0]["cancel_time_fs"] = 500  # type: ignore[index]
    with pytest.raises(ConfigError, match="must not precede arrival"):
        validate_config(json.loads(json.dumps(config)))
