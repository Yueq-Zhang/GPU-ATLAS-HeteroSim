from copy import deepcopy

import pytest

from frontend.hetero.schema import ConfigError, load_and_validate_config, validate_config


SMOKE_CONFIG = "configs/hetero/experiments/m0_smoke.yaml"


def test_smoke_config_is_valid() -> None:
    config = load_and_validate_config(SMOKE_CONFIG)
    assert config["system"]["profile"] == "model1_atlas_native"


def test_unknown_top_level_field_is_rejected() -> None:
    config = load_and_validate_config(SMOKE_CONFIG)
    broken = deepcopy(config)
    broken["unexpected"] = True
    with pytest.raises(ConfigError, match="unknown top-level"):
        validate_config(broken)


def test_model4_remote_requires_request_cycle() -> None:
    config = load_and_validate_config(SMOKE_CONFIG)
    broken = deepcopy(config)
    broken["system"] = {
        "profile": "model4_cxl_memory_tier",
        "access_policy": "remote",
    }
    with pytest.raises(ConfigError, match="requires request_cycle"):
        validate_config(broken)


def test_component_refs_expand_to_canonical_content() -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m1_model3_gpu_native_3ddram.json"
    )
    assert config["model"]["name"] == "tiny_llama_2layer"
    assert config["workload"]["requests"][0]["request_id"] == "R0"
    assert "ref" not in config["model"]
    assert "ref" not in config["scheduling"]


def test_unknown_nested_field_is_rejected() -> None:
    config = load_and_validate_config(SMOKE_CONFIG)
    broken = deepcopy(config)
    broken["address"]["channel_bits"] = 4
    with pytest.raises(ConfigError, match="unknown address fields"):
        validate_config(broken)


def test_duplicate_request_id_is_rejected() -> None:
    config = load_and_validate_config(SMOKE_CONFIG)
    broken = deepcopy(config)
    broken["workload"]["requests"].append(
        deepcopy(broken["workload"]["requests"][0])
    )
    with pytest.raises(ConfigError, match="duplicate request_id"):
        validate_config(broken)


def test_gpu_only_shared_memory_rejects_logic_die_initiator() -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model3_gpu_only_no_logic_die_reference.json"
    )
    broken = deepcopy(config)
    broken["system"]["memory_services"]["shared0.dram3d"]["initiator_order"] = [
        "gpu0",
        "atlas0.compute",
    ]
    with pytest.raises(ConfigError, match="requires initiator_order"):
        validate_config(broken)


def test_gpu_only_shared_memory_requires_disabled_atlas_backend() -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model3_gpu_only_no_logic_die_reference.json"
    )
    broken = deepcopy(config)
    broken["backends"]["atlas"] = {
        "kind": "analytical",
        "effective_compute_flops_per_s": 1,
        "effective_memory_bandwidth_Bps": 1,
        "parameter_source": "test",
    }
    with pytest.raises(ConfigError, match="backends.atlas.kind=none"):
        validate_config(broken)


def test_gpu_only_shared_memory_rejects_non_gpu_placement() -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/m8_model3_gpu_only_no_logic_die_reference.json"
    )
    broken = deepcopy(config)
    broken["placement"]["default_target"] = "atlas0.compute"
    with pytest.raises(ConfigError, match="placement.default_target=gpu0"):
        validate_config(broken)


@pytest.mark.parametrize(
    "name",
    [
        "p10b_b_tinyllama_prefill_1layer_mixed_live_ramulator2.json",
        "p12_tinyllama_prefill_1layer_gpu_live_ramulator2.json",
        "p13_tinyllama_prefill_22layer_ctx16_gpu_live_ramulator2.json",
        "p14_tinyllama_prefill_bs1_ctx1024_gpu_live_ramulator2.json",
    ],
)
def test_prefill_cycle_stage_configs_are_strictly_valid(name: str) -> None:
    config = load_and_validate_config(f"configs/hetero/experiments/{name}")
    assert config["simulation"] == {
        "coupling": "request_cycle",
        "execution_mode": "prefill_cycle",
        "validation_policy": "required",
    }
    assert config["model"]["materialize_parameters"] is True
    assert config["workload"]["requests"][0]["output_length"] == 1


def test_prefill_cycle_rejects_decode_and_nonmaterialized_parameters() -> None:
    config = load_and_validate_config(
        "configs/hetero/experiments/p14_tinyllama_prefill_bs1_ctx1024_gpu_live_ramulator2.json"
    )
    broken = deepcopy(config)
    broken["workload"]["requests"][0]["output_length"] = 2
    with pytest.raises(ConfigError, match="output_length=1"):
        validate_config(broken)
    broken = deepcopy(config)
    broken["model"]["materialize_parameters"] = False
    with pytest.raises(ConfigError, match="materialize_parameters=true"):
        validate_config(broken)
