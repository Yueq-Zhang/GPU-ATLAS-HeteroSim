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
