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

