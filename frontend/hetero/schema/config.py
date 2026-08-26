"""Strict M0 configuration validation using only the Python standard library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a configuration violates the frozen v1 contract."""


_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "simulation",
    "system",
    "backends",
    "model",
    "workload",
    "scheduling",
    "placement",
    "address",
    "metrics",
}

_PROFILES = {
    "model1_atlas_native",
    "model2_host_memory_pcie",
    "model3_gpu_native_3ddram",
    "model4_cxl_memory_tier",
}

_COUPLINGS = {"analytical", "operator_event", "request_cycle"}
_GPU_BACKENDS = {"roofline", "accel_sim"}
_ATLAS_BACKENDS = {"atlasim", "analytical"}
_HOST_BACKENDS = {"none", "analytical", "gem5"}
_GENERATION_MODES = {
    "trace_locked",
    "fixed_tokens",
    "functional_generation",
    "replayed_eos",
}


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be an object")
    return value


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise ConfigError("configuration root must be an object")

    actual = set(config)
    missing = _TOP_LEVEL_KEYS - actual
    unknown = actual - _TOP_LEVEL_KEYS
    if missing:
        raise ConfigError(f"missing top-level fields: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"unknown top-level fields: {sorted(unknown)}")
    if config["schema_version"] != "hetero-sim/v1":
        raise ConfigError("schema_version must be hetero-sim/v1")

    experiment = _mapping(config, "experiment")
    if not isinstance(experiment.get("name"), str) or not experiment["name"].strip():
        raise ConfigError("experiment.name must be a non-empty string")
    seed = experiment.get("seed", 1)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("experiment.seed must be an unsigned integer")
    if experiment.get("generation_mode", "trace_locked") not in _GENERATION_MODES:
        raise ConfigError("invalid experiment.generation_mode")

    simulation = _mapping(config, "simulation")
    coupling = simulation.get("coupling")
    if coupling not in _COUPLINGS:
        raise ConfigError("invalid simulation.coupling")

    system = _mapping(config, "system")
    profile = system.get("profile")
    if profile not in _PROFILES:
        raise ConfigError("invalid system.profile")

    backends = _mapping(config, "backends")
    gpu = _mapping(backends, "gpu")
    atlas = _mapping(backends, "atlas")
    host = _mapping(backends, "host")
    if gpu.get("kind") not in _GPU_BACKENDS:
        raise ConfigError("invalid backends.gpu.kind")
    if atlas.get("kind") not in _ATLAS_BACKENDS:
        raise ConfigError("invalid backends.atlas.kind")
    if host.get("kind", "none") not in _HOST_BACKENDS:
        raise ConfigError("invalid backends.host.kind")

    scheduling = _mapping(config, "scheduling")
    max_tokens = _positive_int(
        scheduling.get("max_batched_tokens"), "scheduling.max_batched_tokens"
    )
    _positive_int(
        scheduling.get("max_num_sequences"), "scheduling.max_num_sequences"
    )
    chunk = _positive_int(
        scheduling.get("prefill_chunk_tokens", 512),
        "scheduling.prefill_chunk_tokens",
    )
    if chunk > max_tokens:
        raise ConfigError(
            "scheduling.prefill_chunk_tokens must not exceed max_batched_tokens"
        )

    workload = _mapping(config, "workload")
    requests = workload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ConfigError("workload.requests must be a non-empty array")
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ConfigError(f"workload.requests[{index}] must be an object")
        _positive_int(request.get("prompt_length"), f"requests[{index}].prompt_length")
        _positive_int(request.get("output_length"), f"requests[{index}].output_length")

    if profile == "model4_cxl_memory_tier":
        access_policy = system.get("access_policy", "copy")
        if access_policy == "remote" and coupling != "request_cycle":
            raise ConfigError("Model 4 remote access requires request_cycle coupling")


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"failed to load {config_path}: {error}") from error
    validate_config(config)
    return config

