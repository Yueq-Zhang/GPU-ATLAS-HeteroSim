"""Strict M0 configuration validation using only the Python standard library."""

from __future__ import annotations

import json
from copy import deepcopy
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

_MODEL_REQUIRED = {
    "name",
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "dtype",
    "bytes_per_element",
}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(f"unknown {path} fields: {sorted(unknown)}")


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be an object")
    return value


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _unsigned_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConfigError(f"{path} must be an unsigned integer")
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
    _reject_unknown(simulation, {"coupling", "execution_mode"}, "simulation")
    coupling = simulation.get("coupling")
    if coupling not in _COUPLINGS:
        raise ConfigError("invalid simulation.coupling")

    system = _mapping(config, "system")
    _reject_unknown(
        system, {"profile", "topology_ref", "access_policy", "links"}, "system"
    )
    profile = system.get("profile")
    if profile not in _PROFILES:
        raise ConfigError("invalid system.profile")

    backends = _mapping(config, "backends")
    _reject_unknown(backends, {"gpu", "atlas", "host"}, "backends")
    gpu = _mapping(backends, "gpu")
    atlas = _mapping(backends, "atlas")
    host = _mapping(backends, "host")
    for backend_name, backend in (("gpu", gpu), ("atlas", atlas), ("host", host)):
        _reject_unknown(
            backend,
            {
                "kind",
                "ref",
                "requested_timing_mode",
                "effective_compute_flops_per_s",
                "effective_memory_bandwidth_Bps",
                "parameter_source",
            },
            f"backends.{backend_name}",
        )
    if gpu.get("kind") not in _GPU_BACKENDS:
        raise ConfigError("invalid backends.gpu.kind")
    if atlas.get("kind") not in _ATLAS_BACKENDS:
        raise ConfigError("invalid backends.atlas.kind")
    if host.get("kind", "none") not in _HOST_BACKENDS:
        raise ConfigError("invalid backends.host.kind")

    execution_mode = simulation.get("execution_mode", "scheduler_validation")
    if execution_mode not in {"scheduler_validation", "analytical_preview", "full_runtime"}:
        raise ConfigError("invalid simulation.execution_mode")
    if execution_mode == "analytical_preview":
        for backend_name, backend in (("gpu", gpu), ("atlas", atlas)):
            _positive_int(
                backend.get("effective_compute_flops_per_s"),
                f"backends.{backend_name}.effective_compute_flops_per_s",
            )
            _positive_int(
                backend.get("effective_memory_bandwidth_Bps"),
                f"backends.{backend_name}.effective_memory_bandwidth_Bps",
            )
            source = backend.get("parameter_source")
            if not isinstance(source, str) or not source:
                raise ConfigError(f"backends.{backend_name}.parameter_source is required")

        links = system.get("links")
        if not isinstance(links, Mapping) or not links:
            raise ConfigError("system.links must be a non-empty object")
        for link_id, link in links.items():
            if not isinstance(link_id, str) or not link_id or not isinstance(link, Mapping):
                raise ConfigError("system.links entries must be named objects")
            _reject_unknown(
                link,
                {
                    "wire_bandwidth_Bps",
                    "latency_fs",
                    "header_bytes",
                    "resource_id",
                    "parameter_source",
                },
                f"system.links.{link_id}",
            )
            _positive_int(
                link.get("wire_bandwidth_Bps"),
                f"system.links.{link_id}.wire_bandwidth_Bps",
            )
            _unsigned_int(link.get("latency_fs", 0), f"system.links.{link_id}.latency_fs")
            _unsigned_int(link.get("header_bytes", 0), f"system.links.{link_id}.header_bytes")
            for field in ("resource_id", "parameter_source"):
                if not isinstance(link.get(field), str) or not link[field]:
                    raise ConfigError(f"system.links.{link_id}.{field} is required")

    model = _mapping(config, "model")
    _reject_unknown(model, _MODEL_REQUIRED, "model")
    missing_model = _MODEL_REQUIRED - set(model)
    if missing_model:
        raise ConfigError(f"missing model fields: {sorted(missing_model)}")
    for key in _MODEL_REQUIRED - {"name", "dtype"}:
        _positive_int(model.get(key), f"model.{key}")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ConfigError("model.name must be a non-empty string")
    if model["num_attention_heads"] * model["head_dim"] != model["hidden_size"]:
        raise ConfigError(
            "model.num_attention_heads * model.head_dim must equal hidden_size"
        )

    scheduling = _mapping(config, "scheduling")
    _reject_unknown(
        scheduling,
        {
            "mode",
            "epoch_mode",
            "admission",
            "decode_priority",
            "prefill_aging",
            "max_num_sequences",
            "max_batched_tokens",
            "prefill_chunk_tokens",
            "max_prefill_wait_epochs",
            "kv_reservation_mode",
            "epoch_duration_fs",
        },
        "scheduling",
    )
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
    _positive_int(scheduling.get("epoch_duration_fs"), "scheduling.epoch_duration_fs")

    workload = _mapping(config, "workload")
    _reject_unknown(workload, {"requests"}, "workload")
    requests = workload.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ConfigError("workload.requests must be a non-empty array")
    request_ids: set[str] = set()
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ConfigError(f"workload.requests[{index}] must be an object")
        _reject_unknown(
            request,
            {
                "request_id",
                "arrival_time_fs",
                "prompt_length",
                "output_length",
                "priority",
            },
            f"workload.requests[{index}]",
        )
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ConfigError(f"requests[{index}].request_id must be non-empty")
        if request_id in request_ids:
            raise ConfigError(f"duplicate request_id: {request_id}")
        request_ids.add(request_id)
        _positive_int(request.get("prompt_length"), f"requests[{index}].prompt_length")
        _positive_int(request.get("output_length"), f"requests[{index}].output_length")
        arrival = request.get("arrival_time_fs", 0)
        if not isinstance(arrival, int) or isinstance(arrival, bool) or arrival < 0:
            raise ConfigError(f"requests[{index}].arrival_time_fs must be unsigned")

    placement = _mapping(config, "placement")
    _reject_unknown(
        placement, {"mode", "unit", "default_target", "rules", "data"}, "placement"
    )
    if placement.get("mode") not in {"manual", "rule_based", "auto_dse"}:
        raise ConfigError("invalid placement.mode")

    address = _mapping(config, "address")
    _reject_unknown(
        address, {"allocator", "page_tokens", "kv_capacity_bytes"}, "address"
    )
    _positive_int(address.get("page_tokens"), "address.page_tokens")
    _positive_int(address.get("kv_capacity_bytes"), "address.kv_capacity_bytes")

    if profile == "model4_cxl_memory_tier":
        access_policy = system.get("access_policy", "copy")
        if access_policy not in {"remote", "copy", "migrate"}:
            raise ConfigError("invalid Model 4 access_policy")
        if access_policy == "remote" and coupling != "request_cycle":
            raise ConfigError("Model 4 remote access requires request_cycle coupling")
        if access_policy == "remote" and gpu.get("kind") != "accel_sim":
            raise ConfigError("Model 4 remote access requires accel_sim GPU backend")


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"failed to load {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be an object")
    resolved = deepcopy(config)
    project_root = next(
        (
            parent
            for parent in (config_path.resolve().parent, *config_path.resolve().parents)
            if (parent / "pyproject.toml").is_file()
        ),
        Path.cwd(),
    )

    def expand(section: str) -> None:
        value = resolved.get(section)
        if not isinstance(value, Mapping) or "ref" not in value:
            return
        if set(value) != {"ref"}:
            raise ConfigError(f"{section}.ref cannot be combined with inline fields")
        ref = value["ref"]
        if not isinstance(ref, str) or not ref:
            raise ConfigError(f"{section}.ref must be a non-empty path")
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = project_root / ref_path
        try:
            loaded = json.loads(ref_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(f"failed to expand {section}.ref {ref_path}: {error}") from error
        if not isinstance(loaded, dict):
            raise ConfigError(f"{section}.ref must resolve to an object")
        resolved[section] = loaded

    for section in ("model", "workload", "scheduling", "placement", "address", "metrics"):
        expand(section)
    validate_config(resolved)
    return resolved
