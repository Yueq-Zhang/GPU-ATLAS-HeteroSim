"""Versioned framework-to-HeteroSim export and shadow-simulation contracts.

The adapters deliberately keep serving frameworks authoritative for tokens,
batch membership and KV ownership.  HeteroSim consumes immutable observations;
it cannot feed scheduling or output changes back into the live framework.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Mapping, Sequence

from .execution_plan import build_single_placement_plan
from .global_memory_map import build_global_memory_map
from .model_graph import ModelSpec, RequestSpec, build_request_graph
from .placement import place_nodes


class FrameworkIntegrationError(ValueError):
    """Raised when framework identity or scheduler state is incomplete."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _positive(config: Mapping[str, object], *keys: str) -> int:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise FrameworkIntegrationError(f"missing positive model field: {'/'.join(keys)}")


def model_spec_from_huggingface(
    config: Mapping[str, object], *, model_name: str, revision: str
) -> ModelSpec:
    """Translate a pinned Hugging Face PretrainedConfig dictionary."""

    if not model_name or not revision:
        raise FrameworkIntegrationError("model name and immutable revision are required")
    hidden = _positive(config, "hidden_size", "word_embed_proj_dim")
    heads = _positive(config, "num_attention_heads")
    if hidden % heads:
        raise FrameworkIntegrationError("hidden size must be divisible by attention heads")
    model_type = str(config.get("model_type", ""))
    if model_type in {"llama", "mistral", "tinyllama"}:
        architecture = "llama"
        mlp_type = "swiglu"
        position = "rope"
        intermediate = _positive(config, "intermediate_size")
        kv_heads = int(config.get("num_key_value_heads", heads))
    elif model_type == "opt":
        architecture = "opt"
        mlp_type = "dense_gelu"
        position = "learned_absolute"
        intermediate = _positive(config, "ffn_dim", "intermediate_size")
        kv_heads = heads
    else:
        raise FrameworkIntegrationError(f"unsupported Hugging Face model_type {model_type!r}")
    dtype = str(config.get("torch_dtype", "float16")).replace("torch.", "")
    dtype_alias = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
    dtype = dtype_alias.get(dtype, dtype)
    bytes_per_element = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype)
    if bytes_per_element is None:
        raise FrameworkIntegrationError(f"unsupported model dtype {dtype}")
    return ModelSpec(
        name=model_name,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_layers=_positive(config, "num_hidden_layers"),
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=hidden // heads,
        vocab_size=_positive(config, "vocab_size"),
        dtype=dtype,
        bytes_per_element=bytes_per_element,
        architecture=architecture,
        mlp_type=mlp_type,
        position_encoding=position,
        tied_embeddings=bool(config.get("tie_word_embeddings", True)),
        input_embedding_mode="token_ids",
        materialize_parameters=True,
        checkpoint_revision=revision,
    )


def load_huggingface_config(
    model_name: str, revision: str, *, local_files_only: bool = True
) -> Mapping[str, object]:
    """Load an actual HF config when transformers is installed; fail closed."""

    try:
        from transformers import AutoConfig  # type: ignore[import-not-found]
    except ImportError as error:
        raise FrameworkIntegrationError(
            "transformers is not installed; use a captured PretrainedConfig dictionary"
        ) from error
    loaded = AutoConfig.from_pretrained(
        model_name, revision=revision, local_files_only=local_files_only
    )
    return dict(loaded.to_dict())


def _graph_record(graph: object) -> dict[str, object]:
    values = []
    for value in graph.values:  # type: ignore[attr-defined]
        item = asdict(value)
        item["storage_class"] = value.storage_class.value
        values.append(item)
    nodes = []
    for node in graph.nodes:  # type: ignore[attr-defined]
        item = asdict(node)
        item["kind"] = node.kind.value
        item["phase"] = node.phase.value
        item["dependencies"] = list(node.dependencies)
        item["read_values"] = list(node.read_values)
        item["write_values"] = list(node.write_values)
        nodes.append(item)
    return {"schema_version": graph.schema_version, "values": values, "nodes": nodes}


def build_framework_export(
    model: ModelSpec,
    requests: Sequence[RequestSpec],
    *,
    framework: str,
    framework_version: str,
    tokenizer_revision: str = "not-applicable",
    adapter_version: str = "heterosim-framework-adapter/v1",
    execution_parameters: Mapping[str, object] | None = None,
    system_profile: str = "model3_gpu_native_3ddram",
    placement: Mapping[str, object] | None = None,
    capacity_bytes: int = 1 << 40,
    alignment_bytes: int = 64,
) -> dict[str, object]:
    """Build canonical graphs, placement tasks and one deterministic Global PA map."""

    if (
        framework not in {"huggingface", "vllm", "tensorrt_llm"}
        or not framework_version
        or not tokenizer_revision
        or not adapter_version
    ):
        raise FrameworkIntegrationError("supported framework and version are required")
    if not requests or len({item.request_id for item in requests}) != len(requests):
        raise FrameworkIntegrationError("request IDs must be non-empty and unique")
    placement = placement or {"mode": "manual", "default_target": "gpu0", "rules": []}
    graph_records: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    placement_records: list[dict[str, object]] = []
    for request in requests:
        graph = build_request_graph(model, request)
        decisions = place_nodes(graph.nodes, placement, active_batch=len(requests))
        plan = build_single_placement_plan(
            graph, decisions, system_profile, "copy", model
        )
        graph_records.append(
            {"request": asdict(request), "graph": _graph_record(graph)}
        )
        for planned in plan.nodes:
            task_id = f"{request.request_id}.{planned.task_id}"
            inputs = [
                {**dict(item), "memory_space_id": "shared0.dram3d"}
                for item in planned.input_values
            ]
            outputs = [
                {**dict(item), "memory_space_id": "shared0.dram3d"}
                for item in planned.output_values
            ]
            tasks.append(
                {
                    "task_id": task_id,
                    "request_id": request.request_id,
                    "op": planned.node.op,
                    "phase": planned.node.phase.value,
                    "layer_id": planned.node.layer_id,
                    "step_id": planned.node.step_id,
                    "device_id": planned.decision.target_device,
                    "dependencies": [
                        f"{request.request_id}.{item}" for item in planned.dependencies
                    ],
                    "input_values": inputs,
                    "output_values": outputs,
                    "attributes": dict(planned.node.attributes),
                }
            )
            placement_records.append(
                {
                    "task_id": task_id,
                    "device_id": planned.decision.target_device,
                    "matched_rule": planned.decision.matched_rule,
                    "reason": planned.decision.reason,
                }
            )
    allocations, global_map = build_global_memory_map(
        {"tasks": tasks}, "shared0.dram3d", capacity_bytes, alignment_bytes
    )
    tensor_bindings = {
        key: {
            "global_address": item.base_address,
            "size_bytes": item.size_bytes,
            "memory_space_id": item.memory_space_id,
        }
        for key, item in sorted(allocations.items())
    }
    model_manifest = {
        "schema_version": "hetero-framework-model-manifest/v1",
        "model": asdict(model),
        "tokenizer_revision": tokenizer_revision,
    }
    request_manifest = {
        "schema_version": "hetero-framework-request-manifest/v1",
        "requests": [asdict(item) for item in requests],
        "execution_parameters": dict(execution_parameters or {}),
    }
    execution_manifest = {
        "schema_version": "hetero-framework-execution-manifest/v1",
        "framework": framework,
        "framework_version": framework_version,
        "adapter_version": adapter_version,
        "system_profile": system_profile,
        "placement_sha256": _digest(placement),
    }
    manifests = {
        "model": model_manifest,
        "requests": request_manifest,
        "execution": execution_manifest,
    }
    identity = {"manifests": manifests}
    return {
        "schema_version": "hetero-framework-export/v1",
        "identity": identity,
        "identity_sha256": _digest(identity),
        "manifests": manifests,
        "graphs": graph_records,
        "execution_graph": {"tasks": tasks, "placements": placement_records},
        "global_memory_map": global_map,
        "tensor_global_pa_bindings": tensor_bindings,
        "shadow_only": True,
        "framework_remains_authoritative": True,
    }


def resolve_gpu_ready_artifacts(
    catalog: Mapping[str, object],
    *,
    model_name: str,
    checkpoint_revision: str,
    phase: str,
    batch_size: int,
    context_length: int,
    q_len: int,
    kv_length: int,
    operators: Sequence[str],
) -> dict[str, str]:
    """Resolve only exact, request-cycle-ready GPU Artifact identities."""

    records = catalog.get("records")
    if not isinstance(records, list):
        raise FrameworkIntegrationError("GPU Ready Catalog records are missing")
    result: dict[str, str] = {}
    for operator in operators:
        candidates = [
            item
            for item in records
            if isinstance(item, Mapping)
            and item.get("operator_type") == operator
            and item.get("batch_size") == batch_size
            and item.get("context_length") == context_length
            and item.get("q_len") == q_len
            and item.get("kv_length") == kv_length
            and item.get("request_cycle_ready") is True
            and item.get("performance_eligible") is False
        ]
        if (
            len(candidates) != 1
            or catalog.get("model") != model_name
            or catalog.get("checkpoint_revision") != checkpoint_revision
            or phase != "decode_step"
        ):
            raise FrameworkIntegrationError(
                f"no exact request-cycle GPU Artifact for {operator}"
            )
        result[operator] = str(candidates[0]["artifact"])
    return result


_ATLAS_GEMM_SHAPES = {
    "qkv_projection": lambda model: (
        model.hidden_size,
        model.hidden_size + 2 * model.num_kv_heads * model.head_dim,
    ),
    "output_projection": lambda model: (model.hidden_size, model.hidden_size),
    "gate_up_projection": lambda model: (model.hidden_size, 2 * model.intermediate_size),
    "fc1_projection": lambda model: (model.hidden_size, model.intermediate_size),
    "down_projection": lambda model: (model.intermediate_size, model.hidden_size),
    "lm_head": lambda model: (model.hidden_size, model.vocab_size),
}


def compile_atlas_tensor_ir(
    *,
    operator: str,
    model: ModelSpec,
    tokens: int,
    core_count: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
) -> dict[str, object]:
    """Lower a projection Tensor IR into ATLAS GEMM and placement records."""

    if operator not in _ATLAS_GEMM_SHAPES:
        raise FrameworkIntegrationError(f"ATLAS Tensor IR lowering is absent for {operator}")
    k_dim, n_dim = _ATLAS_GEMM_SHAPES[operator](model)
    values = (tokens, core_count, tile_m, tile_k, tile_n)
    if any(item <= 0 for item in values) or n_dim % core_count:
        raise FrameworkIntegrationError("invalid ATLAS core or tensor geometry")
    per_core_n = n_dim // core_count
    if tokens % tile_m or k_dim % tile_k or per_core_n % tile_n:
        raise FrameworkIntegrationError("ATLAS tile must exactly divide M/K/per-core-N")
    iterations = (tokens // tile_m) * (k_dim // tile_k) * (per_core_n // tile_n)
    tensor_ir = {
        "op": "matmul",
        "operator": operator,
        "inputs": [
            {"tensor": "activation", "shape": [tokens, k_dim], "dtype": model.dtype},
            {"tensor": "weight", "shape": [k_dim, n_dim], "dtype": model.dtype},
        ],
        "outputs": [
            {"tensor": "output", "shape": [tokens, n_dim], "dtype": model.dtype}
        ],
    }
    operator_description = {
        "operator": [
            {
                "name": operator,
                "type": "gemm",
                "iteration": iterations,
                "execution": {
                    "matrix": [
                        {"name": "gemm_tile", "mac_count": tile_m * tile_k * tile_n}
                    ],
                    "vector": [],
                    "buffer_load": [],
                    "buffer_store": [],
                    "dram": [],
                },
            }
        ]
    }
    per_core = []
    for core_id in range(core_count):
        per_core.append(
            {
                "core_id": core_id,
                "n_begin": core_id * per_core_n,
                "n_end": (core_id + 1) * per_core_n,
                "tensors": {
                    "activation": [tokens, k_dim],
                    "weight": [k_dim, per_core_n],
                    "output": [tokens, per_core_n],
                },
            }
        )
    compiled = {
        "schema_version": "hetero-atlas-tensor-ir-compile/v1",
        "source_tensor_ir": tensor_ir,
        "operator_description": operator_description,
        "data_placement": {"sharding": "N-column", "per_core": per_core},
        "tile": {"M": tile_m, "K": tile_k, "N": tile_n},
        "core_count": core_count,
        "iterations_per_core": iterations,
        "full_atlas_bundle_ready": True,
        "simulation_executed": False,
    }
    return {**compiled, "compile_sha256": _digest(compiled)}


def normalize_vllm_scheduler_events(
    events: Sequence[Mapping[str, object]],
    *,
    page_size_tokens: int,
    page_size_bytes: int = 0,
    global_pa_base: int = 0,
) -> dict[str, object]:
    """Validate continuous/ragged batches and Paged-KV ownership observations."""

    if page_size_tokens <= 0 or page_size_bytes < 0 or global_pa_base < 0:
        raise FrameworkIntegrationError("Paged KV page size must be positive")
    active: set[str] = set()
    page_owner: dict[int, str] = {}
    batches: list[dict[str, object]] = []
    normalized: list[dict[str, object]] = []
    allocation_ledger: list[dict[str, object]] = []
    for sequence, raw in enumerate(events):
        kind = str(raw.get("event", ""))
        request_id = str(raw.get("request_id", ""))
        item = {"sequence": sequence, **dict(raw)}
        if kind == "request_arrived":
            if not request_id or request_id in active:
                raise FrameworkIntegrationError("duplicate or empty vLLM request")
            active.add(request_id)
        elif kind == "kv_page_alloc":
            page = int(raw.get("page_id", -1))
            if request_id not in active or page < 0 or page in page_owner:
                raise FrameworkIntegrationError("invalid or aliased Paged KV allocation")
            page_owner[page] = request_id
            if page_size_bytes:
                item["global_address"] = global_pa_base + page * page_size_bytes
                item["size_bytes"] = page_size_bytes
                allocation_ledger.append(
                    {
                        "sequence": sequence,
                        "event": "allocate",
                        "page_id": page,
                        "request_id": request_id,
                        "global_address": item["global_address"],
                        "size_bytes": page_size_bytes,
                    }
                )
        elif kind == "batch_formed":
            members = list(raw.get("request_ids", []))  # type: ignore[arg-type]
            token_counts = list(raw.get("token_counts", []))  # type: ignore[arg-type]
            if (
                not members
                or len(members) != len(token_counts)
                or len(set(members)) != len(members)
                or any(str(member) not in active for member in members)
                or any(int(count) <= 0 for count in token_counts)
            ):
                raise FrameworkIntegrationError("invalid vLLM continuous/ragged batch")
            batch = {
                "batch_id": str(raw.get("batch_id", f"batch-{sequence}")),
                "request_ids": [str(member) for member in members],
                "token_counts": [int(count) for count in token_counts],
                "ragged": len(set(int(count) for count in token_counts)) > 1,
            }
            batches.append(batch)
        elif kind == "kv_page_free":
            page = int(raw.get("page_id", -1))
            if page_owner.get(page) != request_id:
                raise FrameworkIntegrationError("Paged KV free does not match owner")
            del page_owner[page]
            allocation_ledger.append(
                {
                    "sequence": sequence,
                    "event": "free",
                    "page_id": page,
                    "request_id": request_id,
                }
            )
        elif kind == "request_finished":
            if request_id not in active or request_id in page_owner.values():
                raise FrameworkIntegrationError("vLLM request finished with live KV pages")
            active.remove(request_id)
        elif kind not in {"decode_step", "prefill_step"}:
            raise FrameworkIntegrationError(f"unknown vLLM scheduler event {kind!r}")
        normalized.append(item)
    return {
        "schema_version": "hetero-vllm-shadow-events/v1",
        "page_size_tokens": page_size_tokens,
        "events": normalized,
        "allocation_ledger": allocation_ledger,
        "batches": batches,
        "active_requests_at_snapshot_end": sorted(active),
        "live_page_owners": {str(key): value for key, value in sorted(page_owner.items())},
        "continuous_batching_observed": len(batches) > 1,
        "ragged_batching_observed": any(bool(item["ragged"]) for item in batches),
        "paged_kv_global_pa_bound": bool(page_size_bytes),
        "event_sha256": _digest(normalized),
    }


def normalize_tensorrt_llm_execution(manifest: Mapping[str, object]) -> dict[str, object]:
    """Bind an engine, optimization profile, tactics and plugins by hash."""

    required = {"engine_sha256", "builder_version", "profile", "tactics", "plugins"}
    if set(manifest) != required:
        raise FrameworkIntegrationError("TensorRT-LLM manifest keys are not exact")
    engine = str(manifest["engine_sha256"])
    if len(engine) != 64 or any(char not in "0123456789abcdef" for char in engine):
        raise FrameworkIntegrationError("TensorRT-LLM engine SHA-256 is invalid")
    profile = manifest["profile"]
    if not isinstance(profile, Mapping):
        raise FrameworkIntegrationError("TensorRT-LLM profile must be an object")
    for name in ("min_batch", "max_batch", "min_seq", "max_seq"):
        if int(profile.get(name, 0)) <= 0:
            raise FrameworkIntegrationError(f"invalid TensorRT-LLM profile {name}")
    if int(profile["min_batch"]) > int(profile["max_batch"]) or int(
        profile["min_seq"]
    ) > int(profile["max_seq"]):
        raise FrameworkIntegrationError("TensorRT-LLM profile bounds are inverted")
    tactics = manifest["tactics"]
    plugins = manifest["plugins"]
    if not isinstance(tactics, list) or not tactics or not isinstance(plugins, list):
        raise FrameworkIntegrationError("TensorRT-LLM tactics/plugins are incomplete")
    normalized = dict(manifest)
    return {
        "schema_version": "hetero-tensorrt-llm-shadow-execution/v1",
        "identity": normalized,
        "identity_sha256": _digest(normalized),
        "shadow_only": True,
    }


def build_shadow_simulation_record(
    framework_export: Mapping[str, object],
    observations: Sequence[Mapping[str, object]],
    simulation_results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Join immutable framework observations with simulator results."""

    if framework_export.get("schema_version") != "hetero-framework-export/v1":
        raise FrameworkIntegrationError("invalid framework export")
    observation_ids = [str(item.get("observation_id", "")) for item in observations]
    result_ids = [str(item.get("observation_id", "")) for item in simulation_results]
    if (
        not observation_ids
        or len(set(observation_ids)) != len(observation_ids)
        or sorted(observation_ids) != sorted(result_ids)
    ):
        raise FrameworkIntegrationError("shadow observation/result identity mismatch")
    payload = {
        "framework_identity_sha256": framework_export["identity_sha256"],
        "observations": [dict(item) for item in observations],
        "simulation_results": [dict(item) for item in simulation_results],
    }
    return {
        "schema_version": "hetero-shadow-simulation/v1",
        **payload,
        "record_sha256": _digest(payload),
        "feedback_to_framework": False,
        "tokens_or_scheduler_mutated": False,
        "performance_claim_allowed": False,
    }
