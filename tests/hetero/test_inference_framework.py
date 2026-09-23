from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from frontend.hetero.inference_framework import (
    FrameworkIntegrationError,
    build_framework_export,
    build_shadow_simulation_record,
    compile_atlas_tensor_ir,
    model_spec_from_huggingface,
    normalize_tensorrt_llm_execution,
    normalize_vllm_scheduler_events,
    resolve_gpu_ready_artifacts,
)
from frontend.hetero.framework_runtime import (
    bind_huggingface_runtime_to_export,
    compile_atlas_executable_artifact,
    normalize_huggingface_runtime_observation,
    summarize_atlas_memory_trace,
)
from frontend.hetero.model_graph import RequestSpec


ROOT = Path(__file__).resolve().parents[2]


def _model():
    return model_spec_from_huggingface(
        {
            "model_type": "llama",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "torch_dtype": "float16",
            "tie_word_embeddings": False,
        },
        model_name="fixture/llama",
        revision="0" * 40,
    )


def test_huggingface_graph_tensor_global_pa_and_shadow_join() -> None:
    export = build_framework_export(
        _model(),
        [RequestSpec("request-0", prompt_length=4, output_length=1)],
        framework="huggingface",
        framework_version="4.fixture",
        tokenizer_revision="tok-revision",
        adapter_version="adapter/v1",
        execution_parameters={"max_new_tokens": 1},
        capacity_bytes=1 << 24,
    )
    assert export["global_memory_map"]["non_overlapping"] is True
    assert export["global_memory_map"]["allocation_count"] > 0
    assert export["framework_remains_authoritative"] is True
    assert export["manifests"]["execution"]["adapter_version"] == "adapter/v1"
    shadow = build_shadow_simulation_record(
        export,
        [{"observation_id": "o0", "batch_id": "b0"}],
        [{"observation_id": "o0", "simulated_cycles": 10}],
    )
    assert shadow["feedback_to_framework"] is False
    assert shadow["tokens_or_scheduler_mutated"] is False


def test_atlas_tensor_ir_compiles_projection_and_rejects_unknown_op() -> None:
    compiled = compile_atlas_tensor_ir(
        operator="qkv_projection",
        model=_model(),
        tokens=1,
        core_count=4,
        tile_m=1,
        tile_k=8,
        tile_n=4,
    )
    assert compiled["full_atlas_bundle_ready"] is True
    assert compiled["iterations_per_core"] == 4
    assert len(compiled["data_placement"]["per_core"]) == 4
    with pytest.raises(FrameworkIntegrationError, match="lowering is absent"):
        compile_atlas_tensor_ir(
            operator="softmax",
            model=_model(),
            tokens=1,
            core_count=4,
            tile_m=1,
            tile_k=8,
            tile_n=4,
        )


def test_vllm_continuous_ragged_batch_and_paged_kv_are_conserved() -> None:
    events = [
        {"event": "request_arrived", "request_id": "r0"},
        {"event": "request_arrived", "request_id": "r1"},
        {"event": "kv_page_alloc", "request_id": "r0", "page_id": 10},
        {"event": "kv_page_alloc", "request_id": "r1", "page_id": 11},
        {
            "event": "batch_formed",
            "batch_id": "b0",
            "request_ids": ["r0", "r1"],
            "token_counts": [4, 2],
        },
        {"event": "decode_step", "request_id": "r0"},
        {
            "event": "batch_formed",
            "batch_id": "b1",
            "request_ids": ["r0"],
            "token_counts": [1],
        },
        {"event": "kv_page_free", "request_id": "r0", "page_id": 10},
        {"event": "request_finished", "request_id": "r0"},
        {"event": "kv_page_free", "request_id": "r1", "page_id": 11},
        {"event": "request_finished", "request_id": "r1"},
    ]
    result = normalize_vllm_scheduler_events(
        events,
        page_size_tokens=16,
        page_size_bytes=4096,
        global_pa_base=1 << 30,
    )
    assert result["continuous_batching_observed"] is True
    assert result["ragged_batching_observed"] is True
    assert result["live_page_owners"] == {}
    assert result["active_requests_at_snapshot_end"] == []
    assert result["paged_kv_global_pa_bound"] is True
    assert result["allocation_ledger"][0]["global_address"] == (1 << 30) + 10 * 4096


def test_vllm_page_alias_fails_closed() -> None:
    with pytest.raises(FrameworkIntegrationError, match="aliased"):
        normalize_vllm_scheduler_events(
            [
                {"event": "request_arrived", "request_id": "r0"},
                {"event": "request_arrived", "request_id": "r1"},
                {"event": "kv_page_alloc", "request_id": "r0", "page_id": 4},
                {"event": "kv_page_alloc", "request_id": "r1", "page_id": 4},
            ],
            page_size_tokens=16,
        )


def test_tensorrt_llm_engine_profile_tactics_and_plugins_are_identity_bound() -> None:
    result = normalize_tensorrt_llm_execution(
        {
            "engine_sha256": "a" * 64,
            "builder_version": "fixture",
            "profile": {
                "min_batch": 1,
                "max_batch": 8,
                "min_seq": 1,
                "max_seq": 2048,
            },
            "tactics": [{"layer": "l0", "tactic": "t0"}],
            "plugins": [{"name": "paged_attention", "sha256": "b" * 64}],
        }
    )
    assert result["shadow_only"] is True


def test_gpu_catalog_resolution_is_exact_and_does_not_extrapolate_shape() -> None:
    catalog = json.loads(
        (ROOT / "validation/p23/ready_catalog.json").read_text(encoding="utf-8")
    )
    result = resolve_gpu_ready_artifacts(
        catalog,
        model_name="TinyLlama-1.1B",
        checkpoint_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        phase="decode_step",
        batch_size=2,
        context_length=16,
        q_len=1,
        kv_length=17,
        operators=["attention_norm", "qkv_projection"],
    )
    assert set(result) == {"attention_norm", "qkv_projection"}
    with pytest.raises(FrameworkIntegrationError, match="no exact"):
        resolve_gpu_ready_artifacts(
            catalog,
            model_name="TinyLlama-1.1B",
            checkpoint_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            phase="decode_step",
            batch_size=2,
            context_length=17,
            q_len=1,
            kv_length=18,
            operators=["attention_norm"],
        )


def _runtime_tensor(
    tensor_id: str,
    role: str,
    address: int,
    *,
    storage_base: int,
) -> dict[str, object]:
    return {
        "tensor_id": tensor_id,
        "role": role,
        "path": role,
        "address": address,
        "storage_base": storage_base,
        "storage_size_bytes": 4096,
        "storage_offset_bytes": address - storage_base,
        "logical_nbytes": 128,
        "shape": [1, 64],
        "strides": [64, 1],
        "dtype": "float16",
        "device": "cuda:0",
        "item_size_bytes": 2,
    }


def _runtime_observation() -> dict[str, object]:
    return {
        "schema_version": "hetero-huggingface-runtime-observation/v1",
        "framework": {"name": "huggingface", "version": "4.fixture"},
        "model": {"name": "fixture/llama", "revision": "0" * 40},
        "request": {
            "request_id": "request-0",
            "batch_size": 1,
            "prompt_tokens": 4,
            "decode_tokens": 1,
            "dtype": "fp16",
        },
        "device": {"name": "fixture", "compute_capability": [8, 9]},
        "events": [
            {"sequence": 0, "event": "request_arrive", "request_id": "request-0"},
            {
                "sequence": 1,
                "event": "phase_begin",
                "phase": "prefill",
                "q_len": 4,
                "kv_length": 4,
            },
            {
                "sequence": 2,
                "event": "module_begin",
                "phase": "prefill",
                "module_path": "model.layers.0.input_layernorm",
                "module_class": "RMSNorm",
                "logical_operator": "attention_norm",
                "fusion_group": "attention_norm",
                "tensors": [
                    _runtime_tensor(
                        "prefill.input", "input", 0x1000, storage_base=0x1000
                    )
                ],
            },
            {
                "sequence": 3,
                "event": "module_end",
                "phase": "prefill",
                "module_path": "model.layers.0.input_layernorm",
                "module_class": "RMSNorm",
                "logical_operator": "attention_norm",
                "fusion_group": "attention_norm",
                "tensors": [
                    _runtime_tensor(
                        "prefill.output", "output", 0x1080, storage_base=0x1000
                    )
                ],
            },
            {"sequence": 4, "event": "phase_end", "phase": "prefill"},
            {"sequence": 5, "event": "sampling", "token_id": 7},
            {
                "sequence": 6,
                "event": "phase_begin",
                "phase": "decode_step",
                "q_len": 1,
                "kv_length": 5,
            },
            {"sequence": 7, "event": "phase_end", "phase": "decode_step"},
            {"sequence": 8, "event": "request_finish", "request_id": "request-0"},
        ],
        "result": {
            "input_ids_sha256": "a" * 64,
            "next_token_id": 7,
            "prefill_logits_sha256": "b" * 64,
            "decode_logits_sha256": "c" * 64,
        },
    }


def test_huggingface_runtime_callbacks_aliases_global_pa_and_export_binding() -> None:
    raw = _runtime_observation()
    runtime = normalize_huggingface_runtime_observation(raw, global_pa_base=1 << 20)
    assert runtime["actual_runtime_callbacks_observed"] is True
    assert len(runtime["alias_groups"]) == 1
    bindings = runtime["tensor_global_pa_bindings"]
    assert bindings["prefill.output"]["global_address"] == (
        bindings["prefill.input"]["global_address"] + 128
    )
    export = build_framework_export(
        _model(),
        [RequestSpec("request-0", prompt_length=4, output_length=1)],
        framework="huggingface",
        framework_version="4.fixture",
        tokenizer_revision="0" * 40,
    )
    joined = bind_huggingface_runtime_to_export(export, runtime)
    assert joined["online_runtime_observation_connected"] is True
    assert joined["framework_token_source"] == "huggingface"
    assert joined["performance_claim_allowed"] is False


def test_huggingface_simulation_key_excludes_allocator_workspace_noise() -> None:
    first_raw = _runtime_observation()
    second_raw = copy.deepcopy(first_raw)
    second_input = second_raw["events"][2]["tensors"][0]
    second_input["address"] = 0x2000
    second_input["storage_base"] = 0x2000
    second_output = second_raw["events"][3]["tensors"][0]
    second_output["address"] = 0x3080
    second_output["storage_base"] = 0x3000
    first = normalize_huggingface_runtime_observation(first_raw)
    second = normalize_huggingface_runtime_observation(second_raw)
    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["memory_map_sha256"] != second["memory_map_sha256"]
    export = build_framework_export(
        _model(),
        [RequestSpec("request-0", prompt_length=4, output_length=1)],
        framework="huggingface",
        framework_version="4.fixture",
        tokenizer_revision="0" * 40,
    )
    first_binding = bind_huggingface_runtime_to_export(export, first)
    second_binding = bind_huggingface_runtime_to_export(export, second)
    assert first_binding["simulation_key"] == second_binding["simulation_key"]
    assert (
        first_binding["runtime_memory_map_sha256"]
        != second_binding["runtime_memory_map_sha256"]
    )


def test_huggingface_runtime_rejects_unbalanced_module_callbacks() -> None:
    raw = _runtime_observation()
    raw["events"][3]["module_path"] = "model.layers.0.wrong"  # type: ignore[index]
    with pytest.raises(FrameworkIntegrationError, match="unbalanced"):
        normalize_huggingface_runtime_observation(raw)


def test_atlas_executable_artifact_emits_canonical_global_pa_trace() -> None:
    compiled = compile_atlas_tensor_ir(
        operator="qkv_projection",
        model=_model(),
        tokens=1,
        core_count=4,
        tile_m=1,
        tile_k=8,
        tile_n=4,
    )
    artifact = compile_atlas_executable_artifact(
        compiled,
        tensor_global_pa={
            "activation": 0x100000,
            "weight": 0x200000,
            "output": 0x300000,
        },
        compiler_version="heterosim-atlas-stage-compiler/v1",
    )
    summary = summarize_atlas_memory_trace(artifact)
    assert artifact["global_pa_bound"] is True
    assert summary["request_count"] > 0
    assert summary["read_requests"] > summary["write_requests"] > 0
    assert set(summary["requests_by_tensor"]) == {"activation", "output", "weight"}


def test_atlas_executable_artifact_rejects_unaligned_binding() -> None:
    compiled = compile_atlas_tensor_ir(
        operator="qkv_projection",
        model=_model(),
        tokens=1,
        core_count=4,
        tile_m=1,
        tile_k=8,
        tile_n=4,
    )
    with pytest.raises(FrameworkIntegrationError, match="not aligned"):
        compile_atlas_executable_artifact(
            compiled,
            tensor_global_pa={
                "activation": 0x100001,
                "weight": 0x200000,
                "output": 0x300000,
            },
            compiler_version="heterosim-atlas-stage-compiler/v1",
        )
