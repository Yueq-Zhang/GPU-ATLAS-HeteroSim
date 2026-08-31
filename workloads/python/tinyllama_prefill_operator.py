#!/usr/bin/env python3
"""Execute one shape-locked TinyLlama Prefill operator for NVBit capture.

The program intentionally runs one target operator per process.  Model loading,
host-side tensor construction and host-to-device copies happen before the
optional CUDA profiler range.  The emitted metadata records every persistent
allocation referenced by the target so a later manifest builder can normalize
TraceAddr to TensorID + offset without treating capture addresses as Global PA.

This is an artifact-production workload, not an end-to-end latency benchmark.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from frontend.hetero.capture_allocation_ranges import (
    allocator_ranges_from_events,
    allocator_segment_ranges_for_addresses,
    merge_address_ranges,
)


SUPPORTED_OPERATORS = (
    "attention_norm",
    "qkv_projection",
    "rope",
    "kv_append",
    "causal_attention",
    "output_projection",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
)

FINAL_POSITION_OPERATORS = frozenset(("final_norm", "lm_head", "sampling"))


@dataclass(frozen=True)
class Target:
    run: Callable[[], dict[str, torch.Tensor]]
    inputs: dict[str, torch.Tensor]
    parameters: dict[str, torch.Tensor]
    implementation: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--operator", required=True, choices=SUPPORTED_OPERATORS)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--driver-profiler", action="store_true")
    parser.add_argument("--capture-allocator-history", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args()


def _host_random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float16).cuda()


def _target(
    name: str,
    model: Any,
    context: int,
    batch_size: int,
) -> Target:
    layer = model.model.layers[0]
    hidden_size = int(model.config.hidden_size)
    head_dim = int(model.config.head_dim)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    hidden = _host_random((batch_size, context, hidden_size), 11)

    if name == "attention_norm":
        def run_norm() -> dict[str, torch.Tensor]:
            return {"output": layer.input_layernorm(hidden)}

        return Target(
            run_norm,
            {"input": hidden},
            {"weight": layer.input_layernorm.weight},
            "transformers.LlamaRMSNorm",
        )

    if name == "qkv_projection":
        attention = layer.self_attn

        def run_qkv() -> dict[str, torch.Tensor]:
            query = attention.q_proj(hidden)
            key = attention.k_proj(hidden)
            value = attention.v_proj(hidden)
            packed = torch.cat((query, key, value), dim=-1)
            return {
                "query": query,
                "key": key,
                "value": value,
                "output": packed,
            }

        return Target(
            run_qkv,
            {"input": hidden},
            {
                "q_weight": attention.q_proj.weight,
                "k_weight": attention.k_proj.weight,
                "v_weight": attention.v_proj.weight,
            },
            "three_exact_checkpoint_linear_projections_plus_pack",
        )

    if name == "output_projection":
        attention_context = _host_random(
            (batch_size, context, hidden_size), 31
        )

        def run_output_projection() -> dict[str, torch.Tensor]:
            return {"output": layer.self_attn.o_proj(attention_context)}

        return Target(
            run_output_projection,
            {"input": attention_context},
            {"weight": layer.self_attn.o_proj.weight},
            "transformers.LlamaAttention.o_proj",
        )

    if name == "mlp_norm":
        def run_mlp_norm() -> dict[str, torch.Tensor]:
            return {"output": layer.post_attention_layernorm(hidden)}

        return Target(
            run_mlp_norm,
            {"input": hidden},
            {"weight": layer.post_attention_layernorm.weight},
            "transformers.LlamaRMSNorm.post_attention_layernorm",
        )

    if name == "gate_up_projection":
        mlp = layer.mlp

        def run_gate_up() -> dict[str, torch.Tensor]:
            gate = mlp.gate_proj(hidden)
            up = mlp.up_proj(hidden)
            packed = torch.cat((gate, up), dim=-1)
            return {"gate": gate, "up": up, "output": packed}

        return Target(
            run_gate_up,
            {"input": hidden},
            {
                "gate_weight": mlp.gate_proj.weight,
                "up_weight": mlp.up_proj.weight,
            },
            "two_exact_checkpoint_linear_projections_plus_pack",
        )

    if name == "silu_multiply":
        intermediate_size = int(model.config.intermediate_size)
        gate = _host_random((batch_size, context, intermediate_size), 41)
        up = _host_random((batch_size, context, intermediate_size), 42)

        def run_silu_multiply() -> dict[str, torch.Tensor]:
            return {"output": F.silu(gate) * up}

        return Target(
            run_silu_multiply,
            {"gate": gate, "up": up},
            {},
            "torch.nn.functional.silu_times_up_projection",
        )

    if name == "down_projection":
        intermediate_size = int(model.config.intermediate_size)
        activated = _host_random(
            (batch_size, context, intermediate_size), 51
        )

        def run_down_projection() -> dict[str, torch.Tensor]:
            return {"output": layer.mlp.down_proj(activated)}

        return Target(
            run_down_projection,
            {"input": activated},
            {"weight": layer.mlp.down_proj.weight},
            "transformers.LlamaMLP.down_proj",
        )

    if name == "final_norm":
        final_hidden = _host_random((batch_size, 1, hidden_size), 61)

        def run_final_norm() -> dict[str, torch.Tensor]:
            return {"output": model.model.norm(final_hidden)}

        return Target(
            run_final_norm,
            {"input": final_hidden},
            {"weight": model.model.norm.weight},
            "transformers.LlamaRMSNorm.final_position_only",
        )

    if name == "lm_head":
        normalized = _host_random((batch_size, 1, hidden_size), 71)

        def run_lm_head() -> dict[str, torch.Tensor]:
            return {"logits": model.lm_head(normalized)}

        return Target(
            run_lm_head,
            {"input": normalized},
            {"weight": model.lm_head.weight},
            "transformers.LlamaForCausalLM.lm_head.final_position_only",
        )

    if name == "sampling":
        vocab_size = int(model.config.vocab_size)
        logits = _host_random((batch_size, 1, vocab_size), 81)

        def run_sampling() -> dict[str, torch.Tensor]:
            return {"token": torch.argmax(logits[:, -1, :], dim=-1)}

        return Target(
            run_sampling,
            {"logits": logits},
            {},
            "torch.argmax.greedy_sampling.final_position_only",
        )

    query = _host_random((batch_size, num_heads, context, head_dim), 21)
    key = _host_random((batch_size, num_kv_heads, context, head_dim), 22)
    value = _host_random((batch_size, num_kv_heads, context, head_dim), 23)

    if name == "rope":
        position_ids = torch.arange(context, dtype=torch.long).unsqueeze(0)
        if batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1).contiguous()
        position_ids = position_ids.cuda()

        def run_rope() -> dict[str, torch.Tensor]:
            cos, sin = model.model.rotary_emb(query, position_ids)
            query_out, key_out = apply_rotary_pos_emb(query, key, cos, sin)
            return {
                "cos": cos,
                "sin": sin,
                "query_output": query_out,
                "key_output": key_out,
            }

        return Target(
            run_rope,
            {"query": query, "key": key, "position_ids": position_ids},
            {},
            "transformers.LlamaRotaryEmbedding_plus_apply_rotary_pos_emb",
        )

    if name == "kv_append":
        key_cache = torch.empty_like(key)
        value_cache = torch.empty_like(value)

        def run_kv_append() -> dict[str, torch.Tensor]:
            key_cache.copy_(key)
            value_cache.copy_(value)
            return {"key_cache": key_cache, "value_cache": value_cache}

        return Target(
            run_kv_append,
            {"key": key, "value": value},
            {},
            "explicit_prefill_kv_cache_store",
        )

    if name == "causal_attention":
        repeated_key = repeat_kv(key, num_heads // num_kv_heads)
        repeated_value = repeat_kv(value, num_heads // num_kv_heads)

        def run_attention() -> dict[str, torch.Tensor]:
            output = F.scaled_dot_product_attention(
                query,
                repeated_key,
                repeated_value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=float(layer.self_attn.scaling),
            )
            return {"output": output}

        return Target(
            run_attention,
            {
                "query": query,
                "key": repeated_key,
                "value": repeated_value,
            },
            {},
            "torch.scaled_dot_product_attention_causal",
        )

    raise AssertionError(f"unhandled operator: {name}")


def _tensor_record(
    logical_name: str,
    tensor: torch.Tensor,
    operator: str,
    role: str,
) -> dict[str, object]:
    return {
        "tensor_id": f"tinyllama.layer0.{operator}.{logical_name}",
        "role": role,
        "address": int(tensor.data_ptr()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "layout": "strided",
        "alignment_bytes": 256,
    }


def main() -> None:
    args = _arguments()
    if args.context <= 0 or args.batch_size <= 0:
        raise ValueError("context and batch-size must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be unsigned")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
    ).cuda().eval()
    target = _target(args.operator, model, args.context, args.batch_size)
    profiler = ctypes.CDLL("libcuda.so.1") if args.driver_profiler else None

    with torch.inference_mode():
        for _ in range(args.warmup):
            target.run()
            torch.cuda.synchronize()
        allocation_snapshot: dict[str, object] | None = None
        if args.capture_allocator_history:
            torch.cuda.memory._record_memory_history(
                enabled="all", context=None, stacks="python", max_entries=1_000_000
            )
        try:
            if profiler is not None and profiler.cuProfilerStart() != 0:
                raise RuntimeError("cuProfilerStart failed")
            outputs = target.run()
            torch.cuda.synchronize()
            if profiler is not None and profiler.cuProfilerStop() != 0:
                raise RuntimeError("cuProfilerStop failed")
            if args.capture_allocator_history:
                allocation_snapshot = torch.cuda.memory._snapshot()
        finally:
            if args.capture_allocator_history:
                torch.cuda.memory._record_memory_history(enabled=None)

    records: list[dict[str, object]] = []
    seen_allocations: set[int] = set()
    for role, tensors in (
        ("input", target.inputs),
        ("parameter", target.parameters),
        ("output", outputs),
    ):
        for logical_name, tensor in tensors.items():
            address = int(tensor.data_ptr())
            if address in seen_allocations:
                continue
            seen_allocations.add(address)
            records.append(_tensor_record(logical_name, tensor, args.operator, role))

    allocator_ranges: tuple[tuple[int, int], ...] = ()
    if allocation_snapshot is not None:
        device = torch.cuda.current_device()
        event_ranges = allocator_ranges_from_events(
            allocation_snapshot["device_traces"][device]
        )
        preexisting_addresses = (
            int(tensor.data_ptr())
            for tensors in (target.inputs, target.parameters)
            for tensor in tensors.values()
        )
        backing_segments = allocator_segment_ranges_for_addresses(
            allocation_snapshot["segments"], preexisting_addresses
        )
        allocator_ranges = merge_address_ranges((*event_ranges, *backing_segments))

    revision_path = args.model.parent.parent / "refs" / "main"
    revision = revision_path.read_text(encoding="utf-8").strip()
    metadata = {
        "schema_version": "heterosim-exact-llm-operator/v2",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_spec_name": "TinyLlama-1.1B",
        "revision": revision,
        "operator": args.operator,
        "phase": "prefill",
        "layer_id": 0,
        "batch_size": args.batch_size,
        "context_length": args.context,
        "q_len": 1 if args.operator in FINAL_POSITION_OPERATORS else args.context,
        "kv_length": args.context,
        "dtype": "fp16",
        "implementation": target.implementation,
        "warmup_iterations": args.warmup,
        "capture_selector": (
            "cuda_driver_profiler_range" if args.driver_profiler else "process_target_only"
        ),
        "tensors": records,
        "capture_allocator": (
            {
                "source": (
                    "pytorch_cuda_caching_allocator_target_window_plus_"
                    "tensor_segments"
                ),
                "device": torch.cuda.current_device(),
                "ranges": [
                    {"address": begin, "size_bytes": end - begin}
                    for begin, end in allocator_ranges
                ],
            }
            if allocation_snapshot is not None
            else None
        ),
        "scope": "one_shape_locked_operator_not_end_to_end",
        "performance_eligible": False,
    }
    rendered = json.dumps(metadata, indent=2, sort_keys=True)
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
