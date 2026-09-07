#!/usr/bin/env python3
"""Execute one exact TinyLlama Decode operator for native timing or NVBit capture.

The query length is always one. ``--kv-length`` is the attention-visible length
after appending the current token, so the four P20 steps use 17, 18, 19 and 20.
Each invocation executes one layer-0 operator and emits allocation metadata that
keeps TraceAddr, TensorID + offset and later Global PA binding separate.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from frontend.hetero.capture_allocation_ranges import (
    allocator_ranges_from_events,
    allocator_segment_ranges_for_addresses,
    merge_address_ranges,
)
from workloads.python.tinyllama_prefill_operator import (
    LIGHTWEIGHT_OPERATORS,
    Target,
    _host_random,
    _lightweight_target,
    _measure_native,
    _target as _prefill_target,
)


SUPPORTED_OPERATORS = (
    "token_embedding",
    "attention_norm",
    "qkv_projection",
    "rope",
    "causal_attention",
    "output_projection",
    "residual_add",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
)
CHECKPOINT_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--operator", required=True, choices=SUPPORTED_OPERATORS)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--kv-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--driver-profiler", action="store_true")
    parser.add_argument("--capture-allocator-history", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--native-measurement-output", type=Path)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _launch_program_identity() -> tuple[str, dict[str, object]]:
    python_executable = Path(sys.executable).resolve()
    workload_source = Path(__file__).resolve()
    shared_source = Path(_prefill_target.__code__.co_filename).resolve()
    torch_extension = Path(torch._C.__file__).resolve()
    components: dict[str, object] = {
        "program_kind": "python_pytorch_launch_program",
        "python_executable": str(python_executable),
        "python_executable_sha256": _file_sha256(python_executable),
        "workload_source": str(workload_source),
        "workload_source_sha256": _file_sha256(workload_source),
        "shared_operator_source": str(shared_source),
        "shared_operator_source_sha256": _file_sha256(shared_source),
        "torch_extension": str(torch_extension),
        "torch_extension_sha256": _file_sha256(torch_extension),
        "python_version": sys.version.split()[0],
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    return _canonical_sha256(components), components


def _decode_target(name: str, model: Any, kv_length: int, batch_size: int) -> Target:
    """Build q_len=1 Decode work while preserving the exact checkpoint weights."""

    if name not in {"rope", "causal_attention"}:
        return _prefill_target(name, model, 1, batch_size)

    layer = model.model.layers[0]
    head_dim = int(model.config.head_dim)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    query = _host_random((batch_size, num_heads, 1, head_dim), 21)

    if name == "rope":
        key = _host_random((batch_size, num_kv_heads, 1, head_dim), 22)
        position_ids = torch.full(
            (batch_size, 1), kv_length - 1, dtype=torch.long, device="cuda"
        )

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
            "transformers.LlamaRotaryEmbedding_decode_position_plus_apply_rotary_pos_emb",
        )

    key_cache = _host_random(
        (batch_size, num_kv_heads, kv_length, head_dim), 22
    )
    value_cache = _host_random(
        (batch_size, num_kv_heads, kv_length, head_dim), 23
    )
    repeated_key = repeat_kv(key_cache, num_heads // num_kv_heads)
    repeated_value = repeat_kv(value_cache, num_heads // num_kv_heads)

    def run_attention() -> dict[str, torch.Tensor]:
        output = F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=float(layer.self_attn.scaling),
        )
        return {"output": output}

    return Target(
        run_attention,
        {"query": query, "key": repeated_key, "value": repeated_value},
        {},
        "torch.scaled_dot_product_attention_decode_full_kv_visibility",
    )


def _tensor_record(
    logical_name: str,
    tensor: torch.Tensor,
    operator: str,
    kv_length: int,
    role: str,
) -> dict[str, object]:
    return {
        "tensor_id": (
            f"tinyllama.decode.kv{kv_length}.layer0.{operator}.{logical_name}"
        ),
        "role": role,
        "address": int(tensor.data_ptr()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "layout": "strided",
        "alignment_bytes": 256,
    }


def _checkpoint_revision(model_path: Path, fallback: str) -> str:
    revision_path = model_path.parent.parent / "refs" / "main"
    return (
        revision_path.read_text(encoding="utf-8").strip()
        if revision_path.is_file()
        else fallback
    )


def main() -> None:
    args = _arguments()
    if args.context <= 0 or args.kv_length <= 0 or args.batch_size <= 0:
        raise ValueError("context, kv-length and batch-size must be positive")
    if args.kv_length <= args.context:
        raise ValueError("Decode kv-length must exceed the initial context")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be unsigned and iterations positive")
    if args.native_measurement_output and (
        args.driver_profiler or args.capture_allocator_history
    ):
        raise ValueError(
            "native measurement cannot enable profiler or allocator capture"
        )

    if args.operator in LIGHTWEIGHT_OPERATORS:
        target = _lightweight_target(args.operator, args.model, 1, args.batch_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.float16,
        ).cuda().eval()
        target = _decode_target(args.operator, model, args.kv_length, args.batch_size)

    revision = _checkpoint_revision(args.model, args.checkpoint_revision)
    if revision != args.checkpoint_revision:
        raise ValueError(
            f"checkpoint revision {revision} does not match {args.checkpoint_revision}"
        )
    launch_sha256, program_components = _launch_program_identity()
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    target_sm = device.major * 10 + device.minor

    if args.native_measurement_output:
        with torch.inference_mode():
            summary = _measure_native(target, args.warmup, args.iterations)
        measurement = {
            "schema_version": "hetero-p21-native-decode-operator/v1",
            "model_spec_name": "TinyLlama-1.1B",
            "checkpoint_revision": revision,
            "operator_type": args.operator,
            "implementation": target.implementation,
            "phase": "decode_step",
            "batch_size": args.batch_size,
            "context_length": args.context,
            "q_len": 1,
            "kv_length": args.kv_length,
            "dtype": "fp16",
            "device": {
                "name": device.name,
                "compute_capability": f"{device.major}.{device.minor}",
                "multiprocessors": device.multi_processor_count,
                "global_memory_bytes": device.total_memory,
            },
            "software": {
                "python": sys.version.split()[0],
                "pytorch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "protocol": {
                "warmup_iterations": args.warmup,
                "measured_iterations": args.iterations,
                "timer": "cuda_event_per_iteration",
                "synchronization": "stop_event_synchronize_each_iteration",
                "statistic": "median",
            },
            "launch": {
                "program_kind": "python_pytorch_launch_program",
                "launch_program_sha256": launch_sha256,
                "target_sm": target_sm,
                "program_components": program_components,
            },
            "measurement": summary,
            "measurement_scope": "native_target_gpu_local_vram",
            "performance_eligible": False,
        }
        rendered = json.dumps(measurement, indent=2, sort_keys=True)
        args.native_measurement_output.parent.mkdir(parents=True, exist_ok=True)
        args.native_measurement_output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return

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
            records.append(
                _tensor_record(
                    logical_name, tensor, args.operator, args.kv_length, role
                )
            )

    allocator_ranges: tuple[tuple[int, int], ...] = ()
    if allocation_snapshot is not None:
        device_index = torch.cuda.current_device()
        event_ranges = allocator_ranges_from_events(
            allocation_snapshot["device_traces"][device_index]
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

    metadata = {
        "schema_version": "heterosim-exact-llm-operator/v2",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_spec_name": "TinyLlama-1.1B",
        "revision": revision,
        "operator": args.operator,
        "phase": "decode_step",
        "layer_id": 0,
        "batch_size": args.batch_size,
        "context_length": args.context,
        "q_len": 1,
        "kv_length": args.kv_length,
        "past_kv_length": args.kv_length - 1,
        "dtype": "fp16",
        "implementation": target.implementation,
        "compilation": {
            "framework": "pytorch",
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "target_sm": target_sm,
            "launch_program": {
                "kind": "python_pytorch_launch_program",
                "sha256": launch_sha256,
                "components": program_components,
            },
        },
        "warmup_iterations": args.warmup,
        "capture_selector": (
            "cuda_driver_profiler_range" if args.driver_profiler else "process_target_only"
        ),
        "tensors": records,
        "capture_allocator": (
            {
                "source": (
                    "pytorch_cuda_caching_allocator_target_window_plus_tensor_segments"
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
        "scope": "one_exact_decode_shape_locked_operator_not_end_to_end",
        "performance_eligible": False,
    }
    rendered = json.dumps(metadata, indent=2, sort_keys=True)
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
