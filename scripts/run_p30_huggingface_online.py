#!/usr/bin/env python3
"""Observe a real Hugging Face Prefill/Decode request and bind it to P28."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from frontend.hetero.capture_allocation_ranges import (
    allocator_ranges_from_events,
    merge_address_ranges,
)
from frontend.hetero.framework_runtime import (
    bind_huggingface_runtime_to_export,
    normalize_huggingface_runtime_observation,
)
from frontend.hetero.inference_framework import (
    build_framework_export,
    model_spec_from_huggingface,
)
from frontend.hetero.model_graph import RequestSpec


MODULES = {
    "model.embed_tokens": ("token_embedding", "token_embedding"),
    "model.layers.0": ("transformer_layer", "layer0"),
    "model.layers.0.input_layernorm": ("attention_norm", "attention_norm"),
    "model.layers.0.self_attn": ("causal_attention", "self_attention"),
    "model.layers.0.self_attn.q_proj": ("q_projection", "qkv_projection"),
    "model.layers.0.self_attn.k_proj": ("k_projection", "qkv_projection"),
    "model.layers.0.self_attn.v_proj": ("v_projection", "qkv_projection"),
    "model.layers.0.self_attn.o_proj": ("output_projection", "output_projection"),
    "model.layers.0.post_attention_layernorm": ("mlp_norm", "mlp_norm"),
    "model.layers.0.mlp": ("mlp", "mlp"),
    "model.layers.0.mlp.gate_proj": ("gate_projection", "gate_up_projection"),
    "model.layers.0.mlp.up_proj": ("up_projection", "gate_up_projection"),
    "model.layers.0.mlp.down_proj": ("down_projection", "down_projection"),
    "model.norm": ("final_norm", "final_norm"),
    "lm_head": ("lm_head", "lm_head"),
}


def tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _flatten_tensors(
    value: object, prefix: str = "value"
) -> list[tuple[str, torch.Tensor]]:
    result: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_flatten_tensors(item, f"{prefix}.{key}"))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            result.extend(_flatten_tensors(item, f"{prefix}.{index}"))
        return result
    if hasattr(value, "items"):
        try:
            for key, item in value.items():  # type: ignore[union-attr]
                result.extend(_flatten_tensors(item, f"{prefix}.{key}"))
        except (AttributeError, TypeError):
            pass
    return result


def _tensor_record(
    tensor: torch.Tensor,
    *,
    tensor_id: str,
    role: str,
    path: str,
) -> dict[str, object] | None:
    if tensor.device.type != "cuda":
        return None
    storage = tensor.untyped_storage()
    item_size = tensor.element_size()
    return {
        "tensor_id": tensor_id,
        "role": role,
        "path": path,
        "address": int(tensor.data_ptr()),
        "storage_base": int(storage.data_ptr()),
        "storage_handle": int(storage._cdata),
        "storage_size_bytes": int(storage.nbytes()),
        "storage_offset_bytes": int(tensor.storage_offset()) * item_size,
        "logical_nbytes": int(tensor.numel()) * item_size,
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "item_size_bytes": item_size,
    }


class RuntimeObserver:
    def __init__(
        self,
        request_id: str,
        profiler_phase: str | None,
        nvbit_instrumentation_phase: str | None,
    ) -> None:
        if profiler_phase and nvbit_instrumentation_phase:
            raise RuntimeError("CUDA profiler and direct NVBit controls are exclusive")
        self.request_id = request_id
        self.profiler_phase = profiler_phase
        self.nvbit_instrumentation_phase = nvbit_instrumentation_phase
        self.phase: str | None = None
        self.events: list[dict[str, object]] = []
        self.handles: list[Any] = []
        self.cuda_driver = ctypes.CDLL("libcuda.so.1") if profiler_phase else None
        self.nvbit_tracer = None
        if nvbit_instrumentation_phase:
            tracer = os.environ.get("CUDA_INJECTION64_PATH") or os.environ.get(
                "ACCEL_SIM_TRACER"
            )
            if not tracer:
                raise RuntimeError(
                    "direct NVBit control requires a tracer library path"
                )
            # Resolve the symbols from the already preloaded tracer.  Opening
            # the path again can create an independent tool instance whose
            # enable flag does not control the NVBit callbacks doing capture.
            process_symbols = ctypes.CDLL(None)
            try:
                process_symbols.enable_nvbit_instrumentation
                process_symbols.disable_nvbit_instrumentation
                process_symbols.set_nvbit_instrumentation_tag
                self.nvbit_tracer = process_symbols
            except AttributeError:
                # CUDA_INJECTION64_PATH loads the tracer through the driver;
                # NVBit's official PyTorch hook resolves that injected instance
                # by opening the same path after the CUDA context exists.
                self.nvbit_tracer = ctypes.CDLL(tracer)
            self.nvbit_tracer.enable_nvbit_instrumentation.restype = None
            self.nvbit_tracer.disable_nvbit_instrumentation.restype = None
            self.nvbit_tracer.set_nvbit_instrumentation_tag.restype = None
            self.nvbit_tracer.set_nvbit_instrumentation_tag.argtypes = [ctypes.c_char_p]
            self.nvbit_tracer.disable_nvbit_instrumentation()
        self.profiler_active = False

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append(
            {
                "sequence": len(self.events),
                "event": kind,
                "host_time_ns": time.perf_counter_ns(),
                **fields,
            }
        )

    def _records(
        self,
        module_path: str,
        role: str,
        values: object,
        module: torch.nn.Module,
    ) -> list[dict[str, object]]:
        sequence = len(self.events)
        records: list[dict[str, object]] = []
        seen: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
        sources = _flatten_tensors(values, role)
        if role == "input":
            sources.extend(
                (f"parameter.{name}", parameter)
                for name, parameter in module.named_parameters(recurse=False)
            )
        for index, (path, tensor) in enumerate(sources):
            key = (int(tensor.data_ptr()), tuple(tensor.shape), tuple(tensor.stride()))
            if key in seen:
                continue
            seen.add(key)
            record = _tensor_record(
                tensor,
                tensor_id=(
                    f"{self.request_id}.{self.phase}.{module_path}."
                    f"event{sequence:04d}.{role}.{index:03d}"
                ),
                role="parameter" if path.startswith("parameter.") else role,
                path=path,
            )
            if record is not None:
                records.append(record)
        return records

    def _start_profiler(self, module_path: str) -> None:
        if (
            self.nvbit_tracer is not None
            and self.phase == self.nvbit_instrumentation_phase
            and module_path == "model.layers.0"
        ):
            if self.profiler_active:
                raise RuntimeError("NVBit instrumentation is already active")
            tag = f"{self.phase}_model_layers_0".encode("ascii")
            self.nvbit_tracer.set_nvbit_instrumentation_tag(tag)
            self.nvbit_tracer.enable_nvbit_instrumentation()
            self.profiler_active = True
            return
        if (
            self.cuda_driver is not None
            and self.phase == self.profiler_phase
            and module_path == "model.layers.0"
        ):
            if self.profiler_active or self.cuda_driver.cuProfilerStart() != 0:
                raise RuntimeError("cuProfilerStart failed")
            self.profiler_active = True

    def _stop_profiler(self, module_path: str) -> None:
        if self.profiler_active and module_path == "model.layers.0":
            torch.cuda.synchronize()
            if self.nvbit_tracer is not None:
                self.nvbit_tracer.disable_nvbit_instrumentation()
            else:
                assert self.cuda_driver is not None
                if self.cuda_driver.cuProfilerStop() != 0:
                    raise RuntimeError("cuProfilerStop failed")
            self.profiler_active = False

    def install(self, model: torch.nn.Module) -> None:
        named = dict(model.named_modules())
        missing = sorted(set(MODULES) - set(named))
        if missing:
            raise RuntimeError(f"required Hugging Face modules are absent: {missing}")
        for module_path, (logical_operator, fusion_group) in MODULES.items():
            module = named[module_path]

            def pre_hook(
                current: torch.nn.Module,
                args: tuple[object, ...],
                kwargs: dict[str, object],
                *,
                path: str = module_path,
                logical: str = logical_operator,
                fusion: str = fusion_group,
            ) -> None:
                if self.phase is None:
                    return
                self._start_profiler(path)
                self.emit(
                    "module_begin",
                    phase=self.phase,
                    module_path=path,
                    module_class=type(current).__name__,
                    logical_operator=logical,
                    fusion_group=fusion,
                    tensors=self._records(path, "input", (args, kwargs), current),
                )

            def post_hook(
                current: torch.nn.Module,
                args: tuple[object, ...],
                kwargs: dict[str, object],
                output: object,
                *,
                path: str = module_path,
                logical: str = logical_operator,
                fusion: str = fusion_group,
            ) -> None:
                if self.phase is None:
                    return
                self.emit(
                    "module_end",
                    phase=self.phase,
                    module_path=path,
                    module_class=type(current).__name__,
                    logical_operator=logical,
                    fusion_group=fusion,
                    tensors=self._records(path, "output", output, current),
                )
                self._stop_profiler(path)

            self.handles.append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )
            self.handles.append(
                module.register_forward_hook(post_hook, with_kwargs=True)
            )

    def begin_phase(self, phase: str, q_len: int, kv_length: int) -> None:
        if self.phase is not None:
            raise RuntimeError("nested Hugging Face phase")
        self.phase = phase
        self.emit("phase_begin", phase=phase, q_len=q_len, kv_length=kv_length)

    def end_phase(self) -> None:
        if self.phase is None or self.profiler_active:
            raise RuntimeError("Hugging Face phase/profiler is incomplete")
        phase = self.phase
        self.emit("phase_end", phase=phase)
        self.phase = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _run_request(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    observer: RuntimeObserver | None,
) -> tuple[Any, Any, torch.Tensor]:
    if observer is not None:
        observer.begin_phase("prefill", input_ids.shape[1], input_ids.shape[1])
    prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    if observer is not None:
        observer.end_phase()
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_mask = torch.ones(
        (input_ids.shape[0], input_ids.shape[1] + 1),
        dtype=attention_mask.dtype,
        device=input_ids.device,
    )
    if observer is not None:
        observer.emit("sampling", token_id=int(next_token.item()))
        observer.begin_phase("decode_step", 1, input_ids.shape[1] + 1)
    decode = model(
        input_ids=next_token,
        attention_mask=decode_mask,
        past_key_values=prefill.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    if observer is not None:
        observer.end_phase()
    return prefill, decode, next_token


def _allocator_ranges(
    snapshot: Mapping[str, object], device: int
) -> list[dict[str, int]]:
    traces = snapshot.get("device_traces")
    segments = snapshot.get("segments")
    if not isinstance(traces, Sequence) or not isinstance(segments, Sequence):
        raise RuntimeError("PyTorch allocator snapshot is incomplete")
    event_ranges = allocator_ranges_from_events(traces[device])  # type: ignore[arg-type]
    segment_ranges = (
        (int(item["address"]), int(item["address"]) + int(item["total_size"]))
        for item in segments
        if isinstance(item, Mapping) and int(item.get("device", -1)) == device
    )
    return [
        {"address": begin, "size_bytes": end - begin}
        for begin, end in merge_address_ranges((*event_ranges, *segment_ranges))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--revision", default="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )
    parser.add_argument("--request-id", default="p30-hf-r0")
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--driver-profiler-phase", choices=("prefill", "decode_step"), default=None
    )
    parser.add_argument(
        "--nvbit-instrumentation-phase",
        choices=("prefill", "decode_step"),
        default=None,
    )
    parser.add_argument("--capture-allocator-history", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.prompt_tokens <= 0:
        raise RuntimeError("CUDA and a positive prompt length are required")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    prompt = (
        "The GPU ATLAS online adapter observes a deterministic Hugging Face request. "
    ) * 4
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids[:, : args.prompt_tokens].to("cuda")
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[1] != args.prompt_tokens:
        raise RuntimeError("the deterministic prompt is too short")

    with torch.inference_mode():
        _run_request(model, input_ids, attention_mask, None)
        torch.cuda.synchronize()
    observer = RuntimeObserver(
        args.request_id,
        args.driver_profiler_phase,
        args.nvbit_instrumentation_phase,
    )
    observer.install(model)
    observer.emit("request_arrive", request_id=args.request_id)
    allocation_snapshot: Mapping[str, object] | None = None
    if args.capture_allocator_history:
        torch.cuda.memory._record_memory_history(
            enabled="all", context=None, stacks="python", max_entries=1_000_000
        )
    try:
        with torch.inference_mode():
            prefill, decode, next_token = _run_request(
                model, input_ids, attention_mask, observer
            )
            torch.cuda.synchronize()
        if args.capture_allocator_history:
            allocation_snapshot = torch.cuda.memory._snapshot()
    finally:
        if args.capture_allocator_history:
            torch.cuda.memory._record_memory_history(enabled=None)
        observer.close()
    observer.emit("request_finish", request_id=args.request_id)

    properties = torch.cuda.get_device_properties(0)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != args.revision:
        raise RuntimeError("resolved Hugging Face revision does not match the request")
    raw: dict[str, object] = {
        "schema_version": "hetero-huggingface-runtime-observation/v1",
        "framework": {
            "name": "huggingface",
            "version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "attention_implementation": getattr(
                model.config, "_attn_implementation", "unknown"
            ),
        },
        "model": {"name": args.model, "revision": resolved_revision},
        "request": {
            "request_id": args.request_id,
            "batch_size": int(input_ids.shape[0]),
            "prompt_tokens": int(input_ids.shape[1]),
            "decode_tokens": 1,
            "dtype": "fp16",
        },
        "device": {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        },
        "events": observer.events,
        "result": {
            "input_ids_sha256": tensor_sha256(input_ids),
            "prefill_logits_sha256": tensor_sha256(prefill.logits[:, -1, :]),
            "decode_logits_sha256": tensor_sha256(decode.logits[:, -1, :]),
            "next_token_id": int(next_token.item()),
            "next_token_text": tokenizer.decode(next_token[0]),
        },
        "capture_allocator": (
            {
                "source": "pytorch_cuda_allocator_snapshot_full_runtime_window",
                "ranges": _allocator_ranges(allocation_snapshot, 0),
            }
            if allocation_snapshot is not None
            else None
        ),
        "capture": {
            "nvbit_profiler_phase": (
                args.driver_profiler_phase or args.nvbit_instrumentation_phase
            ),
            "capture_control": (
                "cuda_driver_profiler"
                if args.driver_profiler_phase
                else (
                    "nvbit_exported_instrumentation_api"
                    if args.nvbit_instrumentation_phase
                    else "none"
                )
            ),
            "cuda_profiler_range_emitted": bool(args.driver_profiler_phase),
            "nvbit_instrumentation_range_emitted": bool(
                args.nvbit_instrumentation_phase
            ),
            "nvbit_trace_generated": False,
        },
    }
    normalized = normalize_huggingface_runtime_observation(raw)
    spec = model_spec_from_huggingface(
        model.config.to_dict(), model_name=args.model, revision=args.revision
    )
    export = build_framework_export(
        spec,
        [
            RequestSpec(
                args.request_id,
                prompt_length=args.prompt_tokens,
                output_length=1,
                execution_scope="full_request",
            )
        ],
        framework="huggingface",
        framework_version=transformers.__version__,
        tokenizer_revision=args.revision,
        adapter_version="heterosim-huggingface-runtime-adapter/v1",
        execution_parameters={
            "dtype": "fp16",
            "use_cache": True,
            "attention_implementation": getattr(
                model.config, "_attn_implementation", "unknown"
            ),
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": 1,
        },
    )
    binding = bind_huggingface_runtime_to_export(export, normalized)
    bundle = {
        "schema_version": "hetero-p30-huggingface-online-bundle/v1",
        "raw_observation": raw,
        "normalized_runtime": normalized,
        "framework_export": export,
        "online_binding": binding,
        "qualification": {
            "actual_huggingface_request_executed": True,
            "actual_runtime_callbacks_observed": True,
            "tensor_alias_and_global_pa_bound": True,
            "p28_simulation_key_connected": True,
            "sass_trace_generated": False,
            "performance_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "event_count": len(observer.events),
                "module_count": len(normalized["identity"]["module_sequence"]),
                "storage_count": len(normalized["memory_map"]["storages"]),
                "simulation_key": binding["simulation_key"],
                "next_token_id": int(next_token.item()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
