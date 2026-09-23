#!/usr/bin/env python3
"""Run a deterministic real Hugging Face prefill plus one decode step."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def timed_cuda_call(function: Any) -> tuple[Any, float]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    value = function()
    end.record()
    torch.cuda.synchronize()
    return value, float(begin.elapsed_time(end))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--revision", default="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started

    prompt = (
        "The GPU ATLAS integration test uses a deterministic prompt and records "
        "one real prefill followed by one real decode step. "
    ) * 4
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids[:, : args.prompt_tokens].to("cuda")
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[1] != args.prompt_tokens:
        raise RuntimeError("the fixed prompt did not produce enough tokens")

    with torch.inference_mode():
        prefill, prefill_ms = timed_cuda_call(
            lambda: model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        )
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_mask = torch.ones(
            (input_ids.shape[0], input_ids.shape[1] + 1),
            dtype=attention_mask.dtype,
            device="cuda",
        )
        decode, decode_ms = timed_cuda_call(
            lambda: model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        )

    props = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": "hetero-framework-live-smoke/v1",
        "framework": "huggingface_transformers",
        "framework_version": transformers.__version__,
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(model.dtype),
        "device": {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": props.total_memory,
        },
        "workload": {
            "batch_size": int(input_ids.shape[0]),
            "prefill_tokens": int(input_ids.shape[1]),
            "decode_tokens": 1,
        },
        "execution": {
            "model_load_seconds": load_seconds,
            "prefill_cuda_ms": prefill_ms,
            "decode_cuda_ms": decode_ms,
            "input_ids_sha256": tensor_sha256(input_ids),
            "prefill_last_logits_sha256": tensor_sha256(prefill.logits[:, -1, :]),
            "decode_last_logits_sha256": tensor_sha256(decode.logits[:, -1, :]),
            "next_token_id": int(next_token.item()),
            "next_token_text": tokenizer.decode(next_token[0]),
            "prefill_logits_shape": list(prefill.logits.shape),
            "decode_logits_shape": list(decode.logits.shape),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "qualification": {
            "real_model_loaded": True,
            "real_prefill_executed": True,
            "real_decode_executed": True,
            "runtime_to_p28_event_adapter_connected": False,
            "performance_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
