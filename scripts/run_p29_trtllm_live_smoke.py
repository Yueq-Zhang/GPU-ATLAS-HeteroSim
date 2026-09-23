#!/usr/bin/env python3
"""Run one deterministic request through a real TensorRT-LLM engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import tensorrt_llm
import torch
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--revision", default="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    load_started = time.perf_counter()
    engine = LLM(
        model=args.model,
        tokenizer=tokenizer,
        revision=args.revision,
        dtype="float16",
        tensor_parallel_size=1,
        max_batch_size=1,
        max_input_len=63,
        max_seq_len=64,
        max_num_tokens=64,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25),
    )
    load_seconds = time.perf_counter() - load_started

    prompt = "GPU ATLAS deterministic TensorRT LLM runtime smoke test."
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    generation_started = time.perf_counter()
    request_output = engine.generate([prompt], params, use_tqdm=False)[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    completion = request_output.outputs[0]
    token_ids = [int(token) for token in completion.token_ids]
    token_hash = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    props = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": "hetero-framework-live-smoke/v1",
        "framework": "tensorrt_llm",
        "framework_version": tensorrt_llm.__version__,
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "model": args.model,
        "requested_revision": args.revision,
        "device": {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": props.total_memory,
        },
        "workload": {"request_count": 1, "max_seq_len": 64, "output_tokens": 1},
        "execution": {
            "engine_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "token_ids": token_ids,
            "token_ids_sha256": token_hash,
            "text": completion.text,
            "finished": bool(request_output.finished),
        },
        "qualification": {
            "real_engine_loaded": True,
            "real_request_executed": True,
            "runtime_event_adapter_connected": False,
            "performance_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
