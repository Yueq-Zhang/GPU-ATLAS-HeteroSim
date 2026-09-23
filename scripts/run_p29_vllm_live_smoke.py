#!/usr/bin/env python3
"""Run one deterministic request through a real vLLM engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

# vLLM's V2 GPU model runner requires UVA host buffers. NVIDIA CUDA under WSL2
# exposes the GPU but not that host-memory path, so use the supported V1 fallback.
if "microsoft" in platform.release().lower():
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch
import vllm
from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--revision", default="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    started = time.perf_counter()
    engine = LLM(
        model=args.model,
        revision=args.revision,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=64,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=0,
    )
    load_seconds = time.perf_counter() - started
    params = SamplingParams(temperature=0.0, max_tokens=1)
    prompt = "GPU ATLAS deterministic vLLM runtime smoke test."
    torch.cuda.synchronize()
    generation_started = time.perf_counter()
    outputs = engine.generate([prompt], params)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    token_ids = list(outputs[0].outputs[0].token_ids)
    text = outputs[0].outputs[0].text
    token_hash = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    props = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": "hetero-framework-live-smoke/v1",
        "framework": "vllm",
        "framework_version": vllm.__version__,
        "model_runner": (
            "v1_wsl_uva_fallback"
            if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0"
            else "framework_default"
        ),
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
        "workload": {"request_count": 1, "max_model_len": 64, "output_tokens": 1},
        "execution": {
            "engine_load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "token_ids": token_ids,
            "token_ids_sha256": token_hash,
            "text": text,
        },
        "qualification": {
            "real_engine_loaded": True,
            "real_request_executed": True,
            "scheduler_event_adapter_connected": False,
            "block_table_adapter_connected": False,
            "performance_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
