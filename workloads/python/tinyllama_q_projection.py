#!/usr/bin/env python3
"""Run one real TinyLlama layer-0 Q-projection for NVBit capture.

Model loading and one warm-up execute before the target matrix multiplication.
The qualified RTX 3070 trace selects the stable dynamic kernel range ``4-5``;
``--driver-profiler`` remains available for NVBit versions that honor CUDA
driver profiler ranges.  This is an exact LLM operator, not an end-to-end
model trace.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--driver-profiler", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context <= 0:
        raise ValueError("context must be positive")
    rows = 1 if args.phase == "decode" else args.context
    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
    ).cuda().eval()
    projection = model.model.layers[0].self_attn.q_proj
    if projection.bias is not None:
        raise RuntimeError("this exact workload expects TinyLlama attention_bias=false")
    weight = projection.weight.detach()
    hidden = torch.randn((rows, weight.shape[1]), device="cuda", dtype=torch.float16)
    output = torch.empty((rows, weight.shape[0]), device="cuda", dtype=torch.float16)
    cuda_driver = ctypes.CDLL("libcuda.so.1") if args.driver_profiler else None

    with torch.inference_mode():
        torch.mm(hidden, weight.t(), out=output)
        torch.cuda.synchronize()
        if cuda_driver is not None and cuda_driver.cuProfilerStart() != 0:
            raise RuntimeError("cuProfilerStart failed")
        torch.mm(hidden, weight.t(), out=output)
        torch.cuda.synchronize()
        if cuda_driver is not None and cuda_driver.cuProfilerStop() != 0:
            raise RuntimeError("cuProfilerStop failed")

    revision_path = args.model.parent.parent / "refs" / "main"
    revision = revision_path.read_text(encoding="utf-8").strip()
    metadata = {
        "schema_version": "heterosim-exact-llm-operator/v1",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": revision,
        "operator": "model.layers.0.self_attn.q_proj",
        "phase": args.phase,
        "batch_size": 1,
        "context_length": args.context,
        "m": rows,
        "n": int(weight.shape[0]),
        "k": int(weight.shape[1]),
        "dtype": "fp16",
        "tensors": {
            "input": {"address": hidden.data_ptr(), "bytes": hidden.numel() * 2},
            "weight": {"address": weight.data_ptr(), "bytes": weight.numel() * 2},
            "output": {"address": output.data_ptr(), "bytes": output.numel() * 2},
        },
        "scope": "one_exact_operator_not_end_to_end",
        "capture_selector": (
            "cuda_driver_profiler_range" if args.driver_profiler else "nvbit_dynamic_kernel_range_4_5"
        ),
    }
    rendered = json.dumps(metadata, indent=2, sort_keys=True)
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
