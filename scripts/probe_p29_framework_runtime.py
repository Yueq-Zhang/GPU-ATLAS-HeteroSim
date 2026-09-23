#!/usr/bin/env python3
"""Emit a machine-readable import and CUDA probe for one P29 runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def torch_probe() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    result: dict[str, Any] = {
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
    }
    if available:
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "device_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
            }
        )
        tensor = torch.arange(16, device="cuda", dtype=torch.float32)
        result["cuda_tensor_sum"] = float(tensor.sum().item())
    return result


def probe(profile: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "hetero-framework-runtime-probe/v1",
        "profile": profile,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_probe(),
    }
    if profile == "huggingface":
        result["packages"] = {
            "transformers": module_version("transformers"),
            "accelerate": module_version("accelerate"),
            "huggingface_hub": module_version("huggingface_hub"),
        }
        from transformers import LlamaConfig

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
        )
        result["framework_object_probe"] = config.model_type
    elif profile == "vllm":
        result["packages"] = {"vllm": module_version("vllm")}
        from vllm import SamplingParams

        result["framework_object_probe"] = type(
            SamplingParams(temperature=0.0, max_tokens=1)
        ).__name__
    elif profile == "tensorrt_llm":
        result["packages"] = {"tensorrt_llm": module_version("tensorrt_llm")}
        import tensorrt_llm

        result["framework_object_probe"] = tensorrt_llm.__name__
    else:
        raise ValueError(f"unsupported profile: {profile}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("huggingface", "vllm", "tensorrt_llm"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = probe(args.profile)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
