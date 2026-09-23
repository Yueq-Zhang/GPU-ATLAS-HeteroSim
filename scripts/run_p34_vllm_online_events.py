#!/usr/bin/env python3
"""Observe real vLLM SchedulerOutput and Paged-KV block tables in-process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

# Scheduler hooks must live in the same process as EngineCore.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# P29 installs and pins the model revision before online qualification.  Do not
# let a transient Hub connection turn a local replay into an unbounded network
# operation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler


def _class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _block_groups(value: object) -> list[list[int]] | None:
    if value is None:
        return None
    return [[int(block) for block in group] for group in value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--revision", default="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    scheduler_steps: list[dict[str, object]] = []
    engine_geometry: dict[str, object] = {}
    original_schedule = Scheduler.schedule

    def observed_schedule(self, *schedule_args, **schedule_kwargs):
        result = original_schedule(self, *schedule_args, **schedule_kwargs)
        if not engine_geometry:
            tensors = [
                {
                    "size_bytes": int(item.size),
                    "layer_count": len(item.layers),
                    "layer_stride": int(item.layer_stride),
                    "block_stride": int(item.block_stride),
                    "offset": int(item.offset),
                }
                for item in self.kv_cache_config.kv_cache_tensors
            ]
            num_blocks = int(self.kv_cache_config.num_blocks)
            engine_geometry.update(
                {
                    "scheduler_class": _class_name(self),
                    "multiprocess_engine_core": False,
                    "block_size_tokens": int(self.block_size),
                    "num_gpu_blocks": num_blocks,
                    "kv_cache_group_count": len(self.kv_cache_config.kv_cache_groups),
                    "kv_cache_tensors": tensors,
                    "bytes_per_block": sum(
                        int(item["size_bytes"]) // num_blocks for item in tensors
                    ),
                    "max_num_seqs": int(self.max_num_running_reqs),
                    "max_num_scheduled_tokens": int(self.max_num_scheduled_tokens),
                }
            )
        cached = result.scheduled_cached_reqs
        scheduler_steps.append(
            {
                "sequence": len(scheduler_steps),
                "new_requests": [
                    {
                        "request_id": str(item.req_id),
                        "prompt_tokens": int(item.prompt_len),
                        "block_ids": _block_groups(item.block_ids),
                        "num_computed_tokens": int(item.num_computed_tokens),
                    }
                    for item in result.scheduled_new_reqs
                ],
                "cached_requests": {
                    "request_ids": [str(item) for item in cached.req_ids],
                    "new_block_ids": [
                        _block_groups(item) for item in cached.new_block_ids
                    ],
                    "num_computed_tokens": [
                        int(item) for item in cached.num_computed_tokens
                    ],
                    "num_output_tokens": [
                        int(item) for item in cached.num_output_tokens
                    ],
                },
                "batch": {
                    "request_ids": [str(item) for item in result.num_scheduled_tokens],
                    "token_counts": [
                        int(item) for item in result.num_scheduled_tokens.values()
                    ],
                    "total_num_scheduled_tokens": int(
                        result.total_num_scheduled_tokens
                    ),
                },
                "finished_request_ids": sorted(
                    str(item) for item in result.finished_req_ids
                ),
                "new_block_ids_to_zero": sorted(
                    int(item) for item in (result.new_block_ids_to_zero or [])
                ),
            }
        )
        return result

    Scheduler.schedule = observed_schedule
    try:
        engine = LLM(
            model=args.model,
            revision=args.revision,
            dtype="float16",
            tensor_parallel_size=1,
            max_model_len=64,
            max_num_seqs=2,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            seed=0,
        )
        prompts = [
            "GPU ATLAS vLLM short request.",
            "GPU ATLAS vLLM deliberately longer ragged request for scheduling.",
        ]
        params = [
            SamplingParams(temperature=0.0, max_tokens=1),
            SamplingParams(temperature=0.0, max_tokens=2),
        ]
        outputs = engine.generate(prompts, params, use_tqdm=False)
    finally:
        Scheduler.schedule = original_schedule

    request_ids = sorted(
        {
            str(item["request_id"])
            for step in scheduler_steps
            for item in step["new_requests"]
        }
    )
    token_ids = [
        [int(token) for token in item.outputs[0].token_ids] for item in outputs
    ]
    props = torch.cuda.get_device_properties(0)
    result = {
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "texts": [item.outputs[0].text for item in outputs],
    }
    payload = {
        "schema_version": "hetero-vllm-live-observation/v1",
        "framework": "vllm",
        "framework_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "model": args.model,
        "requested_revision": args.revision,
        "device": {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "engine": {
            "engine_class": _class_name(engine),
            **engine_geometry,
        },
        "workload": {
            "request_count": 2,
            "prompt_text_sha256": hashlib.sha256(
                json.dumps(prompts, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "max_output_tokens": [1, 2],
        },
        "scheduler_steps": scheduler_steps,
        "finished_request_ids": request_ids,
        "result": result,
        "qualification": {
            "real_engine_loaded": True,
            "real_scheduler_callback_connected": True,
            "real_block_table_observed": True,
            "framework_output_used_for_finish": True,
            "performance_claim_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scheduler_steps": len(scheduler_steps),
                "request_ids": request_ids,
                "token_ids": token_ids,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
