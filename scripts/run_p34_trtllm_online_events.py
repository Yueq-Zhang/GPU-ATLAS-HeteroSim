#!/usr/bin/env python3
"""Observe a real TensorRT-LLM LLM API engine, profile and scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Keep TP=1 execution in the caller so the Python scheduler callback is the
# scheduler that actually admits the measured requests, rather than a proxy in
# front of an unobservable worker process.
os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

import tensorrt_llm
import torch
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.scheduler import SimpleScheduler
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from transformers import AutoTokenizer


def _class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__name__}"


def _request_id(request: object) -> str:
    for name in ("py_request_id", "request_id"):
        if hasattr(request, name):
            return str(getattr(request, name))
    raise RuntimeError("TensorRT-LLM request has no observable ID")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument(
        "--revision", default="fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    observed_steps: list[dict[str, object]] = []
    scheduler_classes: set[str] = set()
    original_schedule = SimpleScheduler.schedule_request

    def observed_schedule(self, active_requests, inflight_request_ids):
        result = original_schedule(self, active_requests, inflight_request_ids)
        scheduler_classes.add(_class_name(self))
        scheduled = [*result.context_requests, *result.generation_requests]
        observed_steps.append(
            {
                "event": "scheduler_step",
                "request_ids": [_request_id(item) for item in scheduled],
                "context_request_ids": [
                    _request_id(item) for item in result.context_requests
                ],
                "generation_request_ids": [
                    _request_id(item) for item in result.generation_requests
                ],
                "paused_request_ids": [
                    _request_id(item) for item in result.paused_requests
                ],
                "active_request_states": {
                    _request_id(item): str(item.state) for item in active_requests
                },
            }
        )
        return result

    SimpleScheduler.schedule_request = observed_schedule
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.revision, local_files_only=True
        )
        engine = LLM(
            model=args.model,
            tokenizer=tokenizer,
            revision=args.revision,
            dtype="float16",
            tensor_parallel_size=1,
            max_batch_size=2,
            max_input_len=63,
            max_seq_len=64,
            max_num_tokens=128,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.25),
        )
        prompts = [
            "GPU ATLAS TensorRT LLM request A.",
            "GPU ATLAS TensorRT LLM longer request B for the live profile.",
        ]
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        outputs = engine.generate(prompts, params, use_tqdm=False)
    finally:
        SimpleScheduler.schedule_request = original_schedule

    if not isinstance(outputs, list):
        raise TypeError("TensorRT-LLM batched output is not a list")
    token_ids = [
        [int(token) for token in item.outputs[0].token_ids] for item in outputs
    ]
    request_ids = [_request_id(item) for item in outputs]
    args_profile = engine.args
    executor = engine._executor
    executor_engine = getattr(executor, "engine", None)
    scheduler = getattr(executor_engine, "scheduler", None)
    if scheduler is not None:
        scheduler_class = _class_name(scheduler)
    elif scheduler_classes:
        scheduler_class = min(scheduler_classes)
    else:
        raise RuntimeError("TensorRT-LLM scheduler callback was not observed")
    events = [
        {
            "event": "engine_initialized",
            "engine_class": _class_name(engine),
            "executor_class": _class_name(executor),
        },
        *observed_steps,
        {
            "event": "request_complete",
            "request_ids": request_ids,
            "finished": [bool(item.finished) for item in outputs],
        },
    ]
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    props = torch.cuda.get_device_properties(0)
    payload = {
        "schema_version": "hetero-tensorrt-llm-live-observation/v1",
        "framework": "tensorrt_llm",
        "framework_version": tensorrt_llm.__version__,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "model": args.model,
        "requested_revision": args.revision,
        "device": {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "engine": {
            "backend": "pytorch",
            "engine_class": _class_name(engine),
            "executor_class": _class_name(executor),
            "scheduler_class": scheduler_class,
            "profile": {
                "max_batch_size": int(args_profile.max_batch_size),
                "max_input_len": int(args_profile.max_input_len),
                "max_seq_len": int(args_profile.max_seq_len),
                "max_num_tokens": int(args_profile.max_num_tokens),
            },
            "serialized_engine_sha256": None,
        },
        "events": events,
        "result": {
            "finished": all(bool(item.finished) for item in outputs),
            "token_ids": token_ids,
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "texts": [item.outputs[0].text for item in outputs],
        },
        "qualification": {
            "real_engine_object_observed": True,
            "real_profile_observed": True,
            "real_scheduler_callback_connected": True,
            "serialized_tensorrt_engine_qualified": False,
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
                "scheduler_steps": len(observed_steps),
                "token_ids": token_ids,
                "backend": "pytorch",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
