#!/usr/bin/env python3
"""Measure all exact TinyLlama layer-0 GPU operators on a native CUDA GPU.

The benchmark deliberately reuses the operator builders used for NVBit trace
capture.  Token embedding and residual add are exceptions: their qualified P16
traces come from the shape-locked CUDA reference kernels, so their existing
native CUDA measurements are imported instead of silently substituting the
PyTorch implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.gpu_operator_calibration import (  # noqa: E402
    NATIVE_SCHEMA,
    exact_shape_key,
    file_sha256,
    load_gpu_operator_contracts,
)
from workloads.python.tinyllama_prefill_operator import _target  # noqa: E402


CUDA_REFERENCE_MEASUREMENTS = {
    "token_embedding": "token_embedding_ctx16_hidden2048",
    "residual_add": "residual_add_ctx16_hidden2048",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--revision",
        default="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    )
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=ROOT / "configs/hetero/operator_capabilities/"
        "tinyllama_prefill_layer0_bs1_ctx16.json",
    )
    parser.add_argument(
        "--simple-kernel-measurements",
        required=True,
        type=Path,
        help="P17 CUDA-reference measurements for embedding and residual add",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument(
        "--operators",
        nargs="*",
        help="Optional exact subset; default measures all 14 GPU operators",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, int]:
    if not values:
        raise ValueError("cannot summarize an empty measurement")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = round(fraction * (len(ordered) - 1))
        return round(ordered[index])

    return {
        "min": round(ordered[0]),
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": round(ordered[-1]),
        "mean": round(math.fsum(ordered) / len(ordered)),
    }


def _measure(target: Any, warmup: int, iterations: int) -> dict[str, int]:
    with torch.inference_mode():
        for _ in range(warmup):
            target.run()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        values_fs: list[float] = []
        for _ in range(iterations):
            start.record()
            outputs = target.run()
            stop.record()
            stop.synchronize()
            values_fs.append(float(start.elapsed_time(stop)) * 1.0e12)
        del outputs
    return _summary(values_fs)


def _load_simple_measurements(path: Path) -> tuple[Mapping[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("simple-kernel measurement root must be an object")
    if payload.get("schema_version") != "hetero-p17-native-calibration/v1":
        raise ValueError("invalid simple-kernel measurement schema")
    scope = payload.get("measurement_scope")
    if scope != "native_rtx3070_local_vram_not_external_3ddram":
        raise ValueError("simple-kernel measurement has an unexpected scope")
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("compute_capability") != "8.6":
        raise ValueError("simple-kernel measurements are not SM86")
    return payload, file_sha256(path)


def _import_cuda_reference(
    operator: str,
    payload: Mapping[str, object],
) -> tuple[dict[str, int], int]:
    protocol = payload.get("protocol")
    measurements = payload.get("measurements")
    if not isinstance(protocol, Mapping) or not isinstance(measurements, Mapping):
        raise ValueError("invalid simple-kernel measurement payload")
    source = measurements.get(CUDA_REFERENCE_MEASUREMENTS[operator])
    if not isinstance(source, Mapping):
        raise ValueError(f"missing CUDA reference measurement for {operator}")
    summary = {}
    for target_name, source_name in (
        ("min", "min_fs"),
        ("p10", "p10_fs"),
        ("median", "median_fs"),
        ("p90", "p90_fs"),
        ("max", "max_fs"),
        ("mean", "mean_fs"),
    ):
        value = source.get(source_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid {operator} {source_name}")
        summary[target_name] = round(value)
    repetitions = protocol.get("measured_iterations")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("invalid simple-kernel repetition count")
    return summary, repetitions


def _device_payload() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessors": properties.multi_processor_count,
        "global_memory_bytes": properties.total_memory,
    }


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _arguments()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be unsigned and iterations must be positive")
    if args.batch_size != 1 or args.context != 16:
        raise ValueError("P17 currently qualifies only BS=1, Context=16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    capability_path = args.capabilities.resolve()
    contracts = load_gpu_operator_contracts(capability_path, ROOT)
    selected = list(contracts) if args.operators is None else list(args.operators)
    if len(set(selected)) != len(selected):
        raise ValueError("operator selection contains duplicates")
    unknown = set(selected) - set(contracts)
    if unknown:
        raise ValueError(f"unknown GPU operators: {sorted(unknown)}")
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise ValueError(f"model config mismatch for {name}")
    if any(
        contract["checkpoint_revision"] != args.revision
        for contract in contracts.values()
    ):
        raise ValueError("capability artifacts do not match the requested revision")
    simple, simple_sha256 = _load_simple_measurements(
        args.simple_kernel_measurements.resolve()
    )
    payload: dict[str, object] = {
        "schema_version": NATIVE_SCHEMA,
        "catalog_id": "p17.tinyllama.layer0.bs1.ctx16.rtx3070.native_vram",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": {
            "memory_topology": "gpu_local_vram",
            "timing_owner": "physical_rtx3070_cuda_execution",
            "scope": "one_shape_locked_operator_not_end_to_end",
        },
        "device": _device_payload(),
        "software": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "model_contract": {
            "model_spec_name": "TinyLlama-1.1B",
            "checkpoint_revision": args.revision,
            "dtype": "fp16",
            "model_config_sha256": hashlib.sha256(
                (args.model / "config.json").read_bytes()
            ).hexdigest(),
        },
        "protocol": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "timer": "cuda_event_per_operator_iteration",
            "synchronization": "stop_event_synchronize_each_iteration",
            "statistic": "median",
        },
        "simple_kernel_source": {
            "path": str(args.simple_kernel_measurements.resolve()),
            "sha256": simple_sha256,
        },
        "required_operator_count": len(contracts),
        "selected_operators": selected,
        "operators": [],
        "operator_count": 0,
        "run_status": "in_progress",
        "performance_qualification": {
            "eligible": False,
            "reason": (
                "native local-VRAM measurements still require a matched "
                "native-memory Accel-Sim catalog"
            ),
        },
    }
    records = payload["operators"]
    assert isinstance(records, list)
    try:
        for operator in selected:
            if operator not in CUDA_REFERENCE_MEASUREMENTS:
                continue
            contract = contracts[operator]
            summary, repetitions = _import_cuda_reference(operator, simple)
            records.append(
                {
                    "operator_type": operator,
                    "implementation": contract["implementation"],
                    "shape_key": exact_shape_key(contract),
                    "operator_artifact": contract["artifact_locator"],
                    "operator_artifact_sha256": contract["artifact_sha256"],
                    "measurement_backend": "shape_locked_cuda_reference_kernel",
                    "trace_binary_identity": "unverified_new_native_execution",
                    "measurement_source_sha256": simple_sha256,
                    "measurement_software": simple.get("software"),
                    "operator_latency_fs": summary,
                    "repetitions": repetitions,
                    "statistic": "median",
                }
            )
        payload["operator_count"] = len(records)
        _write(args.output, payload)
        pytorch_operators = [
            item for item in selected if item not in CUDA_REFERENCE_MEASUREMENTS
        ]
        model = None
        if pytorch_operators:
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    local_files_only=True,
                    dtype=torch.float16,
                )
                .cuda()
                .eval()
            )
        for operator in pytorch_operators:
            assert model is not None
            contract = contracts[operator]
            target = _target(operator, model, args.context, args.batch_size)
            if target.implementation != contract["implementation"]:
                raise ValueError(
                    f"implementation mismatch for {operator}: "
                    f"{target.implementation} != {contract['implementation']}"
                )
            summary = _measure(target, args.warmup, args.iterations)
            records.append(
                {
                    "operator_type": operator,
                    "implementation": target.implementation,
                    "shape_key": exact_shape_key(contract),
                    "operator_artifact": contract["artifact_locator"],
                    "operator_artifact_sha256": contract["artifact_sha256"],
                    "measurement_backend": "pytorch_exact_capture_target_cuda_event",
                    "trace_binary_identity": "unverified_new_native_execution",
                    "measurement_software": {
                        "pytorch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                    },
                    "operator_latency_fs": summary,
                    "repetitions": args.iterations,
                    "statistic": "median",
                }
            )
            payload["operator_count"] = len(records)
            _write(args.output, payload)
            del target
            torch.cuda.empty_cache()
        payload["run_status"] = (
            "complete" if len(records) == len(selected) else "partial"
        )
        _write(args.output, payload)
    except Exception as error:
        payload["run_status"] = "failed"
        payload["failure"] = f"{type(error).__name__}: {error}"
        payload["operator_count"] = len(records)
        _write(args.output, payload)
        raise
    print(
        f"P17 native GPU operator catalog written: {args.output} "
        f"({payload['operator_count']}/{len(selected)} selected operators)"
    )


if __name__ == "__main__":
    main()
