#!/usr/bin/env python3
"""Qualify the versioned HF/vLLM/TensorRT-LLM shadow adapter contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from frontend.hetero.inference_framework import (
    build_framework_export,
    build_shadow_simulation_record,
    compile_atlas_tensor_ir,
    model_spec_from_huggingface,
    normalize_tensorrt_llm_execution,
    normalize_vllm_scheduler_events,
    resolve_gpu_ready_artifacts,
)
from frontend.hetero.model_graph import RequestSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hetero/frameworks/p28_tinyllama_bs2_decode_shadow.json"
CATALOG = ROOT / "validation/p23/ready_catalog.json"
OUTPUT = ROOT / "validation/p28/framework_shadow/qualification_record.json"


def _run(config: dict[str, object]) -> dict[str, object]:
    model = model_spec_from_huggingface(
        config["huggingface_config"],  # type: ignore[arg-type]
        model_name=str(config["model_name"]),
        revision=str(config["checkpoint_revision"]),
    )
    requests = [RequestSpec(**item) for item in config["requests"]]  # type: ignore[arg-type]
    export = build_framework_export(
        model,
        requests,
        framework=str(config["framework"]),
        framework_version=str(config["framework_version"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
        adapter_version=str(config["adapter_version"]),
        execution_parameters=config["execution_parameters"],  # type: ignore[arg-type]
        capacity_bytes=1 << 40,
    )
    paged_kv = config["paged_kv"]
    vllm = normalize_vllm_scheduler_events(
        config["vllm_events"],  # type: ignore[arg-type]
        page_size_tokens=int(paged_kv["page_size_tokens"]),  # type: ignore[index]
        page_size_bytes=int(paged_kv["page_size_bytes"]),  # type: ignore[index]
        global_pa_base=int(paged_kv["global_pa_base"]),  # type: ignore[index]
    )
    tensorrt = normalize_tensorrt_llm_execution(
        config["tensorrt_llm_manifest"]  # type: ignore[arg-type]
    )
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    gpu_artifacts = resolve_gpu_ready_artifacts(
        catalog,
        model_name=model.name,
        checkpoint_revision=str(config["checkpoint_revision"]),
        phase="decode_step",
        batch_size=2,
        context_length=16,
        q_len=1,
        kv_length=17,
        operators=["attention_norm", "qkv_projection"],
    )
    atlas = compile_atlas_tensor_ir(
        operator="qkv_projection",
        model=model,
        tokens=1,
        core_count=16,
        tile_m=1,
        tile_k=512,
        tile_n=16,
    )
    observations = [
        {"observation_id": "decode-bs2.attention_norm", "batch_id": "decode-bs2"},
        {"observation_id": "decode-bs2.qkv_projection", "batch_id": "decode-bs2"},
    ]
    simulation = [
        {
            "observation_id": item["observation_id"],
            "gpu_artifact": gpu_artifacts[str(item["observation_id"]).split(".")[-1]],
        }
        for item in observations
    ]
    shadow = build_shadow_simulation_record(export, observations, simulation)
    return {
        "framework_export": export,
        "vllm": vllm,
        "tensorrt_llm": tensorrt,
        "gpu_artifacts": gpu_artifacts,
        "atlas_tensor_ir": atlas,
        "shadow": shadow,
    }


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = _run(config)
    second = _run(config)
    deterministic = first == second
    artifact_summary = {
        "framework_identity_sha256": first["framework_export"]["identity_sha256"],
        "graph_count": len(first["framework_export"]["graphs"]),
        "task_count": len(first["framework_export"]["execution_graph"]["tasks"]),
        "global_pa_allocation_count": first["framework_export"]["global_memory_map"][
            "allocation_count"
        ],
        "vllm_event_sha256": first["vllm"]["event_sha256"],
        "vllm_batch_count": len(first["vllm"]["batches"]),
        "paged_kv_global_pa_bound": first["vllm"]["paged_kv_global_pa_bound"],
        "tensorrt_identity_sha256": first["tensorrt_llm"]["identity_sha256"],
        "gpu_artifacts": first["gpu_artifacts"],
        "atlas_compile_sha256": first["atlas_tensor_ir"]["compile_sha256"],
        "shadow_record_sha256": first["shadow"]["record_sha256"],
        "full_artifact_sha256": hashlib.sha256(
            json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    record = {
        "schema_version": "hetero-p28-framework-shadow-qualification/v1",
        "status": "passed" if deterministic else "failed",
        "deterministic_double_run": deterministic,
        "capabilities": {
            "huggingface_graph_and_tensor_export": True,
            "tensor_to_global_pa": True,
            "exact_gpu_catalog_resolution": True,
            "atlas_tensor_ir_compile": True,
            "shadow_simulation": True,
            "vllm_continuous_ragged_paged_kv_contract": bool(
                first["vllm"]["continuous_batching_observed"]
                and first["vllm"]["ragged_batching_observed"]
                and first["vllm"]["paged_kv_global_pa_bound"]
            ),
            "tensorrt_llm_identity_contract": True,
        },
        "external_runtime_validation": {
            "huggingface_live_model_executed": False,
            "vllm_live_scheduler_executed": False,
            "tensorrt_llm_engine_executed": False,
        },
        "performance_claim_allowed": False,
        "artifact_summary": artifact_summary,
        "qualification_boundary": (
            "The offline, versioned framework adapters and shadow causality are "
            "qualified. Live external framework callbacks, a real TensorRT engine, "
            "and end-to-end hardware performance remain unqualified."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("status", "capabilities", "external_runtime_validation")}, indent=2))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
