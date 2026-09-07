import json
from pathlib import Path

import pytest

from frontend.hetero.batching import BatchPlanError, build_batch_plan
from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec, RequestSpec, build_request_graph
from frontend.hetero.multi_batch_runtime import build_multi_batch_runtime
from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config

MODEL = ModelSpec(
    name="tiny",
    hidden_size=128,
    intermediate_size=256,
    num_layers=1,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=32,
    vocab_size=256,
)
PLACEMENT = {
    "mode": "rule_based",
    "unit": "device_subbatch_operator",
    "default_target": "gpu0",
    "rules": [],
}


def _decode_graph():
    return build_request_graph(
        MODEL,
        RequestSpec(
            "R0",
            prompt_length=1,
            output_length=2,
            execution_scope="decode_loop",
            initial_kv_length=16,
        ),
    )


def _scheduler_result(kv_lengths: tuple[int, ...]) -> dict[str, object]:
    request_ids = tuple(f"R{index}" for index in range(len(kv_lengths)))
    return {
        "schema_version": "hetero-runtime-result/v1",
        "epochs": [
            {
                "epoch_id": 0,
                "boundary_time_fs": 0,
                "completion_time_fs": 1000,
                "admitted_request_ids": list(request_ids),
                "active_request_ids": list(request_ids),
                "retired_request_ids": list(request_ids),
                "selections": [
                    {
                        "request_id": request_id,
                        "phase": "decode",
                        "token_begin": kv_len,
                        "token_count": 1,
                    }
                    for request_id, kv_len in zip(request_ids, kv_lengths)
                ],
            }
        ],
        "requests": [
            {
                "request_id": request_id,
                "generated_length": 1,
                "committed_kv_length": kv_len + 1,
                "token_ready_time_fs": [1000],
                "finish_time_fs": 1000,
            }
            for request_id, kv_len in zip(request_ids, kv_lengths)
        ],
    }


def test_static_homogeneous_bs2_has_bijective_member_fanout() -> None:
    plan = build_batch_plan(
        _scheduler_result((16, 16)),
        _decode_graph().nodes,
        PLACEMENT,
        {
            "mode": "request_batch",
            "batch_policy": "homogeneous",
            "batch_cycle_mode": "request_cycle_composed",
        },
        model_dtype="fp16",
    )
    assert plan["schema_version"] == "hetero-batch-plan/v2"
    assert plan["conservation"]["member_fanout_is_bijective_per_subbatch"]
    assert {len(item["request_ids"]) for item in plan["device_subbatches"]} == {2}
    assert all(not item["request_cycle_ready"] for item in plan["device_subbatches"])
    assert plan["performance_claim_allowed"] is False


def test_bs1_runtime_is_equivalent_to_scheduler_result() -> None:
    scheduler = _scheduler_result((16,))
    plan = build_batch_plan(
        scheduler,
        _decode_graph().nodes,
        PLACEMENT,
        {"batch_policy": "homogeneous"},
        model_dtype="fp16",
    )
    result = build_multi_batch_runtime(
        scheduler,
        plan,
        [
            {
                "request_id": "R0",
                "prompt_length": 1,
                "output_length": 1,
                "execution_scope": "decode_loop",
                "initial_kv_length": 16,
            }
        ],
    )
    assert result["requests"][0]["token_ready_time_fs"] == [1000]
    assert result["requests"][0]["final_committed_kv_length"] == 17
    assert result["metrics"]["max_device_subbatch_size"] == 1


def test_ragged_padding_and_split_are_explicit() -> None:
    scheduler = _scheduler_result((16, 31))
    padded = build_batch_plan(
        scheduler,
        _decode_graph().nodes,
        PLACEMENT,
        {"batch_policy": "padding_dense"},
        model_dtype="fp16",
    )
    split = build_batch_plan(
        scheduler,
        _decode_graph().nodes,
        PLACEMENT,
        {"batch_policy": "ragged_split"},
        model_dtype="fp16",
    )
    assert {len(item["request_ids"]) for item in padded["device_subbatches"]} == {2}
    assert padded["padded_attention_tokens"] > padded["effective_attention_tokens"]
    assert {len(item["request_ids"]) for item in split["device_subbatches"]} == {1}
    with pytest.raises(BatchPlanError, match="incompatible shapes"):
        build_batch_plan(
            scheduler,
            _decode_graph().nodes,
            PLACEMENT,
            {"batch_policy": "homogeneous"},
            model_dtype="fp16",
        )


def test_batched_kernel_cycle_fails_closed_without_exact_artifact() -> None:
    with pytest.raises(BatchPlanError, match="exact batch artifact catalog"):
        build_batch_plan(
            _scheduler_result((16, 16)),
            _decode_graph().nodes,
            PLACEMENT,
            {
                "batch_policy": "homogeneous",
                "batch_cycle_mode": "batched_kernel_cycle",
            },
            model_dtype="fp16",
        )


def test_exact_batched_kernel_artifact_can_be_request_cycle_ready() -> None:
    node = ModelNode(
        node_id="decode.l0.attention",
        kind=NodeKind.COMPUTE,
        op="causal_attention",
        phase=Phase.DECODE,
        layer_id=0,
        step_id=0,
        attributes={"operator_group": "attention"},
    )
    catalog = {
        "schema_version": "hetero-batched-kernel-catalog/v1",
        "entries": [
            {
                "artifact_id": "exact.bs2.ctx16",
                "phase": "decode",
                "layer_id": 0,
                "op": "causal_attention",
                "device_id": "gpu0",
                "dtype": "fp16",
                "batch_size": 2,
                "q_lengths": [1, 1],
                "kv_lengths": [17, 17],
                "request_cycle_ready": True,
                "performance_eligible": False,
            }
        ],
    }
    plan = build_batch_plan(
        _scheduler_result((16, 16)),
        [node],
        PLACEMENT,
        {
            "batch_policy": "homogeneous",
            "batch_cycle_mode": "batched_kernel_cycle",
        },
        model_dtype="fp16",
        batch_artifact_catalog=catalog,
    )
    assert plan["device_subbatches"][0]["artifact_id"] == "exact.bs2.ctx16"
    assert plan["device_subbatches"][0]["request_cycle_ready"] is True
    assert plan["performance_claim_allowed"] is False


def test_device_subbatch_placement_splits_gpu_and_atlas() -> None:
    placement = {
        **PLACEMENT,
        "rules": [
            {
                "match": {"phase": "decode", "operator_group": "attention"},
                "target": "atlas0.compute",
            }
        ],
    }
    plan = build_batch_plan(
        _scheduler_result((16, 16)),
        _decode_graph().nodes,
        placement,
        {"batch_policy": "homogeneous"},
        model_dtype="fp16",
    )
    assert {item["device_id"] for item in plan["device_subbatches"]} == {
        "gpu0",
        "atlas0.compute",
    }


def test_request_lifecycle_versions_and_global_pa_are_isolated() -> None:
    scheduler = _scheduler_result((16, 16))
    plan = build_batch_plan(
        scheduler,
        _decode_graph().nodes,
        PLACEMENT,
        {"batch_policy": "homogeneous"},
        model_dtype="fp16",
    )
    lifecycle = {
        "events": [
            {
                "time_fs": 0,
                "operation": "allocate",
                "allocation_id": "R0.kv",
                "memory_space_id": "shared0.dram3d",
                "offset_bytes": 0,
                "allocation_epoch": 1,
                "size_bytes": 4096,
            },
            {
                "time_fs": 0,
                "operation": "allocate",
                "allocation_id": "R1.kv",
                "memory_space_id": "shared0.dram3d",
                "offset_bytes": 4096,
                "allocation_epoch": 2,
                "size_bytes": 4096,
            },
            {
                "time_fs": 1000,
                "operation": "release",
                "allocation_id": "R0.kv",
                "memory_space_id": "shared0.dram3d",
                "offset_bytes": 0,
                "allocation_epoch": 1,
                "size_bytes": 4096,
            },
            {
                "time_fs": 1000,
                "operation": "release",
                "allocation_id": "R1.kv",
                "memory_space_id": "shared0.dram3d",
                "offset_bytes": 4096,
                "allocation_epoch": 2,
                "size_bytes": 4096,
            },
        ],
        "memory_spaces": [
            {"memory_space_id": "shared0.dram3d", "used_bytes": 0}
        ],
    }
    requests = [
        {
            "request_id": request_id,
            "prompt_length": 1,
            "output_length": 1,
            "execution_scope": "decode_loop",
            "initial_kv_length": 16,
        }
        for request_id in ("R0", "R1")
    ]
    result = build_multi_batch_runtime(scheduler, plan, requests, lifecycle)
    assert result["conservation"]["admitted_requests"] == 2
    assert result["conservation"]["retired_requests"] == 2
    assert result["conservation"]["zero_in_flight"]
    assert result["memory_isolation"]["active_range_overlap_count"] == 0
    assert {item["final_kv_version"] for item in result["requests"]} == {1}


@pytest.mark.parametrize(
    "config_name,expected_requests",
    [
        ("p22_tinyllama_decode4_1layer_static_bs2.json", 2),
        ("p22_tinyllama_decode4_22layer_continuous_bs4.json", 4),
        ("p22_tinyllama_2layer_mixed_continuous_bs4.json", 4),
    ],
)
def test_p22_runner_emits_reproducible_multi_batch_contract(
    tmp_path: Path, config_name: str, expected_requests: int
) -> None:
    config = load_and_validate_config(f"configs/hetero/experiments/{config_name}")
    first = execute_run(config, Path.cwd(), tmp_path / "leg1")
    second = execute_run(config, Path.cwd(), tmp_path / "leg2")
    first_payload = json.loads((first / "multi_batch_runtime.json").read_text())
    second_payload = json.loads((second / "multi_batch_runtime.json").read_text())
    assert first_payload == second_payload
    assert len(first_payload["requests"]) == expected_requests
    assert first_payload["conservation"]["all_requests_finished"]
    assert first_payload["conservation"]["zero_in_flight"]
    assert first_payload["performance_claim_allowed"] is False
    assert first_payload["memory_isolation"]["zero_bytes_after_retirement"]
