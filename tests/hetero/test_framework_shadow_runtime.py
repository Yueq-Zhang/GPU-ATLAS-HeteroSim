from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from frontend.hetero.framework_live_adapters import (
    FrameworkLiveAdapterError,
    normalize_tensorrt_llm_live_observation,
    normalize_vllm_live_observation,
    qualify_live_framework_pair,
)
from frontend.hetero.framework_shadow_runtime import (
    FrameworkShadowRuntimeError,
    build_p32_online_shadow_timeline,
    qualify_p33_cycle_replays,
    run_atlas_cycle_replay,
)
from frontend.hetero.live_ramulator2 import LiveRamulator2Bridge


class _FakeBridge:
    def __init__(self, project_root: Path, config: dict[str, object]) -> None:
        del project_root, config
        self.current_cycle = 0
        self.global_time_fs = 0
        self._pending: list[dict[str, int]] = []
        self.accepted = 0
        self.reads = 0
        self.writes = 0

    def send(
        self,
        parent_id: int,
        global_address: int,
        size_bytes: int,
        operation: str,
        initiator: int,
        ordering_domain: int,
        sequence_number: int,
    ) -> int:
        del global_address, size_bytes, ordering_domain, sequence_number
        assert initiator == LiveRamulator2Bridge.ATLAS_INITIATOR
        self.accepted += 1
        self.reads += operation == "read"
        self.writes += operation == "write"
        self._pending.append(
            {
                "parent_id": parent_id,
                "initiator": initiator,
                "operation_code": operation == "write",
                "total_children": 1,
                "completion_cycle": parent_id,
                "completion_time_fs": parent_id * 10,
            }
        )
        return LiveRamulator2Bridge.SEND_ACCEPTED

    def pop_completions(self) -> list[dict[str, int]]:
        result, self._pending = self._pending, []
        return result

    def advance_until_event(self, cycles: int) -> int:
        self.current_cycle += cycles
        self.global_time_fs += cycles * 10
        return cycles

    def close(self) -> dict[str, object]:
        return {
            "instances": 1,
            "outstanding": 0,
            "accepted_parent_ids": self.accepted,
            "observed_completion_ids": self.accepted,
            "durable_completed": self.accepted,
            "children_sent": self.accepted,
            "children_completed": self.accepted,
            "reads": self.reads,
            "writes": self.writes,
            "gpu_cycles": max(self.accepted, 1),
            "global_time_fs": max(self.accepted, 1) * 10,
            "initiators": {
                "gpu0": {"parents": 0, "completed": 0, "children": 0},
                "atlas0.compute": {
                    "parents": self.accepted,
                    "completed": self.accepted,
                    "children": self.accepted,
                },
            },
        }


def test_atlas_trace_replays_through_one_durable_owner(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl.gz"
    requests = [
        {
            "sequence": 0,
            "command": "read",
            "core_id": 0,
            "global_address": 0x2000,
            "size_bytes": 64,
            "tensor": "activation",
        },
        {
            "sequence": 1,
            "command": "read",
            "core_id": 0,
            "global_address": 0x3000,
            "size_bytes": 64,
            "tensor": "weight",
        },
        {
            "sequence": 2,
            "command": "write",
            "core_id": 0,
            "global_address": 0x4000,
            "size_bytes": 64,
            "tensor": "output",
        },
    ]
    with gzip.open(trace, "wt", encoding="utf-8") as stream:
        for item in requests:
            stream.write(json.dumps(item) + "\n")
    result = run_atlas_cycle_replay(tmp_path, {}, trace, bridge_factory=_FakeBridge)
    assert result["request_count"] == 3
    assert result["read_requests"] == 2
    assert result["write_requests"] == 1
    assert result["durable_completion_qualified"] is True


def _gpu_stats() -> dict[str, object]:
    return {
        "cycles": 100,
        "instructions": 200,
        "duration_fs": 300,
        "simulation_key": "gpu-sim-key",
        "external_memory_stats": {
            "instances": 1,
            "reads": 2,
            "writes": 1,
            "gpu_parents": 3,
            "gpu_completed": 3,
            "completed": 3,
            "durable_completed": 3,
            "children_sent": 4,
            "children_completed": 4,
            "address_unmapped": 0,
            "address_translated": 10,
            "atlas_parents": 0,
            "outstanding": 0,
        },
    }


def _atlas_replay() -> dict[str, object]:
    stats = {
        "instances": 1,
        "global_time_fs": 400,
        "outstanding": 0,
        "durable_completed": 3,
    }
    return {
        "schema_version": "hetero-p33-atlas-cycle-replay/v1",
        "trace_file_sha256": "a" * 64,
        "request_stream_sha256": "b" * 64,
        "completion_stream_sha256": "c" * 64,
        "request_count": 3,
        "read_requests": 2,
        "write_requests": 1,
        "logical_bytes": 192,
        "first_completion": {"parent_id": 1},
        "last_completion": {"parent_id": 3},
        "ramulator2": stats,
        "single_ramulator2": True,
        "durable_completion_qualified": True,
    }


def test_p33_double_replay_and_p32_non_additive_timeline() -> None:
    p33 = qualify_p33_cycle_replays(
        [_gpu_stats(), _gpu_stats()],
        [_atlas_replay(), _atlas_replay()],
        framework_simulation_key="framework-key",
    )
    atlas_artifact = {
        "schema_version": "hetero-atlas-executable-artifact/v1",
        "artifact_key": "atlas-key",
        "identity": {
            "tensor_global_pa": {
                "activation": 1 << 33,
                "weight": (1 << 33) + 4096,
                "output": (1 << 33) + 8192,
            }
        },
        "source_tensor_ir": {
            "inputs": [
                {"shape": [1, 16], "dtype": "fp16"},
                {"shape": [16, 16], "dtype": "fp16"},
            ],
            "outputs": [{"shape": [1, 16], "dtype": "fp16"}],
        },
    }
    record = build_p32_online_shadow_timeline(
        {"status": "passed", "workload": {"batch_size": 1}},
        {
            "status": "passed",
            "p30_simulation_key": "framework-key",
            "gpu": {"artifact_id": "gpu-layer0"},
        },
        p33,
        {
            "schema_version": "hetero-online-address-binding/v1",
            "bindings": [{"physical_offset_bytes": 0, "size_bytes": 4096}],
        },
        atlas_artifact,
    )
    assert record["qualification"]["end_to_end_framework_timeline_qualified"]
    assert record["qualification"]["single_global_time_owner"]
    assert (
        record["events"][-1]["time_fs"]
        == record["global_time"]["observation_barrier_fs"]
    )
    assert record["artifact_semantics"]["qkv_double_count_prevented"]
    assert not record["artifact_semantics"]["mixed_placement_makespan_qualified"]


def _vllm_observation() -> dict[str, object]:
    return {
        "schema_version": "hetero-vllm-live-observation/v1",
        "framework": "vllm",
        "framework_version": "0.29.0",
        "model": "fixture",
        "requested_revision": "0" * 40,
        "device": {"name": "fixture", "compute_capability": [8, 9]},
        "engine": {
            "engine_class": "vllm.entrypoints.llm.LLM",
            "scheduler_class": "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
            "multiprocess_engine_core": False,
            "block_size_tokens": 16,
            "bytes_per_block": 4096,
            "num_gpu_blocks": 128,
            "max_num_seqs": 2,
        },
        "workload": {"request_count": 2},
        "scheduler_steps": [
            {
                "sequence": 0,
                "new_requests": [
                    {"request_id": "0", "block_ids": [[3]]},
                    {"request_id": "1", "block_ids": [[4]]},
                ],
                "cached_requests": {"request_ids": [], "new_block_ids": []},
                "batch": {"request_ids": ["0", "1"], "token_counts": [4, 7]},
            },
            {
                "sequence": 1,
                "new_requests": [],
                "cached_requests": {
                    "request_ids": ["1"],
                    "new_block_ids": [None],
                },
                "batch": {"request_ids": ["1"], "token_counts": [1]},
            },
        ],
        "finished_request_ids": ["0", "1"],
        "result": {"token_ids": [[1], [2, 3]], "texts": ["a", "bc"]},
    }


def _trt_observation() -> dict[str, object]:
    return {
        "schema_version": "hetero-tensorrt-llm-live-observation/v1",
        "framework": "tensorrt_llm",
        "framework_version": "1.2.1",
        "model": "fixture",
        "requested_revision": "0" * 40,
        "device": {"name": "fixture", "compute_capability": [8, 9]},
        "engine": {
            "backend": "pytorch",
            "engine_class": "tensorrt_llm.llmapi.llm.LLM",
            "executor_class": "tensorrt_llm.executor.Executor",
            "scheduler_class": "tensorrt_llm.scheduler.SimpleScheduler",
            "profile": {
                "max_batch_size": 2,
                "max_input_len": 63,
                "max_seq_len": 64,
                "max_num_tokens": 128,
            },
            "serialized_engine_sha256": None,
        },
        "events": [
            {"sequence": 0, "event": "engine_initialized"},
            {
                "sequence": 1,
                "event": "scheduler_step",
                "request_ids": ["0", "1"],
            },
            {
                "sequence": 2,
                "event": "request_complete",
                "request_ids": ["0", "1"],
            },
        ],
        "result": {"finished": True, "token_ids": [[1], [2]]},
    }


def test_live_vllm_and_tensorrt_adapters_are_fail_closed() -> None:
    vllm = normalize_vllm_live_observation(_vllm_observation())
    tensorrt = normalize_tensorrt_llm_live_observation(_trt_observation())
    assert vllm["real_block_table_observed"]
    assert vllm["continuous_batching_observed"]
    assert vllm["ragged_batching_observed"]
    assert tensorrt["real_profile_observed"]
    assert not tensorrt["serialized_tensorrt_engine_qualified"]
    qualified = qualify_live_framework_pair(
        [_vllm_observation(), _vllm_observation_with_runtime_ids()],
        [_trt_observation(), copy.deepcopy(_trt_observation())],
    )
    assert qualified["status"] == "passed"

    broken = _vllm_observation()
    broken["scheduler_steps"][0]["new_requests"][1]["block_ids"] = [[3]]
    with pytest.raises(FrameworkLiveAdapterError, match="aliased"):
        normalize_vllm_live_observation(broken)


def _vllm_observation_with_runtime_ids() -> dict[str, object]:
    observation = copy.deepcopy(_vllm_observation())
    aliases = {"0": "0-random", "1": "1-random"}
    for step in observation["scheduler_steps"]:
        for request in step["new_requests"]:
            request["request_id"] = aliases[request["request_id"]]
        step["cached_requests"]["request_ids"] = [
            aliases[item] for item in step["cached_requests"]["request_ids"]
        ]
        step["batch"]["request_ids"] = [
            aliases[item] for item in step["batch"]["request_ids"]
        ]
    observation["finished_request_ids"] = [
        aliases[item] for item in observation["finished_request_ids"]
    ]
    return observation


def test_p33_rejects_gpu_nondeterminism() -> None:
    second = _gpu_stats()
    second["cycles"] = 101
    with pytest.raises(FrameworkShadowRuntimeError, match="mismatch"):
        qualify_p33_cycle_replays(
            [_gpu_stats(), second],
            [_atlas_replay(), _atlas_replay()],
            framework_simulation_key="framework-key",
        )
