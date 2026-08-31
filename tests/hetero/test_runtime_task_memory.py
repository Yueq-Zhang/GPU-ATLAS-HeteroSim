from pathlib import Path

from frontend.hetero.global_memory_map import GlobalAllocation
from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec
from frontend.hetero.runtime_task_memory import (
    RuntimeTaskAddressBinding,
    plan_runtime_task_requests,
    run_runtime_task_memory,
)
from frontend.hetero.runtime_task_model import RuntimeTaskModelCatalog


CATALOG = Path(
    "configs/hetero/runtime_tasks/"
    "tinyllama_prefill_layer0_bs1_ctx16_uncalibrated.json"
)


def _model() -> ModelSpec:
    return ModelSpec(
        "TinyLlama-1.1B",
        2048,
        5632,
        1,
        32,
        4,
        64,
        32000,
        dtype="fp16",
        checkpoint_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    )


def _kv_node() -> ModelNode:
    return ModelNode(
        "r.prefill.s0.l0.attention.kv_append",
        NodeKind.STATE,
        "kv_append",
        Phase.PREFILL,
        0,
        0,
        attributes={
            "batch_size": 1,
            "context_length": 16,
            "q_len": 16,
            "past_kv_len": 0,
            "attention_kv_len": 16,
        },
    )


def _allocation(value_id: str, base: int, size: int) -> GlobalAllocation:
    return GlobalAllocation(
        value_id,
        "shared0.dram3d",
        base,
        size,
        64,
        "activation",
        "fp16",
    )


def test_kv_append_lowers_exact_packed_kv_slices_to_global_pa() -> None:
    catalog = RuntimeTaskModelCatalog.load(CATALOG)
    node = _kv_node()
    contract = catalog.contract_for(node)
    estimate = catalog.estimate(node, _model())
    inputs = (
        {"value_id": "positioned", "version": 1, "size_bytes": 81_920},
        {"value_id": "key", "version": 0, "size_bytes": 8_192},
        {"value_id": "value", "version": 0, "size_bytes": 8_192},
    )
    outputs = (
        {"value_id": "query", "version": 1, "size_bytes": 65_536},
        {"value_id": "key", "version": 1, "size_bytes": 8_192},
        {"value_id": "value", "version": 1, "size_bytes": 8_192},
    )
    allocations = {
        "positioned": _allocation("positioned", 0x100000, 81_920),
        "key": _allocation("key", 0x200000, 8_192),
        "value": _allocation("value", 0x300000, 8_192),
        "query": _allocation("query", 0x400000, 65_536),
    }
    requests = plan_runtime_task_requests(
        node,
        _model(),
        contract,
        estimate,
        RuntimeTaskAddressBinding("task.kv", inputs, outputs),
        allocations,
    )
    assert len(requests) == 512
    assert sum(item["size_bytes"] for item in requests[:256]) == 16_384
    assert sum(item["size_bytes"] for item in requests[256:]) == 16_384
    assert requests[0]["global_address"] == 0x100000 + 65_536
    assert requests[128]["global_address"] == 0x100000 + 65_536 + 8_192
    assert requests[256]["global_address"] == 0x200000
    assert requests[384]["global_address"] == 0x300000
    assert all(item["operation"] == "read" for item in requests[:256])
    assert all(item["operation"] == "write" for item in requests[256:])


class _FakeBridge:
    GPU_INITIATOR = 0
    SEND_ACCEPTED = 1

    def __init__(self, _root: Path, _config: object) -> None:
        self.current_cycle = 0
        self.global_time_fs = 0
        self.accepted: list[int] = []
        self.pending: list[int] = []
        self.ready: list[int] = []

    def send(
        self,
        parent_id: int,
        _address: int,
        _size: int,
        _operation: str,
        _initiator: int,
        _ordering: int,
        _sequence: int,
    ) -> int:
        self.accepted.append(parent_id)
        self.pending.append(parent_id)
        return self.SEND_ACCEPTED

    def advance_until_event(self, cycles: int) -> int:
        self.current_cycle += max(1, cycles)
        self.global_time_fs = self.current_cycle * 1_000
        self.ready.extend(self.pending)
        self.pending.clear()
        return max(1, cycles)

    def pop_completions(self) -> list[dict[str, int]]:
        result = [
            {
                "parent_id": parent_id,
                "completion_cycle": self.current_cycle,
                "completion_time_fs": self.global_time_fs,
            }
            for parent_id in self.ready
        ]
        self.ready.clear()
        return result

    def close(self) -> dict[str, object]:
        count = len(self.accepted)
        return {
            "instances": 1,
            "outstanding": 0,
            "accepted_parent_ids": count,
            "observed_completion_ids": count,
            "global_time_fs": self.global_time_fs,
            "initiators": {
                "gpu0": {"parents": count, "completed": count, "children": count},
                "atlas0.compute": {"parents": 0, "completed": 0, "children": 0},
            },
        }


def test_metadata_requests_complete_before_runtime_task_returns() -> None:
    catalog = RuntimeTaskModelCatalog.load(CATALOG)
    node = ModelNode(
        "r.kv_allocate",
        NodeKind.STATE,
        "kv_allocate",
        Phase.CONTROL,
        0,
        0,
        attributes={"batch_size": 1, "context_length": 16},
    )
    contract = catalog.contract_for(node)
    estimate = catalog.estimate(node, _model())
    binding = RuntimeTaskAddressBinding(
        "task.allocate", tuple(), tuple(), 0x500000, 128
    )
    requests = plan_runtime_task_requests(
        node, _model(), contract, estimate, binding, {}
    )
    result = run_runtime_task_memory(
        Path.cwd(),
        {"gpu_clock_hz": 1_132_000_000},
        contract,
        estimate,
        requests,
        bridge_factory=_FakeBridge,
    )
    assert len(requests) == 3
    assert result.statistics["request_count"] == 3
    assert result.statistics["outstanding"] == 0
    assert len(result.completions) == 3
    assert result.duration_fs > result.statistics["global_time_fs"]
