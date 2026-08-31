from frontend.hetero.global_memory_map import GlobalAllocation
from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec
from frontend.hetero.online_operator_runtime import OnlineDispatchSpec
from frontend.hetero.prefill_cycle_artifact import CycleTaskPlan
from frontend.hetero.prefill_cycle_runtime import run_prefill_cycle_dag


class _Dispatcher:
    def __init__(self) -> None:
        self.plans: dict[str, CycleTaskPlan] = {}

    def dispatch(self, spec: OnlineDispatchSpec) -> CycleTaskPlan:
        plan = CycleTaskPlan(
            task_id=spec.task_id,
            backend_id=f"{spec.backend_key}.golden",
            device_id=spec.device_id,
            device_clock_hz=1_000_000_000,
            native_compute_cycles=5,
            global_compute_cycles=5,
            formula={"kind": "golden"},
            fidelity={"performance_eligible": False},
            artifact={"operator": spec.node.op},
        )
        self.plans[spec.task_id] = plan
        return plan

    def coverage(self, expected: set[str]) -> dict[str, object]:
        assert set(self.plans) == expected
        return {
            "all_tasks_covered": True,
            "covered_tasks": len(expected),
            "analytical_fallback_tasks": 0,
        }

    def memory_traffic_mode(self, task_id: str) -> str:
        assert task_id in self.plans
        return "sampled"


class _Bridge:
    def __init__(self) -> None:
        self.cycle = 0
        self.pending: list[dict[str, int]] = []
        self.accepted = 0
        self.completed = 0
        self.by_initiator = {0: 0, 1: 0}
        self._closed = False

    @property
    def current_cycle(self) -> int:
        return self.cycle

    @property
    def global_time_fs(self) -> int:
        return self.cycle * 1_000_000

    def send(
        self,
        parent_id: int,
        address: int,
        size: int,
        operation: str,
        initiator: int,
        ordering_domain: int,
        user_tag: int,
    ) -> int:
        del address, size, operation, ordering_domain, user_tag
        self.accepted += 1
        self.by_initiator[initiator] += 1
        self.pending.append(
            {"parent_id": parent_id, "initiator": initiator, "done": self.cycle + 2}
        )
        return 1

    def advance_until_event(self, maximum: int) -> int:
        target = self.cycle + maximum
        if self.pending:
            target = min(target, min(item["done"] for item in self.pending))
        advanced = target - self.cycle
        self.cycle = target
        return advanced

    def pop_completions(self) -> list[dict[str, int]]:
        ready = [item for item in self.pending if item["done"] <= self.cycle]
        self.pending = [item for item in self.pending if item["done"] > self.cycle]
        self.completed += len(ready)
        return [
            {
                "parent_id": item["parent_id"],
                "initiator": item["initiator"],
                "completion_cycle": self.cycle,
                "completion_time_fs": self.global_time_fs,
            }
            for item in ready
        ]

    def close(self) -> dict[str, object]:
        assert not self.pending
        self._closed = True
        return {
            "instances": 1,
            "accepted_parent_ids": self.accepted,
            "observed_completion_ids": self.completed,
            "outstanding": 0,
            "initiators": {
                "gpu0": {"parents": self.by_initiator[0]},
                "atlas0.compute": {"parents": self.by_initiator[1]},
            },
        }


def _value(value_id: str, version: int) -> dict[str, object]:
    return {
        "value_id": value_id,
        "version": version,
        "memory_space_id": "shared0.dram3d",
        "size_bytes": 64,
        "storage_class": "activation",
        "dtype": "fp16",
    }


def _spec(task_id: str, device: str) -> OnlineDispatchSpec:
    model = ModelSpec("tiny", 128, 256, 1, 4, 2, 32, 256)
    node = ModelNode(
        task_id.removeprefix("task."),
        NodeKind.COMPUTE,
        "residual_add",
        Phase.PREFILL,
        0,
        0,
    )
    return OnlineDispatchSpec(
        task_id, "gpu" if device == "gpu0" else "atlas", node, model, device
    )


def test_live_runtime_orders_reads_compute_writes_route_and_consumer() -> None:
    graph = {
        "tasks": [
            {
                "task_id": "task.gpu",
                "task_kind": "device",
                "device_id": "gpu0",
                "resource_id": "gpu0",
                "dependencies": [],
                "release_time_fs": 0,
                "input_values": [_value("a", 0)],
                "output_values": [_value("x", 1)],
            },
            {
                "task_id": "task.atlas",
                "task_kind": "device",
                "device_id": "atlas0.compute",
                "resource_id": "atlas0.compute",
                "dependencies": ["task.gpu", "route.x"],
                "release_time_fs": 0,
                "input_values": [_value("x", 1)],
                "output_values": [_value("y", 1)],
            },
        ],
        "routes": [
            {
                "task_id": "route.x",
                "task_kind": "route",
                "resource_id": "shared3d.sync",
                "dependencies": ["task.gpu"],
                "release_time_fs": 0,
                "duration_fs": 3_000_000,
                "value_id": "x",
                "value_version": 1,
                "producer_device": "gpu0",
                "consumer_device": "atlas0.compute",
                "payload_bytes": 64,
                "actions": ["release_fence", "acquire_fence"],
            }
        ],
        "residency_plan": {
            "events": [
                {
                    "event": "register_external_input",
                    "trigger_task_id": "task.gpu",
                    "value_id": "a",
                    "version": 0,
                    "device_id": "gpu0",
                }
            ]
        },
    }
    allocations = {
        value_id: GlobalAllocation(
            value_id, "shared0.dram3d", index * 64, 64, 64, "activation", "fp16"
        )
        for index, value_id in enumerate(("a", "x", "y"))
    }
    bridge = _Bridge()
    result = run_prefill_cycle_dag(
        graph,
        {
            "task.gpu": _spec("task.gpu", "gpu0"),
            "task.atlas": _spec("task.atlas", "atlas0.compute"),
        },
        _Dispatcher(),
        bridge,  # type: ignore[arg-type]
        allocations,
        global_clock_hz=1_000_000_000,
        transaction_bytes=64,
        max_samples_per_value=1,
    )
    timing = {item["task_id"]: item for item in result["tasks"]}
    assert timing["task.gpu"]["completion_cycle"] == 9
    assert timing["route.x"]["start_cycle"] == 9
    assert timing["route.x"]["completion_cycle"] == 14
    assert timing["task.atlas"]["start_cycle"] == 14
    assert timing["task.atlas"]["completion_cycle"] == 23
    trace = result["memory_trace"]
    assert [item["issue_cycle"] for item in trace] == [0, 7, 9, 14, 21]
    assert [item["operation"] for item in trace] == [
        "read",
        "write",
        "read",
        "read",
        "write",
    ]
    assert result["memory_statistics"]["represented_total_bytes"] == 256
    assert result["backend_dispatch_count"] == 2
    assert result["artifact_coverage"]["analytical_fallback_tasks"] == 0
    assert result["memory_statistics"]["outstanding"] == 0
    assert result["final_versions"] == {"a": 0, "x": 1, "y": 1}
