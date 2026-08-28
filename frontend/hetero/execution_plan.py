"""Strict single-placement lowering with versioned value residency.

This module is intentionally timing-independent.  It turns one logical model
graph and one placement decision per node into device tasks, value-granular
cross-device routes, and an auditable residency plan.  Concrete Backends and
the global event runtime consume this plan later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Mapping, Sequence

from .ir import ModelGraph, ModelNode, Value
from .model_graph import ModelSpec
from .placement import PlacementDecision
from .topology import device_memory, lower_cross_device_dependency


class SinglePlacementError(ValueError):
    """Raised when a logical graph cannot be lowered without ambiguity."""


@dataclass(frozen=True, slots=True)
class PlannedRoute:
    task_id: str
    consumer_task_id: str
    dependencies: tuple[str, ...]
    producer_device: str
    consumer_device: str
    value_id: str
    value_version: int
    payload_bytes: int
    lowering: object


@dataclass(frozen=True, slots=True)
class PlannedNode:
    node: ModelNode
    decision: PlacementDecision
    task_id: str
    dependencies: tuple[str, ...]
    input_values: tuple[dict[str, object], ...]
    output_values: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class SinglePlacementPlan:
    nodes: tuple[PlannedNode, ...]
    routes: tuple[PlannedRoute, ...]
    residency_events: tuple[dict[str, object], ...]
    final_records: tuple[dict[str, object], ...]
    conservation: Mapping[str, object]


@dataclass(slots=True)
class _ValueState:
    version: int
    memory_space_id: str
    owner_device: str
    producer_task_id: str | None
    state: str


def _value_size_bytes(value: Value, node: ModelNode, model: ModelSpec) -> int:
    dimensions: list[int] = []
    for dimension in value.shape_expr:
        if isinstance(dimension, int):
            dimensions.append(dimension)
        elif dimension == "kv_tokens":
            dimensions.append(max(1, int(node.attributes.get("attention_kv_len", 1))))
        else:
            raise SinglePlacementError(
                f"cannot materialize shape dimension {dimension!r} for {value.value_id}"
            )
    return max(1, prod(dimensions) * model.bytes_per_element)


def _validate_decisions(
    graph: ModelGraph, decisions: Sequence[PlacementDecision], profile: str
) -> dict[str, PlacementDecision]:
    expected = {node.node_id for node in graph.nodes}
    actual: dict[str, PlacementDecision] = {}
    duplicates: list[str] = []
    for decision in decisions:
        if decision.node_id in actual:
            duplicates.append(decision.node_id)
        actual[decision.node_id] = decision
        try:
            device_memory(profile, decision.target_device)
        except ValueError as error:
            raise SinglePlacementError(str(error)) from error
    missing = expected - actual.keys()
    extra = actual.keys() - expected
    if duplicates or missing or extra:
        raise SinglePlacementError(
            "placement must contain exactly one decision per logical node: "
            f"duplicates={sorted(set(duplicates))}, missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return actual


def build_single_placement_plan(
    graph: ModelGraph,
    decisions: Sequence[PlacementDecision],
    profile: str,
    access_policy: str,
    model: ModelSpec,
) -> SinglePlacementPlan:
    """Lower a graph while preserving single execution and value versions."""

    graph.validate()
    by_node = _validate_decisions(graph, decisions, profile)
    values = {value.value_id: value for value in graph.values}
    states: dict[str, _ValueState] = {}
    planned_nodes: list[PlannedNode] = []
    planned_routes: list[PlannedRoute] = []
    residency_events: list[dict[str, object]] = []
    sequence = 0

    def event(payload: dict[str, object]) -> None:
        nonlocal sequence
        residency_events.append({"sequence": sequence, **payload})
        sequence += 1

    for node in graph.nodes:
        decision = by_node[node.node_id]
        task_id = f"task.{node.node_id}"
        dependencies = [f"task.{dependency}" for dependency in node.dependencies]
        input_values: list[dict[str, object]] = []

        for input_index, value_id in enumerate(node.read_values):
            value = values[value_id]
            size_bytes = _value_size_bytes(value, node, model)
            state = states.get(value_id)
            if state is None:
                state = _ValueState(
                    version=0,
                    memory_space_id=device_memory(profile, decision.target_device),
                    owner_device=decision.target_device,
                    producer_task_id=None,
                    state="exclusive_clean",
                )
                states[value_id] = state
                event(
                    {
                        "event": "register_external_input",
                        "trigger": "task_start",
                        "trigger_task_id": task_id,
                        "value_id": value_id,
                        "version": 0,
                        "device_id": decision.target_device,
                        "memory_space_id": state.memory_space_id,
                        "size_bytes": size_bytes,
                        "initialization_policy": "first_consumer_binding",
                    }
                )

            if state.owner_device != decision.target_device:
                lowering = lower_cross_device_dependency(
                    profile,
                    state.owner_device,
                    decision.target_device,
                    access_policy,
                )
                route_task_id = f"route.{node.node_id}.input{input_index}"
                route_dependencies = (
                    (state.producer_task_id,) if state.producer_task_id else tuple()
                )
                planned_routes.append(
                    PlannedRoute(
                        task_id=route_task_id,
                        consumer_task_id=task_id,
                        dependencies=route_dependencies,
                        producer_device=state.owner_device,
                        consumer_device=decision.target_device,
                        value_id=value_id,
                        value_version=state.version,
                        payload_bytes=size_bytes,
                        lowering=lowering,
                    )
                )
                dependencies.append(route_task_id)
                event(
                    {
                        "event": "cross_device_route",
                        "trigger": "task_completion",
                        "trigger_task_id": route_task_id,
                        "value_id": value_id,
                        "version": state.version,
                        "source_device": state.owner_device,
                        "destination_device": decision.target_device,
                        "source_space": state.memory_space_id,
                        "destination_space": lowering.destination_space,
                        "route_kind": lowering.kind.value,
                        "route_id": lowering.route_id,
                        "actions": list(lowering.actions),
                        "size_bytes": size_bytes,
                    }
                )
                state = _ValueState(
                    version=state.version,
                    memory_space_id=lowering.destination_space,
                    owner_device=decision.target_device,
                    producer_task_id=route_task_id,
                    state=(
                        "shared_clean"
                        if lowering.kind.value in {"synchronization", "remote_access"}
                        else "exclusive_clean"
                    ),
                )
                states[value_id] = state

            input_values.append(
                {
                    "value_id": value_id,
                    "version": state.version,
                    "offset_bytes": 0,
                    "size_bytes": size_bytes,
                    "access_mode": "read",
                    "memory_space_id": state.memory_space_id,
                    "storage_class": value.storage_class.value,
                    "dtype": value.dtype,
                }
            )
            event(
                {
                    "event": "read",
                    "trigger": "task_start",
                    "trigger_task_id": task_id,
                    "value_id": value_id,
                    "version": state.version,
                    "device_id": decision.target_device,
                    "memory_space_id": state.memory_space_id,
                    "size_bytes": size_bytes,
                }
            )

        dependencies = list(dict.fromkeys(dependencies))
        output_values: list[dict[str, object]] = []
        for value_id in node.write_values:
            value = values[value_id]
            size_bytes = _value_size_bytes(value, node, model)
            previous = states.get(value_id)
            version = 1 if previous is None else previous.version + 1
            memory_space = device_memory(profile, decision.target_device)
            states[value_id] = _ValueState(
                version=version,
                memory_space_id=memory_space,
                owner_device=decision.target_device,
                producer_task_id=task_id,
                state="exclusive_dirty",
            )
            output_values.append(
                {
                    "value_id": value_id,
                    "version": version,
                    "offset_bytes": 0,
                    "size_bytes": size_bytes,
                    "access_mode": "write",
                    "memory_space_id": memory_space,
                    "storage_class": value.storage_class.value,
                    "dtype": value.dtype,
                }
            )
            event(
                {
                    "event": "write",
                    "trigger": "task_completion",
                    "trigger_task_id": task_id,
                    "value_id": value_id,
                    "version": version,
                    "device_id": decision.target_device,
                    "memory_space_id": memory_space,
                    "size_bytes": size_bytes,
                }
            )

        planned_nodes.append(
            PlannedNode(
                node=node,
                decision=decision,
                task_id=task_id,
                dependencies=tuple(dependencies),
                input_values=tuple(input_values),
                output_values=tuple(output_values),
            )
        )

    node_ids = [planned.node.node_id for planned in planned_nodes]
    conservation = {
        "logical_node_count": len(graph.nodes),
        "placement_decision_count": len(decisions),
        "planned_device_task_count": len(planned_nodes),
        "unique_logical_node_count": len(set(node_ids)),
        "each_logical_node_exactly_once": (
            len(graph.nodes) == len(decisions) == len(planned_nodes) == len(set(node_ids))
        ),
    }
    if not conservation["each_logical_node_exactly_once"]:
        raise SinglePlacementError("single-placement conservation failed")

    final_records = tuple(
        {
            "value_id": value_id,
            "version": state.version,
            "memory_space_id": state.memory_space_id,
            "owner_device": state.owner_device,
            "state": state.state,
            "producer_task_id": state.producer_task_id,
        }
        for value_id, state in sorted(states.items())
    )
    return SinglePlacementPlan(
        nodes=tuple(planned_nodes),
        routes=tuple(planned_routes),
        residency_events=tuple(residency_events),
        final_records=final_records,
        conservation=conservation,
    )


def route_to_dict(route: PlannedRoute) -> dict[str, object]:
    """Serialize the timing-independent portion of a planned route."""

    return {
        "task_id": route.task_id,
        "consumer_task_id": route.consumer_task_id,
        "dependencies": list(route.dependencies),
        "producer_device": route.producer_device,
        "consumer_device": route.consumer_device,
        "value_id": route.value_id,
        "value_version": route.value_version,
        "payload_bytes": route.payload_bytes,
        **asdict(route.lowering),
    }
