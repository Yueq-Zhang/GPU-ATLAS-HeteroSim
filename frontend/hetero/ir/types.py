"""Versioned, device-independent and executable IR types.

The classes in this module deliberately carry no timing implementation.  They
are the Python control-plane representation described by sections 8 and 13 of
the frozen design specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class StorageClass(str, Enum):
    PARAMETER = "parameter"
    ACTIVATION = "activation"
    TEMPORARY = "temporary"
    KV_CACHE = "kv_cache"
    METADATA = "metadata"


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    FINALIZE = "finalize"
    CONTROL = "control"


class NodeKind(str, Enum):
    COMPUTE = "compute"
    STATE = "state"
    CONTROL = "control"
    COLLECTIVE = "collective"


class TaskKind(str, Enum):
    DEVICE = "device"
    TRANSFER = "transfer"
    MIGRATION = "migration"
    SYNCHRONIZATION = "synchronization"


@dataclass(frozen=True, slots=True)
class PhysicalAddress:
    memory_space_id: str
    offset_bytes: int
    allocation_epoch: int

    def __post_init__(self) -> None:
        if not self.memory_space_id:
            raise ValueError("memory_space_id must not be empty")
        if self.offset_bytes < 0 or self.allocation_epoch < 0:
            raise ValueError("address fields must be unsigned")


@dataclass(frozen=True, slots=True)
class ValueRef:
    value_id: str
    version: int
    offset_bytes: int
    size_bytes: int
    access_mode: AccessMode

    def __post_init__(self) -> None:
        if not self.value_id or self.version < 0 or self.offset_bytes < 0:
            raise ValueError("invalid ValueRef identity")
        if self.size_bytes <= 0:
            raise ValueError("ValueRef size_bytes must be positive")


@dataclass(frozen=True, slots=True)
class Value:
    value_id: str
    shape_expr: Sequence[int | str]
    dtype: str
    layout: str
    storage_class: StorageClass
    mutable: bool = False
    alias_or_view: str | None = None
    lifetime: str = "request"

    def __post_init__(self) -> None:
        if not self.value_id or not self.dtype or not self.layout:
            raise ValueError("Value identity, dtype and layout are required")


@dataclass(frozen=True, slots=True)
class ModelNode:
    node_id: str
    kind: NodeKind
    op: str
    phase: Phase
    layer_id: int | None
    step_id: int
    dependencies: Sequence[str] = field(default_factory=tuple)
    read_values: Sequence[str] = field(default_factory=tuple)
    write_values: Sequence[str] = field(default_factory=tuple)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.op or self.step_id < 0:
            raise ValueError("invalid ModelNode identity")
        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("layer_id must be unsigned")


@dataclass(frozen=True, slots=True)
class ModelGraph:
    schema_version: str
    values: Sequence[Value]
    nodes: Sequence[ModelNode]

    def validate(self) -> None:
        value_ids = [value.value_id for value in self.values]
        node_ids = [node.node_id for node in self.nodes]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError("duplicate value_id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node_id")
        known_nodes: set[str] = set()
        known_values = set(value_ids)
        for node in self.nodes:
            missing_dependencies = set(node.dependencies) - known_nodes
            if missing_dependencies:
                raise ValueError(
                    f"node {node.node_id} has unresolved dependencies: "
                    f"{sorted(missing_dependencies)}"
                )
            missing_values = (
                set(node.read_values) | set(node.write_values)
            ) - known_values
            if missing_values:
                raise ValueError(
                    f"node {node.node_id} references unknown values: "
                    f"{sorted(missing_values)}"
                )
            known_nodes.add(node.node_id)


@dataclass(frozen=True, slots=True)
class ExecutionTask:
    task_id: str
    template_node_id: str
    task_kind: TaskKind
    phase: Phase
    layer_id: int | None
    step_id: int
    device_id: str
    backend_id: str
    dependencies: Sequence[str] = field(default_factory=tuple)
    input_values: Sequence[ValueRef] = field(default_factory=tuple)
    output_values: Sequence[ValueRef] = field(default_factory=tuple)
    read_memory_spaces: Sequence[str] = field(default_factory=tuple)
    write_memory_spaces: Sequence[str] = field(default_factory=tuple)
    resource_requirements: Mapping[str, Any] = field(default_factory=dict)
    compiled_artifact_ref: str | None = None
    fidelity: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.template_node_id:
            raise ValueError("ExecutionTask identity is required")


@dataclass(frozen=True, slots=True)
class ArtifactRequest:
    request_id: str
    epoch_id: int
    backend_id: str
    artifact_kind: str
    compile_plan_key: str
    task_signature: str
    shape_signature: str
    placement_result_hash: str
    simulation_buffer_bindings: Mapping[str, PhysicalAddress]
    expected_artifact_key: str


@dataclass(frozen=True, slots=True)
class ExecutionInstance:
    epoch_id: int
    committed_state_hash: str
    task_ids: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
