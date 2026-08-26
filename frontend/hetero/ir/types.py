"""Minimal versioned IR types shared by control-plane tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


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

