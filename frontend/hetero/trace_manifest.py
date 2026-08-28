"""Strict trace metadata and TraceAddr-to-Global-PA translation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class TraceManifestError(ValueError):
    """Raised when trace metadata is incomplete or internally inconsistent."""


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TraceManifestError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as error:
            raise TraceManifestError(f"{field} is not an integer: {value}") from error
    raise TraceManifestError(f"{field} must be an integer or base-prefixed string")


@dataclass(frozen=True)
class TraceAddressRange:
    """Capture-only binding from a raw TraceAddr to stable tensor identity."""

    capture_allocation_id: str
    trace_base: int
    size_bytes: int
    tensor_id: str
    tensor_offset_bytes: int
    capture_epoch: int
    backing_allocation_id: str
    view_offset_bytes: int
    alignment_bytes: int
    shape: tuple[int, ...]
    layout: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TraceAddressRange":
        required = {
            "capture_allocation_id",
            "trace_base",
            "size_bytes",
            "tensor_id",
            "tensor_offset_bytes",
            "capture_epoch",
            "backing_allocation_id",
            "view_offset_bytes",
            "alignment_bytes",
            "shape",
            "layout",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing or extra:
            raise TraceManifestError(
                f"address range keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        result = cls(
            capture_allocation_id=str(payload["capture_allocation_id"]),
            trace_base=_integer(payload["trace_base"], "trace_base"),
            size_bytes=_integer(payload["size_bytes"], "size_bytes"),
            tensor_id=str(payload["tensor_id"]),
            tensor_offset_bytes=_integer(
                payload["tensor_offset_bytes"], "tensor_offset_bytes"
            ),
            capture_epoch=_integer(payload["capture_epoch"], "capture_epoch"),
            backing_allocation_id=str(payload["backing_allocation_id"]),
            view_offset_bytes=_integer(payload["view_offset_bytes"], "view_offset_bytes"),
            alignment_bytes=_integer(payload["alignment_bytes"], "alignment_bytes"),
            shape=tuple(_integer(item, "shape item") for item in payload["shape"])
            if isinstance(payload["shape"], list)
            else (),
            layout=str(payload["layout"]),
        )
        if not all(
            (
                result.capture_allocation_id,
                result.tensor_id,
                result.backing_allocation_id,
                result.layout,
            )
        ):
            raise TraceManifestError("capture/tensor/backing IDs and layout must be non-empty")
        if result.trace_base < 0 or result.tensor_offset_bytes < 0:
            raise TraceManifestError("trace_base and tensor_offset_bytes must be non-negative")
        if result.size_bytes <= 0:
            raise TraceManifestError("size_bytes must be positive")
        if result.capture_epoch < 0 or result.view_offset_bytes < 0:
            raise TraceManifestError("capture_epoch and view_offset_bytes must be non-negative")
        if result.alignment_bytes <= 0 or result.trace_base % result.alignment_bytes:
            raise TraceManifestError("trace_base must satisfy positive alignment_bytes")
        if not result.shape or any(dimension <= 0 for dimension in result.shape):
            raise TraceManifestError("shape must contain positive dimensions")
        return result

    @property
    def trace_end(self) -> int:
        return self.trace_base + self.size_bytes

    @property
    def to_dict(self) -> dict[str, object]:
        return {
            "capture_allocation_id": self.capture_allocation_id,
            "trace_base": hex(self.trace_base),
            "size_bytes": self.size_bytes,
            "tensor_id": self.tensor_id,
            "tensor_offset_bytes": self.tensor_offset_bytes,
            "capture_epoch": self.capture_epoch,
            "backing_allocation_id": self.backing_allocation_id,
            "view_offset_bytes": self.view_offset_bytes,
            "alignment_bytes": self.alignment_bytes,
            "shape": list(self.shape),
            "layout": self.layout,
        }


@dataclass(frozen=True)
class AddressTranslation:
    tensor_id: str
    tensor_offset: int


@dataclass(frozen=True)
class PhysicalAddress:
    memory_space_id: str
    offset_bytes: int


@dataclass(frozen=True)
class SimulationBufferBinding:
    """Candidate-specific mapping from tensor identity to a PhysicalAddress range."""

    tensor_id: str
    tensor_offset_bytes: int
    size_bytes: int
    memory_space_id: str
    physical_offset_bytes: int

    def __post_init__(self) -> None:
        if not self.tensor_id or not self.memory_space_id:
            raise TraceManifestError("simulation binding IDs must be non-empty")
        if min(
            self.tensor_offset_bytes,
            self.physical_offset_bytes,
        ) < 0 or self.size_bytes <= 0:
            raise TraceManifestError(
                "simulation binding offsets must be non-negative and size positive"
            )

    @property
    def tensor_end(self) -> int:
        return self.tensor_offset_bytes + self.size_bytes

    @property
    def physical_end(self) -> int:
        return self.physical_offset_bytes + self.size_bytes


@dataclass(frozen=True)
class TraceManifest:
    schema_version: str
    trace_id: str
    trace_semantics: str
    replay_safe: bool
    qualification_record: str | None
    kernels_list: Path | None
    capture: Mapping[str, object]
    compilation: Mapping[str, object]
    address_ranges: tuple[TraceAddressRange, ...]
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], source_path: Path | None = None
    ) -> "TraceManifest":
        required = {
            "schema_version",
            "trace_id",
            "trace_semantics",
            "replay_safe",
            "qualification_record",
            "kernels_list",
            "capture",
            "compilation",
            "address_ranges",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing or extra:
            raise TraceManifestError(
                f"manifest keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if payload["schema_version"] != "hetero-trace-manifest/v1":
            raise TraceManifestError("schema_version must be hetero-trace-manifest/v1")
        semantics = str(payload["trace_semantics"])
        if semantics not in {"none", "functional", "timing_feedback"}:
            raise TraceManifestError(
                "trace_semantics must be none, functional or timing_feedback"
            )
        replay_safe = payload["replay_safe"]
        if not isinstance(replay_safe, bool):
            raise TraceManifestError("replay_safe must be a boolean")
        qualification = payload["qualification_record"]
        if qualification is not None and not isinstance(qualification, str):
            raise TraceManifestError("qualification_record must be a path or null")
        if replay_safe and not qualification:
            raise TraceManifestError(
                "replay_safe=true requires a qualification_record"
            )
        capture = payload["capture"]
        compilation = payload["compilation"]
        ranges = payload["address_ranges"]
        if not isinstance(capture, Mapping) or not isinstance(compilation, Mapping):
            raise TraceManifestError("capture and compilation must be objects")
        if not isinstance(ranges, list):
            raise TraceManifestError("address_ranges must be an array")
        parsed_ranges = tuple(
            TraceAddressRange.from_dict(item)
            for item in ranges
            if isinstance(item, Mapping)
        )
        if len(parsed_ranges) != len(ranges):
            raise TraceManifestError("every address_ranges item must be an object")
        base = source_path.parent if source_path else Path.cwd()
        kernels_value = payload["kernels_list"]
        kernels_list: Path | None
        if kernels_value is None:
            kernels_list = None
        elif isinstance(kernels_value, str) and kernels_value:
            kernels_list = Path(kernels_value)
            if not kernels_list.is_absolute():
                kernels_list = (base / kernels_list).resolve()
        else:
            raise TraceManifestError("kernels_list must be a non-empty path or null")
        result = cls(
            schema_version=str(payload["schema_version"]),
            trace_id=str(payload["trace_id"]),
            trace_semantics=semantics,
            replay_safe=replay_safe,
            qualification_record=qualification,
            kernels_list=kernels_list,
            capture=dict(capture),
            compilation=dict(compilation),
            address_ranges=parsed_ranges,
            source_path=source_path,
        )
        if not result.trace_id:
            raise TraceManifestError("trace_id must be non-empty")
        if semantics == "none":
            if kernels_list is not None or parsed_ranges or replay_safe:
                raise TraceManifestError(
                    "trace_semantics=none requires null kernels_list, no ranges and replay_safe=false"
                )
        elif kernels_list is None:
            raise TraceManifestError("captured trace semantics require kernels_list")
        result._validate_ranges()
        if result.replay_safe:
            result._validate_qualification()
        return result

    @classmethod
    def load(cls, path: Path) -> "TraceManifest":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TraceManifestError(f"cannot read trace manifest {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise TraceManifestError("trace manifest root must be an object")
        return cls.from_dict(payload, path.resolve())

    def _validate_ranges(self) -> None:
        by_trace = sorted(self.address_ranges, key=lambda item: item.trace_base)
        for left, right in zip(by_trace, by_trace[1:]):
            if left.trace_end > right.trace_base:
                raise TraceManifestError(
                    f"trace ranges overlap: {left.tensor_id} and {right.tensor_id}"
                )
    def _validate_qualification(self) -> None:
        assert self.qualification_record is not None
        record_path = Path(self.qualification_record)
        if not record_path.is_absolute():
            base = self.source_path.parent if self.source_path else Path.cwd()
            record_path = (base / record_path).resolve()
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TraceManifestError(
                f"cannot validate qualification_record {record_path}: {error}"
            ) from error
        if record.get("schema_version") != "hetero-accel-sim-qualification/v1":
            raise TraceManifestError("qualification_record has the wrong schema_version")
        if record.get("status") != "passed":
            raise TraceManifestError("qualification_record status is not passed")
        if record.get("replay_safety_qualified") is not True:
            raise TraceManifestError(
                "qualification_record does not qualify cross-configuration replay safety"
            )
        if record.get("trace_key") != self.trace_key():
            raise TraceManifestError("qualification_record trace_key does not match")

    def normalize(self, trace_address: int) -> AddressTranslation:
        for item in self.address_ranges:
            if item.trace_base <= trace_address < item.trace_end:
                offset = trace_address - item.trace_base
                return AddressTranslation(
                    tensor_id=item.tensor_id,
                    tensor_offset=item.tensor_offset_bytes + offset,
                )
        raise TraceManifestError(f"unmapped trace address: {hex(trace_address)}")

    def translate(
        self,
        trace_address: int,
        simulation_bindings: tuple[SimulationBufferBinding, ...],
    ) -> PhysicalAddress:
        by_space: dict[str, list[SimulationBufferBinding]] = {}
        for item in simulation_bindings:
            by_space.setdefault(item.memory_space_id, []).append(item)
        for space, items in by_space.items():
            ordered = sorted(items, key=lambda item: item.physical_offset_bytes)
            for left, right in zip(ordered, ordered[1:]):
                if left.physical_end > right.physical_offset_bytes:
                    raise TraceManifestError(
                        f"SimulationBufferBinding overlap in {space}: "
                        f"{left.tensor_id} and {right.tensor_id}"
                    )
        normalized = self.normalize(trace_address)
        candidates = [
            item
            for item in simulation_bindings
            if item.tensor_id == normalized.tensor_id
            and item.tensor_offset_bytes <= normalized.tensor_offset < item.tensor_end
        ]
        if len(candidates) != 1:
            raise TraceManifestError(
                f"expected one SimulationBufferBinding for {normalized.tensor_id}+"
                f"{normalized.tensor_offset}, found {len(candidates)}"
            )
        binding = candidates[0]
        return PhysicalAddress(
            memory_space_id=binding.memory_space_id,
            offset_bytes=binding.physical_offset_bytes
            + normalized.tensor_offset
            - binding.tensor_offset_bytes,
        )

    def trace_key(self) -> str:
        """Hash only capture/program identity; exclude simulated DRAM parameters."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "trace_semantics": self.trace_semantics,
            "capture": self.capture,
            "compilation": self.compilation,
            "address_ranges": [item.to_dict for item in self.address_ranges],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
