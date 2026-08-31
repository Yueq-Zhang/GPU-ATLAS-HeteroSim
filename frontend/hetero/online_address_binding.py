"""Materialize strict online TraceAddr-to-Global-PA binding tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .trace_manifest import SimulationBufferBinding, TraceManifest, TraceManifestError


class OnlineAddressBindingError(ValueError):
    """Raised when a runtime binding table cannot be made unambiguous."""


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PackedRangeRebasePolicy:
    mode: str
    memory_space_id: str
    physical_base_bytes: int
    capacity_bytes: int
    alignment_bytes: int
    require_nonzero_translations: bool

    @classmethod
    def load(cls, payload: object) -> "PackedRangeRebasePolicy":
        if not isinstance(payload, dict):
            raise OnlineAddressBindingError("address_translation must be an object")
        required = {
            "mode",
            "memory_space_id",
            "physical_base_bytes",
            "capacity_bytes",
            "alignment_bytes",
            "require_nonzero_translations",
        }
        missing = required - payload.keys()
        extra = payload.keys() - required
        if missing or extra:
            raise OnlineAddressBindingError(
                "address_translation keys mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        result = cls(
            mode=str(payload["mode"]),
            memory_space_id=str(payload["memory_space_id"]),
            physical_base_bytes=int(payload["physical_base_bytes"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            alignment_bytes=int(payload["alignment_bytes"]),
            require_nonzero_translations=bool(payload["require_nonzero_translations"]),
        )
        if result.mode != "range_rebase_packed_manifest":
            raise OnlineAddressBindingError(
                "only range_rebase_packed_manifest is implemented"
            )
        if not result.memory_space_id or any(
            character.isspace() for character in result.memory_space_id
        ):
            raise OnlineAddressBindingError(
                "memory_space_id must be non-empty and contain no whitespace"
            )
        if (
            result.physical_base_bytes < 0
            or result.capacity_bytes <= 0
            or result.alignment_bytes <= 0
        ):
            raise OnlineAddressBindingError(
                "physical base must be unsigned; capacity/alignment must be positive"
            )
        if result.physical_base_bytes % result.alignment_bytes:
            raise OnlineAddressBindingError("physical_base_bytes is not aligned")
        return result


def materialize_online_address_bindings(
    manifest: TraceManifest,
    policy: PackedRangeRebasePolicy,
    output_directory: Path,
) -> dict[str, object]:
    """Pack tensor identities and emit the table consumed before GPU caches."""

    if not manifest.address_ranges:
        raise OnlineAddressBindingError(
            "range_rebase requires non-empty trace manifest address ranges"
        )
    tensor_extents: dict[str, int] = {}
    tensor_alignments: dict[str, int] = {}
    for item in manifest.address_ranges:
        tensor_extents[item.tensor_id] = max(
            tensor_extents.get(item.tensor_id, 0),
            item.tensor_offset_bytes + item.size_bytes,
        )
        tensor_alignments[item.tensor_id] = max(
            tensor_alignments.get(item.tensor_id, 1), item.alignment_bytes
        )

    cursor = policy.physical_base_bytes
    bindings: list[SimulationBufferBinding] = []
    for tensor_id in sorted(tensor_extents):
        alignment = max(policy.alignment_bytes, tensor_alignments[tensor_id])
        cursor = _align_up(cursor, alignment)
        size = tensor_extents[tensor_id]
        binding = SimulationBufferBinding(
            tensor_id=tensor_id,
            tensor_offset_bytes=0,
            size_bytes=size,
            memory_space_id=policy.memory_space_id,
            physical_offset_bytes=cursor,
        )
        bindings.append(binding)
        cursor += size
    capacity_end = policy.physical_base_bytes + policy.capacity_bytes
    if cursor > capacity_end:
        raise OnlineAddressBindingError(
            f"packed Global PA requires {cursor - policy.physical_base_bytes} bytes, "
            f"capacity is {policy.capacity_bytes}"
        )

    rows: list[tuple[int, int, int, int]] = []
    try:
        for item in sorted(manifest.address_ranges, key=lambda value: value.trace_base):
            physical = manifest.translate(item.trace_base, tuple(bindings))
            if physical.memory_space_id != policy.memory_space_id:
                raise OnlineAddressBindingError(
                    "binding escaped configured memory space"
                )
            rows.append(
                (
                    item.trace_base,
                    item.trace_end,
                    physical.offset_bytes,
                    physical.offset_bytes + item.size_bytes,
                )
            )
    except TraceManifestError as error:
        raise OnlineAddressBindingError(str(error)) from error

    # The C++ fast path treats already-rebased addresses as idempotent.  Keep
    # capture and Global-PA regions disjoint so that classification is unique.
    for trace_begin, trace_end, _, _ in rows:
        for _, _, global_begin, global_end in rows:
            if max(trace_begin, global_begin) < min(trace_end, global_end):
                raise OnlineAddressBindingError(
                    "capture and Global-PA ranges overlap; online translation "
                    "would not be idempotent"
                )

    output_directory.mkdir(parents=True, exist_ok=True)
    table_path = output_directory / "online_address_bindings.tsv"
    lines = [
        "HETEROSIM_ADDRESS_BINDINGS_V1",
        f"trace_key\t{manifest.trace_key()}",
        f"memory_space\t{policy.memory_space_id}",
        f"range_count\t{len(rows)}",
    ]
    lines.extend(
        f"range\t{trace_begin}\t{trace_end}\t{global_begin}\t{global_end}"
        for trace_begin, trace_end, global_begin, global_end in rows
    )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "hetero-online-address-binding/v1",
        "translation_point": "mem_fetch_before_gpu_cache_lookup",
        "pipeline": "TraceAddr->TensorID+offset->GlobalPA",
        "dram_tuple_mapping": "deferred_to_single_ramulator2_after_llc",
        "policy": asdict(policy),
        "trace_key": manifest.trace_key(),
        "table_path": str(table_path),
        "table_sha256": _sha256(table_path),
        "range_count": len(rows),
        "tensor_count": len(bindings),
        "allocated_bytes": cursor - policy.physical_base_bytes,
        "bindings": [
            {
                "tensor_id": item.tensor_id,
                "tensor_offset_bytes": item.tensor_offset_bytes,
                "size_bytes": item.size_bytes,
                "memory_space_id": item.memory_space_id,
                "physical_offset_bytes": item.physical_offset_bytes,
            }
            for item in bindings
        ],
    }
    metadata_path = output_directory / "online_address_binding.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def materialize_explicit_online_address_bindings(
    manifest: TraceManifest,
    policy: PackedRangeRebasePolicy,
    simulation_bindings: Sequence[SimulationBufferBinding],
    output_directory: Path,
) -> dict[str, object]:
    """Emit a caller-owned Global-PA table for one request-cycle execution.

    Unlike the packed standalone qualification path, these bindings may point
    semantic tensors at allocations owned by the global execution graph. The
    same strict coverage, capacity and ambiguity checks remain mandatory.
    """

    if not manifest.address_ranges:
        raise OnlineAddressBindingError(
            "range_rebase requires non-empty trace manifest address ranges"
        )
    bindings = tuple(simulation_bindings)
    if not bindings:
        raise OnlineAddressBindingError("explicit simulation bindings are empty")
    if any(item.memory_space_id != policy.memory_space_id for item in bindings):
        raise OnlineAddressBindingError("binding escaped configured memory space")

    capacity_end = policy.physical_base_bytes + policy.capacity_bytes
    for item in bindings:
        if (
            item.physical_offset_bytes < policy.physical_base_bytes
            or item.physical_end > capacity_end
        ):
            raise OnlineAddressBindingError(
                f"explicit Global PA for {item.tensor_id} exceeds configured capacity"
            )

    physical = sorted(bindings, key=lambda item: item.physical_offset_bytes)
    for left, right in zip(physical, physical[1:]):
        if left.physical_end > right.physical_offset_bytes:
            raise OnlineAddressBindingError(
                "explicit Global-PA bindings overlap: "
                f"{left.tensor_id} and {right.tensor_id}"
            )

    rows: list[tuple[int, int, int, int]] = []
    try:
        for item in sorted(manifest.address_ranges, key=lambda value: value.trace_base):
            translated = manifest.translate(item.trace_base, bindings)
            rows.append(
                (
                    item.trace_base,
                    item.trace_end,
                    translated.offset_bytes,
                    translated.offset_bytes + item.size_bytes,
                )
            )
    except TraceManifestError as error:
        raise OnlineAddressBindingError(str(error)) from error

    for trace_begin, trace_end, _, _ in rows:
        for _, _, global_begin, global_end in rows:
            if max(trace_begin, global_begin) < min(trace_end, global_end):
                raise OnlineAddressBindingError(
                    "capture and Global-PA ranges overlap; online translation "
                    "would not be idempotent"
                )

    output_directory.mkdir(parents=True, exist_ok=True)
    table_path = output_directory / "online_address_bindings.tsv"
    lines = [
        "HETEROSIM_ADDRESS_BINDINGS_V1",
        f"trace_key\t{manifest.trace_key()}",
        f"memory_space\t{policy.memory_space_id}",
        f"range_count\t{len(rows)}",
    ]
    lines.extend(
        f"range\t{trace_begin}\t{trace_end}\t{global_begin}\t{global_end}"
        for trace_begin, trace_end, global_begin, global_end in rows
    )
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "hetero-online-address-binding/v1",
        "translation_point": "mem_fetch_before_gpu_cache_lookup",
        "pipeline": "TraceAddr->TensorID+offset->GlobalPA",
        "dram_tuple_mapping": "deferred_to_single_ramulator2_after_llc",
        "allocation_owner": "prefill_global_timeline",
        "policy": asdict(policy),
        "trace_key": manifest.trace_key(),
        "table_path": str(table_path),
        "table_sha256": _sha256(table_path),
        "range_count": len(rows),
        "tensor_count": len(bindings),
        "allocated_bytes": sum(item.size_bytes for item in bindings),
        "bindings": [
            {
                "tensor_id": item.tensor_id,
                "tensor_offset_bytes": item.tensor_offset_bytes,
                "size_bytes": item.size_bytes,
                "memory_space_id": item.memory_space_id,
                "physical_offset_bytes": item.physical_offset_bytes,
            }
            for item in bindings
        ],
    }
    metadata_path = output_directory / "online_address_binding.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
