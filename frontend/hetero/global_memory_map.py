"""Deterministic Global-PA allocation and bounded request sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence


class GlobalMemoryMapError(ValueError):
    """Raised when value ranges cannot be allocated or sampled safely."""


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, slots=True)
class GlobalAllocation:
    value_id: str
    memory_space_id: str
    base_address: int
    size_bytes: int
    alignment_bytes: int
    storage_class: str
    dtype: str

    @property
    def end_address_exclusive(self) -> int:
        return self.base_address + self.size_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "value_id": self.value_id,
            "memory_space_id": self.memory_space_id,
            "base_address": self.base_address,
            "end_address_exclusive": self.end_address_exclusive,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
            "storage_class": self.storage_class,
            "dtype": self.dtype,
        }


def build_global_memory_map(
    execution_graph: Mapping[str, object],
    memory_space_id: str,
    capacity_bytes: int,
    alignment_bytes: int,
) -> tuple[dict[str, GlobalAllocation], dict[str, object]]:
    if not memory_space_id or capacity_bytes <= 0 or alignment_bytes <= 0:
        raise GlobalMemoryMapError("Global-PA allocator configuration is invalid")
    raw_tasks = execution_graph.get("tasks")
    if not isinstance(raw_tasks, list):
        raise GlobalMemoryMapError("execution graph tasks must be an array")
    requirements: dict[str, dict[str, object]] = {}
    for raw_task in raw_tasks:
        if not isinstance(raw_task, Mapping):
            raise GlobalMemoryMapError("execution task must be an object")
        for field in ("input_values", "output_values"):
            values = raw_task.get(field)
            if not isinstance(values, list):
                raise GlobalMemoryMapError(f"{field} must be an array")
            for raw in values:
                if not isinstance(raw, Mapping):
                    raise GlobalMemoryMapError("value reference must be an object")
                value_id = str(raw["value_id"])
                space = str(raw["memory_space_id"])
                if space != memory_space_id:
                    raise GlobalMemoryMapError(
                        f"P10b-B live memory accepts only {memory_space_id}; "
                        f"{value_id} is in {space}"
                    )
                size = int(raw["size_bytes"])
                if size <= 0:
                    raise GlobalMemoryMapError(f"{value_id} has invalid size")
                candidate = {
                    "size_bytes": size,
                    "storage_class": str(raw.get("storage_class", "unknown")),
                    "dtype": str(raw.get("dtype", "unknown")),
                }
                previous = requirements.get(value_id)
                if previous is None or size > int(previous["size_bytes"]):
                    requirements[value_id] = candidate

    priority = {
        "parameter": 0,
        "kv_cache": 1,
        "activation": 2,
        "temporary": 3,
        "metadata": 4,
        "unknown": 5,
    }
    ordered = sorted(
        requirements.items(),
        key=lambda item: (priority.get(str(item[1]["storage_class"]), 6), item[0]),
    )
    cursor = 0
    allocations: dict[str, GlobalAllocation] = {}
    for value_id, requirement in ordered:
        cursor = _align_up(cursor, alignment_bytes)
        size = int(requirement["size_bytes"])
        if size > capacity_bytes - cursor:
            raise GlobalMemoryMapError(
                f"Global-PA capacity exceeded by {value_id}: "
                f"required_end={cursor + size}, capacity={capacity_bytes}"
            )
        allocations[value_id] = GlobalAllocation(
            value_id=value_id,
            memory_space_id=memory_space_id,
            base_address=cursor,
            size_bytes=size,
            alignment_bytes=alignment_bytes,
            storage_class=str(requirement["storage_class"]),
            dtype=str(requirement["dtype"]),
        )
        cursor += size

    ranges = [allocation.to_dict() for allocation in allocations.values()]
    ranges.sort(key=lambda item: int(item["base_address"]))
    for left, right in zip(ranges, ranges[1:]):
        if int(left["end_address_exclusive"]) > int(right["base_address"]):
            raise GlobalMemoryMapError("Global-PA allocations overlap")
    payload = {
        "schema_version": "hetero-global-memory-map/v1",
        "address_semantics": "allocated_global_pa_identity_untranslated",
        "memory_space_id": memory_space_id,
        "capacity_bytes": capacity_bytes,
        "alignment_bytes": alignment_bytes,
        "allocated_bytes": cursor,
        "allocation_count": len(ranges),
        "non_overlapping": True,
        "ranges": ranges,
    }
    return allocations, payload


def sampled_requests_for_values(
    task_id: str,
    device_id: str,
    values: Sequence[Mapping[str, object]],
    allocations: Mapping[str, GlobalAllocation],
    operation: str,
    transaction_bytes: int,
    max_samples_per_value: int,
    *,
    full_traffic: bool = False,
) -> list[dict[str, object]]:
    return list(
        iter_requests_for_values(
            task_id,
            device_id,
            values,
            allocations,
            operation,
            transaction_bytes,
            max_samples_per_value,
            full_traffic=full_traffic,
        )
    )


def iter_requests_for_values(
    task_id: str,
    device_id: str,
    values: Sequence[Mapping[str, object]],
    allocations: Mapping[str, GlobalAllocation],
    operation: str,
    transaction_bytes: int,
    max_samples_per_value: int,
    *,
    full_traffic: bool = False,
) -> Iterator[dict[str, object]]:
    """Yield request records lazily, including full-value traffic."""
    if operation not in {"read", "write"}:
        raise GlobalMemoryMapError("sample operation must be read or write")
    if transaction_bytes <= 0 or max_samples_per_value <= 0:
        raise GlobalMemoryMapError("sample geometry must be positive")
    for raw in values:
        value_id = str(raw["value_id"])
        version = int(raw["version"])
        logical_size = int(raw["size_bytes"])
        allocation = allocations.get(value_id)
        if allocation is None or logical_size > allocation.size_bytes:
            raise GlobalMemoryMapError(f"missing Global-PA range for {value_id}")
        available_transactions = max(1, _align_up(logical_size, transaction_bytes) // transaction_bytes)
        sample_count = (
            available_transactions
            if full_traffic
            else min(max_samples_per_value, available_transactions)
        )
        indices = (
            range(available_transactions)
            if full_traffic
            else (
                [0]
                if sample_count == 1
                else [
                    (index * (available_transactions - 1)) // (sample_count - 1)
                    for index in range(sample_count)
                ]
            )
        )
        if full_traffic:
            request_indices = indices
            request_count = available_transactions
        else:
            request_indices = tuple(dict.fromkeys(indices))
            request_count = len(request_indices)
        base_share, remainder = divmod(logical_size, request_count)
        for sample_index, transaction_index in enumerate(request_indices):
            offset = transaction_index * transaction_bytes
            address = allocation.base_address + offset
            actual_bytes = min(transaction_bytes, logical_size - offset)
            if actual_bytes <= 0:
                raise GlobalMemoryMapError(f"sample escaped allocation {value_id}")
            yield {
                "task_id": task_id,
                "device_id": device_id,
                "value_id": value_id,
                "version": version,
                "operation": operation,
                "global_address": address,
                "size_bytes": actual_bytes,
                "represented_bytes": (
                    actual_bytes
                    if full_traffic
                    else base_share + (1 if sample_index < remainder else 0)
                ),
                "sample_index": sample_index,
                "sample_count": request_count,
                "logical_value_bytes": logical_size,
                "traffic_mode": "full" if full_traffic else "sampled",
            }
