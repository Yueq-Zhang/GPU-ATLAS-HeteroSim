"""Live Global-PA memory execution for non-SM runtime tasks.

The runtime task path is deliberately narrow.  It models KV allocator metadata
and the explicit K/V copy performed by Prefill KV append.  Request markers are
host control events and therefore produce no device-memory traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .global_memory_map import GlobalAllocation
from .ir import ModelNode
from .live_ramulator2 import LiveRamulator2Bridge, LiveRamulator2Error
from .model_graph import ModelSpec
from .runtime_task_model import RuntimeTaskContract, RuntimeTaskEstimate


class RuntimeTaskMemoryError(RuntimeError):
    """Raised when a runtime task cannot be represented by live requests."""


@dataclass(frozen=True, slots=True)
class RuntimeTaskAddressBinding:
    task_id: str
    input_values: tuple[Mapping[str, object], ...]
    output_values: tuple[Mapping[str, object], ...]
    metadata_base_address: int | None = None
    metadata_size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeTaskMemoryResult:
    duration_fs: int
    requests: tuple[Mapping[str, object], ...]
    completions: tuple[Mapping[str, object], ...]
    statistics: Mapping[str, object]


def _chunks(
    *,
    task_id: str,
    value_id: str,
    value_version: int,
    operation: str,
    base_address: int,
    size_bytes: int,
    transaction_bytes: int,
    semantic: str,
) -> list[dict[str, object]]:
    if base_address < 0 or size_bytes <= 0 or transaction_bytes <= 0:
        raise RuntimeTaskMemoryError("invalid runtime request range")
    result: list[dict[str, object]] = []
    for offset in range(0, size_bytes, transaction_bytes):
        result.append(
            {
                "task_id": task_id,
                "value_id": value_id,
                "value_version": value_version,
                "operation": operation,
                "global_address": base_address + offset,
                "size_bytes": min(transaction_bytes, size_bytes - offset),
                "semantic": semantic,
            }
        )
    return result


def _allocation(
    raw: Mapping[str, object], allocations: Mapping[str, GlobalAllocation]
) -> GlobalAllocation:
    value_id = str(raw["value_id"])
    allocation = allocations.get(value_id)
    if allocation is None:
        raise RuntimeTaskMemoryError(f"runtime task has no Global PA for {value_id}")
    if int(raw["size_bytes"]) > allocation.size_bytes:
        raise RuntimeTaskMemoryError(
            f"runtime value exceeds Global PA allocation: {value_id}"
        )
    return allocation


def plan_runtime_task_requests(
    node: ModelNode,
    model: ModelSpec,
    contract: RuntimeTaskContract,
    estimate: RuntimeTaskEstimate,
    binding: RuntimeTaskAddressBinding,
    allocations: Mapping[str, GlobalAllocation],
) -> tuple[dict[str, object], ...]:
    """Lower an exact runtime-task semantic into 64-B Global-PA requests."""

    requests: list[dict[str, object]] = []
    if contract.model_kind == "fixed_control":
        if estimate.memory_transactions:
            raise RuntimeTaskMemoryError("host control task unexpectedly has traffic")
        return tuple()

    if contract.model_kind == "metadata_state":
        if binding.metadata_base_address is None:
            raise RuntimeTaskMemoryError(
                f"metadata task {binding.task_id} has no private Global PA workspace"
            )
        required = max(
            contract.metadata_read_bytes,
            contract.metadata_write_bytes,
        )
        if required > binding.metadata_size_bytes:
            raise RuntimeTaskMemoryError("metadata workspace is too small")
        if contract.metadata_read_bytes:
            requests.extend(
                _chunks(
                    task_id=binding.task_id,
                    value_id=f"workspace:{binding.task_id}:metadata",
                    value_version=0,
                    operation="read",
                    base_address=binding.metadata_base_address,
                    size_bytes=contract.metadata_read_bytes,
                    transaction_bytes=contract.transaction_bytes,
                    semantic=f"{node.op}.metadata_read",
                )
            )
        if contract.metadata_write_bytes:
            requests.extend(
                _chunks(
                    task_id=binding.task_id,
                    value_id=f"workspace:{binding.task_id}:metadata",
                    value_version=1,
                    operation="write",
                    base_address=binding.metadata_base_address,
                    size_bytes=contract.metadata_write_bytes,
                    transaction_bytes=contract.transaction_bytes,
                    semantic=f"{node.op}.metadata_write",
                )
            )
    elif contract.model_kind == "kv_copy_engine":
        if len(binding.input_values) != 3 or len(binding.output_values) != 3:
            raise RuntimeTaskMemoryError(
                "kv_append requires positioned/K/V inputs and query/K/V outputs"
            )
        positioned = _allocation(binding.input_values[0], allocations)
        key_output = _allocation(binding.output_values[1], allocations)
        value_output = _allocation(binding.output_values[2], allocations)
        batch = int(node.attributes.get("batch_size", 1))
        q_len = int(node.attributes.get("q_len", 1))
        past_len = int(node.attributes.get("past_kv_len", 0))
        query_bytes = batch * q_len * model.hidden_size * model.bytes_per_element
        one_kv_bytes = (
            batch
            * q_len
            * model.num_kv_heads
            * model.head_dim
            * model.bytes_per_element
        )
        destination_offset = (
            batch
            * past_len
            * model.num_kv_heads
            * model.head_dim
            * model.bytes_per_element
        )
        if query_bytes + 2 * one_kv_bytes > positioned.size_bytes:
            raise RuntimeTaskMemoryError("packed QKV source range is too small")
        if destination_offset + one_kv_bytes > min(
            key_output.size_bytes, value_output.size_bytes
        ):
            raise RuntimeTaskMemoryError("KV destination range is too small")
        source_version = int(binding.input_values[0]["version"])
        for value_id, address, semantic in (
            (
                str(binding.input_values[0]["value_id"]),
                positioned.base_address + query_bytes,
                "kv_append.key_source",
            ),
            (
                str(binding.input_values[0]["value_id"]),
                positioned.base_address + query_bytes + one_kv_bytes,
                "kv_append.value_source",
            ),
        ):
            requests.extend(
                _chunks(
                    task_id=binding.task_id,
                    value_id=value_id,
                    value_version=source_version,
                    operation="read",
                    base_address=address,
                    size_bytes=one_kv_bytes,
                    transaction_bytes=contract.transaction_bytes,
                    semantic=semantic,
                )
            )
        for raw, allocation, semantic in (
            (binding.output_values[1], key_output, "kv_append.key_destination"),
            (binding.output_values[2], value_output, "kv_append.value_destination"),
        ):
            requests.extend(
                _chunks(
                    task_id=binding.task_id,
                    value_id=str(raw["value_id"]),
                    value_version=int(raw["version"]),
                    operation="write",
                    base_address=allocation.base_address + destination_offset,
                    size_bytes=one_kv_bytes,
                    transaction_bytes=contract.transaction_bytes,
                    semantic=semantic,
                )
            )
    else:
        raise RuntimeTaskMemoryError(
            f"unsupported live runtime model {contract.model_kind}"
        )

    read_bytes = sum(
        int(item["size_bytes"])
        for item in requests
        if item["operation"] == "read"
    )
    write_bytes = sum(
        int(item["size_bytes"])
        for item in requests
        if item["operation"] == "write"
    )
    if (
        read_bytes != estimate.memory_read_bytes
        or write_bytes != estimate.memory_write_bytes
        or len(requests) != estimate.memory_transactions
    ):
        raise RuntimeTaskMemoryError(
            "runtime request plan disagrees with byte/transaction contract"
        )
    return tuple(requests)


def run_runtime_task_memory(
    project_root: Path,
    bridge_config: Mapping[str, object],
    contract: RuntimeTaskContract,
    estimate: RuntimeTaskEstimate,
    requests: Sequence[Mapping[str, object]],
    *,
    bridge_factory: Callable[
        [Path, Mapping[str, object]], LiveRamulator2Bridge
    ] = LiveRamulator2Bridge,
) -> RuntimeTaskMemoryResult:
    """Execute ordered reads then durable writes through one live bridge."""

    if not requests:
        return RuntimeTaskMemoryResult(
            duration_fs=estimate.duration_fs,
            requests=tuple(),
            completions=tuple(),
            statistics={
                "instances": 0,
                "accepted_parent_ids": 0,
                "observed_completion_ids": 0,
                "outstanding": 0,
                "host_control_only": True,
            },
        )
    bridge = bridge_factory(project_root, bridge_config)
    issued: list[dict[str, object]] = []
    completions: list[dict[str, object]] = []
    next_parent_id = 1

    try:
        for operation in ("read", "write"):
            phase = [item for item in requests if item["operation"] == operation]
            pending_index = 0
            outstanding: set[int] = set()
            next_issue_cycle = bridge.current_cycle
            while pending_index < len(phase) or outstanding:
                progress = False
                if (
                    pending_index < len(phase)
                    and bridge.current_cycle >= next_issue_cycle
                ):
                    item = phase[pending_index]
                    result = bridge.send(
                        next_parent_id,
                        int(item["global_address"]),
                        int(item["size_bytes"]),
                        operation,
                        LiveRamulator2Bridge.GPU_INITIATOR,
                        1,
                        next_parent_id,
                    )
                    if result == LiveRamulator2Bridge.SEND_ACCEPTED:
                        record = {
                            "parent_id": next_parent_id,
                            "issue_cycle": bridge.current_cycle,
                            "issue_time_fs": bridge.global_time_fs,
                            **dict(item),
                        }
                        issued.append(record)
                        outstanding.add(next_parent_id)
                        pending_index += 1
                        next_parent_id += 1
                        next_issue_cycle = (
                            bridge.current_cycle + contract.cycles_per_transaction
                        )
                        progress = True
                for completion in bridge.pop_completions():
                    parent_id = int(completion["parent_id"])
                    if parent_id not in outstanding:
                        raise RuntimeTaskMemoryError(
                            f"unexpected runtime completion {parent_id}"
                        )
                    outstanding.remove(parent_id)
                    completions.append(completion)
                    progress = True
                if pending_index < len(phase) or outstanding:
                    wait_cycles = max(1, next_issue_cycle - bridge.current_cycle)
                    bridge.advance_until_event(wait_cycles if progress else 1_000_000)
        final_stats = bridge.close()
    except RuntimeTaskMemoryError:
        raise
    except LiveRamulator2Error as error:
        raise RuntimeTaskMemoryError(str(error)) from error
    except Exception as error:  # pragma: no cover - defensive ABI boundary
        raise RuntimeTaskMemoryError(str(error)) from error

    gpu = final_stats.get("initiators", {}).get("gpu0", {})
    atlas = final_stats.get("initiators", {}).get("atlas0.compute", {})
    accepted = len(issued)
    if (
        int(final_stats.get("instances", 0)) != 1
        or int(final_stats.get("outstanding", -1)) != 0
        or int(final_stats.get("accepted_parent_ids", -1)) != accepted
        or int(final_stats.get("observed_completion_ids", -1)) != accepted
        or int(gpu.get("parents", -1)) != accepted
        or int(gpu.get("completed", -1)) != accepted
        or int(atlas.get("parents", -1)) != 0
    ):
        raise RuntimeTaskMemoryError("live runtime request conservation failed")
    gpu_clock_hz = int(bridge_config["gpu_clock_hz"])
    fixed_duration_fs = (
        contract.fixed_cycles * 1_000_000_000_000_000 + gpu_clock_hz - 1
    ) // gpu_clock_hz
    return RuntimeTaskMemoryResult(
        duration_fs=int(final_stats["global_time_fs"]) + fixed_duration_fs,
        requests=tuple(issued),
        completions=tuple(completions),
        statistics={
            **dict(final_stats),
            "runtime_fixed_cycles": contract.fixed_cycles,
            "runtime_fixed_duration_fs": fixed_duration_fs,
            "request_count": accepted,
            "read_request_count": sum(
                1 for item in issued if item["operation"] == "read"
            ),
            "write_request_count": sum(
                1 for item in issued if item["operation"] == "write"
            ),
        },
    )
