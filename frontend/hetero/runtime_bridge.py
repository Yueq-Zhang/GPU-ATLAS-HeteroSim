"""Narrow Python-to-C++ dynamic runtime boundary."""

from __future__ import annotations

from typing import Any


class RuntimeUnavailableError(RuntimeError):
    pass


def _runtime_module() -> Any:
    try:
        from . import _heterosim_runtime
    except ImportError as error:
        raise RuntimeUnavailableError(
            "C++ runtime module is not built; configure and build simulator first"
        ) from error
    return _heterosim_runtime


def simulate_token_barrier(
    requests: list[dict[str, object]], scheduler: dict[str, object]
) -> dict[str, object]:
    return _runtime_module().simulate_token_barrier(requests, scheduler)


def allocate_paged_kv(
    requests: list[dict[str, object]],
    model: dict[str, object],
    address: dict[str, object],
    memory_space_id: str,
) -> dict[str, object]:
    return _runtime_module().allocate_paged_kv(
        requests, model, address, memory_space_id
    )


def run_task_dag(tasks: list[dict[str, object]]) -> dict[str, object]:
    return _runtime_module().run_task_dag(tasks)


def ideal_link_completion_fs(
    issue_time_fs: int,
    latency_fs: int,
    payload_bytes: int,
    header_bytes: int,
    wire_bandwidth_bytes_per_second: int,
) -> int:
    return int(
        _runtime_module().ideal_link_completion_fs(
            issue_time_fs,
            latency_fs,
            payload_bytes,
            header_bytes,
            wire_bandwidth_bytes_per_second,
        )
    )
