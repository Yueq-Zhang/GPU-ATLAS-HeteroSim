"""Versioned file protocol for Accel-Sim/ATLAS external memory requests.

The protocol is intentionally transport-neutral: a patched simulator can emit
and consume the same JSONL records over files, pipes, sockets or shared memory.
This implementation provides the deterministic offline file path used before
the live stall/resume adapter is qualified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..runtime_bridge import simulate_shared_3d_memory
from ..trace_manifest import SimulationBufferBinding, TraceManifest


class MemoryBridgeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BridgeRequest:
    request_id: int
    parent_task_id: int
    initiator_id: str
    trace_address: int
    size_bytes: int
    operation: str
    issue_time_fs: int
    sequence_number: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BridgeRequest":
        if payload.get("type") != "memory_request":
            raise MemoryBridgeError("bridge record type must be memory_request")
        result = cls(
            int(payload["request_id"]),
            int(payload.get("parent_task_id", 0)),
            str(payload["initiator_id"]),
            int(str(payload["trace_address"]), 0)
            if isinstance(payload["trace_address"], str)
            else int(payload["trace_address"]),
            int(payload["size_bytes"]),
            str(payload.get("operation", "read")),
            int(payload.get("issue_time_fs", 0)),
            int(payload.get("sequence_number", payload["request_id"])),
        )
        if (
            result.request_id < 0
            or not result.initiator_id
            or result.trace_address < 0
            or result.size_bytes <= 0
            or result.issue_time_fs < 0
            or result.operation not in {"read", "write"}
        ):
            raise MemoryBridgeError("invalid memory bridge request")
        return result


def load_bindings(path: Path) -> tuple[SimulationBufferBinding, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "hetero-simulation-buffer-bindings/v1":
        raise MemoryBridgeError("invalid simulation buffer bindings schema")
    records = payload.get("bindings")
    if not isinstance(records, list) or not records:
        raise MemoryBridgeError("simulation buffer bindings must be non-empty")
    return tuple(
        SimulationBufferBinding(
            tensor_id=str(item["tensor_id"]),
            tensor_offset_bytes=int(item["tensor_offset_bytes"]),
            size_bytes=int(item["size_bytes"]),
            memory_space_id=str(item["memory_space_id"]),
            physical_offset_bytes=int(item["physical_offset_bytes"]),
        )
        for item in records
    )


def translate_requests(
    manifest: TraceManifest,
    bindings: Sequence[SimulationBufferBinding],
    requests: Sequence[BridgeRequest],
) -> list[dict[str, object]]:
    translated: list[dict[str, object]] = []
    seen: set[int] = set()
    for request in requests:
        if request.request_id in seen:
            raise MemoryBridgeError(f"duplicate bridge request id: {request.request_id}")
        seen.add(request.request_id)
        physical = manifest.translate(request.trace_address, tuple(bindings))
        normalized = manifest.normalize(request.trace_address)
        translated.append(
            {
                "request_id": request.request_id,
                "parent_task_id": request.parent_task_id,
                "initiator_id": request.initiator_id,
                "offset_bytes": physical.offset_bytes,
                "allocation_epoch": 1,
                "value_id": normalized.tensor_id,
                "value_version": 0,
                "size_bytes": request.size_bytes,
                "operation": request.operation,
                "issue_time_fs": request.issue_time_fs,
                "ordering_domain": request.parent_task_id,
                "sequence_number": request.sequence_number,
                "qos_class": 0,
            }
        )
    return translated


def run_jsonl_bridge(
    manifest_path: Path,
    bindings_path: Path,
    memory_config_path: Path,
    request_path: Path,
    response_path: Path,
) -> dict[str, object]:
    manifest = TraceManifest.load(manifest_path)
    bindings = load_bindings(bindings_path)
    memory_config = json.loads(memory_config_path.read_text(encoding="utf-8"))
    if not isinstance(memory_config, dict):
        raise MemoryBridgeError("memory configuration root must be an object")
    requests: list[BridgeRequest] = []
    for line_number, line in enumerate(
        request_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise MemoryBridgeError(
                f"invalid bridge JSONL at line {line_number}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise MemoryBridgeError(f"bridge line {line_number} must be an object")
        requests.append(BridgeRequest.from_dict(payload))
    translated = translate_requests(manifest, bindings, requests)
    result = simulate_shared_3d_memory(memory_config, translated)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    with response_path.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "bridge_header",
                    "schema_version": "hetero-memory-bridge/v1",
                    "timing_owner": result["timing_owner"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        for response in result["parent_responses"]:  # type: ignore[index]
            stream.write(
                json.dumps(
                    {"type": "memory_response", **dict(response)}, sort_keys=True
                )
                + "\n"
            )
    return result
