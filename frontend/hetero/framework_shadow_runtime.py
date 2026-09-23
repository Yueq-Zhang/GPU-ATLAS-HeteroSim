"""Online Shadow timeline and P33 request-cycle qualification helpers.

The P31 GPU Artifact is a whole-layer reference envelope while the ATLAS
Artifact is a QKV offload candidate.  They are therefore joined as two
non-additive observation branches.  A mixed-placement makespan requires a GPU
trace that excludes the offloaded QKV work and is deliberately not inferred
here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from .live_ramulator2 import LiveRamulator2Bridge, LiveRamulator2Error


class FrameworkShadowRuntimeError(RuntimeError):
    """Raised when an online Artifact or durable-completion invariant fails."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrameworkShadowRuntimeError(f"{name} must be an object")
    return value


def _positive(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise FrameworkShadowRuntimeError(f"{name} must be positive")
    return parsed


def _gpu_replay_signature(stats: Mapping[str, object]) -> dict[str, object]:
    external = _mapping(stats.get("external_memory_stats"), "GPU external memory")
    reads = int(external.get("reads", -1))
    writes = int(external.get("writes", -1))
    parents = int(external.get("gpu_parents", -1))
    signature = {
        "cycles": _positive(stats.get("cycles"), "GPU cycles"),
        "instructions": _positive(stats.get("instructions"), "GPU instructions"),
        "duration_fs": _positive(stats.get("duration_fs"), "GPU duration"),
        "simulation_key": str(stats.get("simulation_key", "")),
        "external_memory_stats": dict(external),
    }
    if (
        not signature["simulation_key"]
        or int(external.get("instances", 0)) != 1
        or int(external.get("outstanding", -1)) != 0
        or int(external.get("address_unmapped", -1)) != 0
        or int(external.get("address_translated", 0)) <= 0
        or int(external.get("atlas_parents", -1)) != 0
        or reads < 0
        or writes < 0
        or parents != reads + writes
        or int(external.get("gpu_completed", -1)) != parents
        or int(external.get("completed", -1)) != parents
        or int(external.get("durable_completed", -1)) != parents
        or int(external.get("children_sent", -1))
        != int(external.get("children_completed", -2))
    ):
        raise FrameworkShadowRuntimeError("GPU request-cycle conservation failed")
    return signature


def iter_atlas_trace(path: Path) -> Iterator[dict[str, object]]:
    """Stream and validate one P31 compressed ATLAS Global-PA request trace."""

    expected_sequence = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise FrameworkShadowRuntimeError(
                    f"invalid ATLAS trace JSON at line {line_number}"
                ) from error
            item = _mapping(raw, f"ATLAS request {line_number}")
            sequence = int(item.get("sequence", -1))
            operation = str(item.get("command", ""))
            address = int(item.get("global_address", -1))
            size_bytes = int(item.get("size_bytes", 0))
            core_id = int(item.get("core_id", -1))
            tensor = str(item.get("tensor", ""))
            if (
                sequence != expected_sequence
                or operation not in {"read", "write"}
                or address < 0
                or size_bytes <= 0
                or core_id < 0
                or not tensor
            ):
                raise FrameworkShadowRuntimeError(
                    f"invalid ATLAS request contract at sequence {expected_sequence}"
                )
            expected_sequence += 1
            yield {
                "sequence": sequence,
                "operation": operation,
                "global_address": address,
                "size_bytes": size_bytes,
                "core_id": core_id,
                "tensor": tensor,
            }
    if expected_sequence == 0:
        raise FrameworkShadowRuntimeError("ATLAS request trace is empty")


def run_atlas_cycle_replay(
    project_root: Path,
    bridge_config: Mapping[str, object],
    trace_path: Path,
    *,
    bridge_factory: Callable[
        [Path, Mapping[str, object]], LiveRamulator2Bridge
    ] = LiveRamulator2Bridge,
) -> dict[str, object]:
    """Replay one full ATLAS request stream through one live Ramulator2 owner."""

    bridge = bridge_factory(project_root, bridge_config)
    trace = iter(iter_atlas_trace(trace_path))
    pending = next(trace, None)
    outstanding: set[int] = set()
    accepted = 0
    completed = 0
    reads = 0
    writes = 0
    logical_bytes = 0
    request_digest = hashlib.sha256()
    completion_digest = hashlib.sha256()
    first_completion: dict[str, int] | None = None
    last_completion: dict[str, int] | None = None

    try:
        while pending is not None or outstanding:
            issued_any = False
            while pending is not None:
                sequence = int(pending["sequence"])
                parent_id = sequence + 1
                result = bridge.send(
                    parent_id,
                    int(pending["global_address"]),
                    int(pending["size_bytes"]),
                    str(pending["operation"]),
                    LiveRamulator2Bridge.ATLAS_INITIATOR,
                    int(pending["core_id"]) + 1,
                    sequence,
                )
                if result == LiveRamulator2Bridge.SEND_RETRY:
                    break
                if result != LiveRamulator2Bridge.SEND_ACCEPTED:
                    raise FrameworkShadowRuntimeError(
                        f"unexpected ATLAS send result {result}"
                    )
                request_digest.update(
                    json.dumps(pending, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                outstanding.add(parent_id)
                accepted += 1
                logical_bytes += int(pending["size_bytes"])
                if pending["operation"] == "read":
                    reads += 1
                else:
                    writes += 1
                pending = next(trace, None)
                issued_any = True

            for completion in bridge.pop_completions():
                parent_id = int(completion["parent_id"])
                if (
                    int(completion["initiator"]) != LiveRamulator2Bridge.ATLAS_INITIATOR
                    or parent_id not in outstanding
                ):
                    raise FrameworkShadowRuntimeError(
                        f"unexpected ATLAS completion {parent_id}"
                    )
                outstanding.remove(parent_id)
                completed += 1
                stable_completion = {
                    key: int(completion[key])
                    for key in (
                        "parent_id",
                        "initiator",
                        "operation_code",
                        "total_children",
                        "completion_cycle",
                        "completion_time_fs",
                    )
                }
                completion_digest.update(
                    json.dumps(
                        stable_completion, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                )
                if first_completion is None:
                    first_completion = stable_completion
                last_completion = stable_completion

            if pending is not None or outstanding:
                bridge.advance_until_event(1 if issued_any else 1_000_000)
        stats = bridge.close()
    except FrameworkShadowRuntimeError:
        raise
    except LiveRamulator2Error as error:
        raise FrameworkShadowRuntimeError(str(error)) from error

    initiators = _mapping(stats.get("initiators"), "Ramulator2 initiators")
    gpu = _mapping(initiators.get("gpu0"), "GPU initiator counters")
    atlas = _mapping(initiators.get("atlas0.compute"), "ATLAS initiator counters")
    if (
        accepted != completed
        or int(stats.get("instances", 0)) != 1
        or int(stats.get("outstanding", -1)) != 0
        or int(stats.get("accepted_parent_ids", -1)) != accepted
        or int(stats.get("observed_completion_ids", -1)) != accepted
        or int(stats.get("durable_completed", -1)) != accepted
        or int(stats.get("children_sent", -1))
        != int(stats.get("children_completed", -2))
        or int(atlas.get("parents", -1)) != accepted
        or int(atlas.get("completed", -1)) != accepted
        or int(gpu.get("parents", -1)) != 0
        or int(gpu.get("completed", -1)) != 0
    ):
        raise FrameworkShadowRuntimeError("ATLAS request-cycle conservation failed")

    identity = {
        "trace_file_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "request_stream_sha256": request_digest.hexdigest(),
        "completion_stream_sha256": completion_digest.hexdigest(),
        "request_count": accepted,
        "read_requests": reads,
        "write_requests": writes,
        "logical_bytes": logical_bytes,
        "first_completion": first_completion,
        "last_completion": last_completion,
        "ramulator2": dict(stats),
    }
    return {
        "schema_version": "hetero-p33-atlas-cycle-replay/v1",
        **identity,
        "record_sha256": _digest(identity),
        "single_ramulator2": True,
        "durable_completion_qualified": True,
        "performance_claim_allowed": False,
    }


def qualify_p33_cycle_replays(
    gpu_runs: Sequence[Mapping[str, object]],
    atlas_runs: Sequence[Mapping[str, object]],
    *,
    framework_simulation_key: str,
) -> dict[str, object]:
    """Require deterministic GPU and ATLAS double replay with zero in-flight."""

    if len(gpu_runs) != 2 or len(atlas_runs) != 2 or not framework_simulation_key:
        raise FrameworkShadowRuntimeError("P33 requires exactly two runs per device")
    gpu_signatures = [_gpu_replay_signature(item) for item in gpu_runs]
    if gpu_signatures[0] != gpu_signatures[1]:
        raise FrameworkShadowRuntimeError("GPU deterministic double replay mismatch")

    atlas_fields = (
        "trace_file_sha256",
        "request_stream_sha256",
        "completion_stream_sha256",
        "request_count",
        "read_requests",
        "write_requests",
        "logical_bytes",
        "first_completion",
        "last_completion",
        "ramulator2",
    )
    atlas_signatures = []
    for item in atlas_runs:
        if (
            item.get("schema_version") != "hetero-p33-atlas-cycle-replay/v1"
            or item.get("durable_completion_qualified") is not True
            or item.get("single_ramulator2") is not True
        ):
            raise FrameworkShadowRuntimeError("invalid ATLAS cycle replay record")
        atlas_signatures.append({field: item.get(field) for field in atlas_fields})
    if atlas_signatures[0] != atlas_signatures[1]:
        raise FrameworkShadowRuntimeError("ATLAS deterministic double replay mismatch")

    payload = {
        "framework_simulation_key": framework_simulation_key,
        "gpu": {
            "deterministic_double_replay": True,
            "cycles": gpu_signatures[0]["cycles"],
            "instructions": gpu_signatures[0]["instructions"],
            "duration_fs": gpu_signatures[0]["duration_fs"],
            "external_memory_stats": gpu_signatures[0]["external_memory_stats"],
        },
        "atlas": {
            "deterministic_double_replay": True,
            **atlas_signatures[0],
        },
        "persistent_state_scope": {
            "gpu_40_kernel_trace_one_process_per_leg": True,
            "atlas_full_request_stream_one_ramulator2_per_leg": True,
            "cross_leg_state_reuse": False,
            "gpu_and_atlas_same_concurrent_ramulator2_instance": False,
        },
        "qualification": {
            "gpu_double_replay_qualified": True,
            "atlas_ramulator2_cycle_replay_qualified": True,
            "parent_child_durable_conservation": True,
            "zero_in_flight": True,
            "performance_claim_allowed": False,
        },
    }
    return {
        "schema_version": "hetero-p33-cycle-qualification/v1",
        "status": "passed",
        **payload,
        "record_sha256": _digest(payload),
    }


def build_p32_online_shadow_timeline(
    p30: Mapping[str, object],
    p31: Mapping[str, object],
    p33: Mapping[str, object],
    gpu_binding: Mapping[str, object],
    atlas_artifact: Mapping[str, object],
) -> dict[str, object]:
    """Join one real HF request and both qualified P31 branches causally."""

    if p30.get("status") != "passed" or p31.get("status") != "passed":
        raise FrameworkShadowRuntimeError("P30/P31 source qualification is absent")
    if (
        p33.get("schema_version") != "hetero-p33-cycle-qualification/v1"
        or p33.get("status") != "passed"
    ):
        raise FrameworkShadowRuntimeError("P33 cycle qualification is absent")
    simulation_key = str(p31.get("p30_simulation_key", ""))
    if (
        not simulation_key
        or p33.get("framework_simulation_key") != simulation_key
        or gpu_binding.get("schema_version") != "hetero-online-address-binding/v1"
        or atlas_artifact.get("schema_version") != "hetero-atlas-executable-artifact/v1"
    ):
        raise FrameworkShadowRuntimeError("P32 Artifact identity mismatch")

    bindings = gpu_binding.get("bindings")
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        raise FrameworkShadowRuntimeError("GPU Global PA bindings are absent")
    gpu_ranges = []
    for raw in bindings:
        item = _mapping(raw, "GPU binding")
        begin = int(item.get("physical_offset_bytes", -1))
        size = int(item.get("size_bytes", 0))
        if begin < 0 or size <= 0:
            raise FrameworkShadowRuntimeError("invalid GPU Global PA binding")
        gpu_ranges.append((begin, begin + size))
    atlas_identity = _mapping(atlas_artifact.get("identity"), "ATLAS identity")
    tensor_pa = _mapping(atlas_identity.get("tensor_global_pa"), "ATLAS tensor PA")
    source = _mapping(atlas_artifact.get("source_tensor_ir"), "ATLAS Tensor IR")
    inputs = source.get("inputs")
    outputs = source.get("outputs")
    if not isinstance(inputs, Sequence) or not isinstance(outputs, Sequence):
        raise FrameworkShadowRuntimeError("ATLAS Tensor IR tensors are absent")
    dtype = str(inputs[0]["dtype"])  # type: ignore[index]
    element_bytes = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype)
    if element_bytes is None:
        raise FrameworkShadowRuntimeError("unsupported ATLAS Tensor IR dtype")

    def tensor_bytes(raw: Mapping[str, object]) -> int:
        size = element_bytes
        for extent in raw["shape"]:  # type: ignore[index]
            size *= int(extent)
        return size

    atlas_ranges = [
        (
            int(tensor_pa[name]),
            int(tensor_pa[name]) + tensor_bytes(raw),
        )
        for name, raw in (
            ("activation", inputs[0]),  # type: ignore[index]
            ("weight", inputs[1]),  # type: ignore[index]
            ("output", outputs[0]),  # type: ignore[index]
        )
    ]
    if any(
        left_begin < right_end and right_begin < left_end
        for left_begin, left_end in gpu_ranges
        for right_begin, right_end in atlas_ranges
    ):
        raise FrameworkShadowRuntimeError("GPU and ATLAS Global PA ranges overlap")

    gpu = _mapping(p33.get("gpu"), "P33 GPU")
    atlas = _mapping(p33.get("atlas"), "P33 ATLAS")
    gpu_duration_fs = int(gpu["duration_fs"])
    atlas_duration_fs = int(
        _mapping(atlas["ramulator2"], "ATLAS stats")["global_time_fs"]
    )
    observation_barrier_fs = max(gpu_duration_fs, atlas_duration_fs)
    events = [
        {
            "sequence": 0,
            "event": "request_arrive",
            "request_id": "request-0",
            "time_fs": 0,
            "depends_on": [],
        },
        {
            "sequence": 1,
            "event": "decode_phase_begin",
            "request_id": "request-0",
            "q_len": 1,
            "kv_length": 17,
            "time_fs": 0,
            "depends_on": [0],
        },
        {
            "sequence": 2,
            "event": "gpu_reference_envelope_begin",
            "artifact_id": _mapping(p31.get("gpu"), "P31 GPU").get("artifact_id"),
            "resource": "gpu0",
            "time_fs": 0,
            "depends_on": [1],
        },
        {
            "sequence": 3,
            "event": "gpu_reference_durable",
            "resource": "shared0.dram3d",
            "time_fs": gpu_duration_fs,
            "duration_fs": gpu_duration_fs,
            "parent_count": int(
                _mapping(gpu["external_memory_stats"], "GPU external stats")[
                    "durable_completed"
                ]
            ),
            "depends_on": [2],
        },
        {
            "sequence": 4,
            "event": "atlas_qkv_candidate_begin",
            "artifact_key": atlas_artifact.get("artifact_key"),
            "resource": "atlas0.compute",
            "time_fs": 0,
            "depends_on": [1],
        },
        {
            "sequence": 5,
            "event": "atlas_qkv_candidate_durable",
            "resource": "shared0.dram3d",
            "time_fs": atlas_duration_fs,
            "duration_fs": atlas_duration_fs,
            "parent_count": int(atlas["request_count"]),
            "depends_on": [4],
        },
        {
            "sequence": 6,
            "event": "shadow_observation_barrier",
            "version": "layer0.decode.observation.v1",
            "time_fs": observation_barrier_fs,
            "depends_on": [3, 5],
        },
        {
            "sequence": 7,
            "event": "framework_version_commit",
            "framework_token_authority": "huggingface",
            "kv_state": "retained_not_released",
            "previous_version": "layer0.decode.observation.pending",
            "committed_version": "layer0.decode.observation.v1",
            "time_fs": observation_barrier_fs,
            "depends_on": [6],
        },
        {
            "sequence": 8,
            "event": "request_finish",
            "request_id": "request-0",
            "time_fs": observation_barrier_fs,
            "depends_on": [7],
        },
    ]
    for event in events:
        sequence = int(event["sequence"])
        if any(int(dependency) >= sequence for dependency in event["depends_on"]):
            raise FrameworkShadowRuntimeError("P32 dependency is not causal")
        if any(
            int(events[int(dependency)]["time_fs"]) > int(event["time_fs"])
            for dependency in event["depends_on"]
        ):
            raise FrameworkShadowRuntimeError("P32 dependency time moved backwards")

    payload = {
        "framework_simulation_key": simulation_key,
        "workload": dict(_mapping(p30.get("workload"), "P30 workload")),
        "artifact_semantics": {
            "gpu": "whole_layer_reference_envelope",
            "atlas": "qkv_projection_offload_candidate",
            "timing_aggregation": "non_additive_reference_candidate",
            "qkv_double_count_prevented": True,
            "mixed_placement_makespan_qualified": False,
        },
        "global_time": {
            "owner": "shadow0",
            "unit": "fs",
            "observation_barrier_fs": observation_barrier_fs,
            "aggregation": "max_of_non_additive_branches",
        },
        "resource_occupancy": [
            {
                "branch": "gpu_whole_layer_reference",
                "resource": "gpu0",
                "start_fs": 0,
                "end_fs": gpu_duration_fs,
            },
            {
                "branch": "gpu_whole_layer_reference",
                "resource": "shared0.dram3d",
                "start_fs": 0,
                "end_fs": gpu_duration_fs,
            },
            {
                "branch": "atlas_qkv_candidate",
                "resource": "atlas0.compute",
                "start_fs": 0,
                "end_fs": atlas_duration_fs,
            },
            {
                "branch": "atlas_qkv_candidate",
                "resource": "shared0.dram3d",
                "start_fs": 0,
                "end_fs": atlas_duration_fs,
            },
        ],
        "global_pa": {
            "memory_space_id": "shared0.dram3d",
            "gpu_ranges": [list(item) for item in gpu_ranges],
            "atlas_ranges": [list(item) for item in atlas_ranges],
            "non_overlapping": True,
        },
        "events": events,
        "qualification": {
            "online_huggingface_request_connected": True,
            "dependency_causality": True,
            "single_global_time_owner": True,
            "resource_occupancy_recorded": True,
            "global_pa_non_overlap": True,
            "durable_before_version_commit": True,
            "request_finish_after_version_commit": True,
            "kv_release_before_durable": False,
            "end_to_end_framework_timeline_qualified": True,
            "mixed_placement_makespan_qualified": False,
            "performance_claim_allowed": False,
        },
        "feedback_to_framework": False,
        "tokens_or_scheduler_mutated": False,
    }
    return {
        "schema_version": "hetero-p32-online-shadow-timeline/v1",
        "status": "passed",
        **payload,
        "record_sha256": _digest(payload),
    }
