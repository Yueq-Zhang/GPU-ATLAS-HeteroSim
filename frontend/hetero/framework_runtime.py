"""Runtime-observed framework events and executable ATLAS Artifact contracts.

Raw CUDA addresses are retained as capture evidence only.  Stable identities use
canonical storage IDs plus byte offsets, and Global PA is assigned afterwards.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence

from .inference_framework import FrameworkIntegrationError


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrameworkIntegrationError(f"{name} must be an object")
    return value


def _require_positive(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise FrameworkIntegrationError(f"{name} must be positive")
    return parsed


def _normalize_tensor(
    raw: Mapping[str, object],
    *,
    storage_ids: dict[tuple[str, int, int], str],
    storage_sizes: dict[str, int],
) -> dict[str, object]:
    tensor_id = str(raw.get("tensor_id", ""))
    if not tensor_id:
        raise FrameworkIntegrationError("runtime tensor identity is empty")
    address = int(raw.get("address", -1))
    storage_base = int(raw.get("storage_base", -1))
    storage_size = _require_positive(raw.get("storage_size_bytes", 0), "storage size")
    storage_handle = int(raw.get("storage_handle", -1))
    storage_offset = int(raw.get("storage_offset_bytes", -1))
    item_size = _require_positive(raw.get("item_size_bytes", 0), "tensor item size")
    shape = [int(item) for item in raw.get("shape", [])]  # type: ignore[arg-type]
    strides = [int(item) for item in raw.get("strides", [])]  # type: ignore[arg-type]
    if (
        address < 0
        or storage_base < 0
        or storage_offset < 0
        or len(shape) != len(strides)
        or any(item < 0 for item in shape)
        or any(item < 0 for item in strides)
        or address != storage_base + storage_offset
    ):
        raise FrameworkIntegrationError(f"invalid runtime tensor geometry: {tensor_id}")
    # An allocator can reuse one CUDA address for a new Storage with a different
    # extent during the same request.  The PyTorch Storage handle distinguishes
    # those allocation epochs, while first-observation numbering keeps raw
    # process-specific handles and addresses out of the stable identity.
    storage_key = (
        ("handle_extent", storage_handle, storage_size)
        if storage_handle >= 0
        else ("address_extent", storage_base, storage_size)
    )
    if storage_key not in storage_ids:
        storage_ids[storage_key] = f"runtime-storage-{len(storage_ids):06d}"
    storage_id = storage_ids[storage_key]
    previous_size = storage_sizes.setdefault(storage_id, storage_size)
    if previous_size != storage_size:
        raise FrameworkIntegrationError(f"storage size changed: {storage_id}")
    if shape and all(shape):
        maximum_element = sum(
            (size - 1) * stride for size, stride in zip(shape, strides)
        )
        view_end = storage_offset + (maximum_element + 1) * item_size
    else:
        view_end = storage_offset
    if view_end > storage_size:
        raise FrameworkIntegrationError(f"tensor exceeds backing storage: {tensor_id}")
    return {
        "tensor_id": tensor_id,
        "role": str(raw.get("role", "unknown")),
        "path": str(raw.get("path", "")),
        "storage_id": storage_id,
        "storage_offset_bytes": storage_offset,
        "storage_size_bytes": storage_size,
        "view_span_bytes": view_end - storage_offset,
        "logical_nbytes": int(raw.get("logical_nbytes", 0)),
        "shape": shape,
        "strides": strides,
        "dtype": str(raw.get("dtype", "")),
        "device": str(raw.get("device", "")),
        "item_size_bytes": item_size,
    }


def normalize_huggingface_runtime_observation(
    raw: Mapping[str, object],
    *,
    global_pa_base: int = 1 << 36,
    capacity_bytes: int = 1 << 40,
    alignment_bytes: int = 256,
) -> dict[str, object]:
    """Validate actual Hugging Face callbacks and bind canonical storages to PA."""

    if raw.get("schema_version") != "hetero-huggingface-runtime-observation/v1":
        raise FrameworkIntegrationError("unsupported Hugging Face runtime observation")
    if global_pa_base < 0 or capacity_bytes <= 0 or alignment_bytes <= 0:
        raise FrameworkIntegrationError("invalid runtime Global PA geometry")
    if alignment_bytes & (alignment_bytes - 1):
        raise FrameworkIntegrationError(
            "runtime Global PA alignment must be a power of two"
        )
    framework = _require_mapping(raw.get("framework"), "framework")
    model = _require_mapping(raw.get("model"), "model")
    request = _require_mapping(raw.get("request"), "request")
    if framework.get("name") != "huggingface":
        raise FrameworkIntegrationError("runtime observation is not Hugging Face")
    if (
        not framework.get("version")
        or not model.get("name")
        or not model.get("revision")
    ):
        raise FrameworkIntegrationError("framework/model identity is incomplete")
    request_id = str(request.get("request_id", ""))
    if not request_id:
        raise FrameworkIntegrationError("runtime request identity is empty")

    raw_events = raw.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise FrameworkIntegrationError("runtime events must be an array")
    storage_ids: dict[tuple[str, int, int], str] = {}
    storage_sizes: dict[str, int] = {}
    normalized_events: list[dict[str, object]] = []
    phase: str | None = None
    module_stack: list[str] = []
    request_arrived = False
    request_finished = False
    phase_counts = {"prefill": 0, "decode_step": 0}
    module_sequence: list[dict[str, object]] = []
    all_tensors: list[dict[str, object]] = []

    for expected_sequence, value in enumerate(raw_events):
        event = _require_mapping(value, f"event {expected_sequence}")
        if int(event.get("sequence", -1)) != expected_sequence:
            raise FrameworkIntegrationError("runtime event sequence is not contiguous")
        kind = str(event.get("event", ""))
        stable: dict[str, object] = {"sequence": expected_sequence, "event": kind}
        for key in (
            "request_id",
            "phase",
            "module_path",
            "module_class",
            "logical_operator",
            "fusion_group",
            "q_len",
            "kv_length",
        ):
            if key in event:
                stable[key] = event[key]
        tensors_raw = event.get("tensors", [])
        if not isinstance(tensors_raw, Sequence) or isinstance(
            tensors_raw, (str, bytes)
        ):
            raise FrameworkIntegrationError("event tensors must be an array")
        tensors = [
            _normalize_tensor(
                _require_mapping(item, "runtime tensor"),
                storage_ids=storage_ids,
                storage_sizes=storage_sizes,
            )
            for item in tensors_raw
        ]
        if tensors:
            stable["tensors"] = tensors
            all_tensors.extend(tensors)

        if kind == "request_arrive":
            if (
                expected_sequence != 0
                or request_arrived
                or event.get("request_id") != request_id
            ):
                raise FrameworkIntegrationError("invalid request_arrive event")
            request_arrived = True
        elif kind == "phase_begin":
            observed_phase = str(event.get("phase", ""))
            if (
                not request_arrived
                or phase is not None
                or observed_phase not in phase_counts
            ):
                raise FrameworkIntegrationError("invalid phase_begin event")
            phase = observed_phase
            phase_counts[observed_phase] += 1
        elif kind == "module_begin":
            module_path = str(event.get("module_path", ""))
            if phase is None or event.get("phase") != phase or not module_path:
                raise FrameworkIntegrationError("module_begin is outside its phase")
            module_stack.append(module_path)
            module_sequence.append(
                {
                    "phase": phase,
                    "module_path": module_path,
                    "module_class": str(event.get("module_class", "")),
                    "logical_operator": str(event.get("logical_operator", "")),
                    "fusion_group": str(event.get("fusion_group", "")),
                }
            )
        elif kind == "module_end":
            module_path = str(event.get("module_path", ""))
            if not module_stack or module_stack.pop() != module_path:
                raise FrameworkIntegrationError(
                    "runtime module callbacks are unbalanced"
                )
        elif kind == "phase_end":
            if phase is None or event.get("phase") != phase or module_stack:
                raise FrameworkIntegrationError("invalid phase_end event")
            phase = None
        elif kind == "sampling":
            if phase is not None or "token_id" not in event:
                raise FrameworkIntegrationError("invalid sampling event")
            stable["token_id"] = int(event["token_id"])
        elif kind == "request_finish":
            if (
                phase is not None
                or module_stack
                or event.get("request_id") != request_id
            ):
                raise FrameworkIntegrationError("invalid request_finish event")
            request_finished = True
        else:
            raise FrameworkIntegrationError(f"unknown runtime event {kind!r}")
        normalized_events.append(stable)

    if (
        not request_arrived
        or not request_finished
        or phase is not None
        or module_stack
        or phase_counts != {"prefill": 1, "decode_step": 1}
    ):
        raise FrameworkIntegrationError("runtime request/phase lifecycle is incomplete")

    storage_bases: dict[str, int] = {}
    cursor = _align_up(global_pa_base, alignment_bytes)
    limit = global_pa_base + capacity_bytes
    for storage_id in sorted(storage_sizes):
        cursor = _align_up(cursor, alignment_bytes)
        size = storage_sizes[storage_id]
        if cursor + size > limit:
            raise FrameworkIntegrationError(
                "runtime storage exceeds Global PA capacity"
            )
        storage_bases[storage_id] = cursor
        cursor += size
    tensor_bindings = {
        str(item["tensor_id"]): {
            "storage_id": item["storage_id"],
            "global_address": storage_bases[str(item["storage_id"])]
            + int(item["storage_offset_bytes"]),
            "size_bytes": item["logical_nbytes"],
            "storage_offset_bytes": item["storage_offset_bytes"],
        }
        for item in all_tensors
    }
    alias_groups: dict[str, list[str]] = {}
    for item in all_tensors:
        alias_groups.setdefault(str(item["storage_id"]), []).append(
            str(item["tensor_id"])
        )
    alias_groups = {
        key: sorted(set(value))
        for key, value in sorted(alias_groups.items())
        if len(set(value)) > 1
    }
    semantic_events: list[dict[str, object]] = []
    for event in normalized_events:
        semantic_event = dict(event)
        event_tensors = event.get("tensors")
        if isinstance(event_tensors, list):
            semantic_event["tensors"] = [
                {
                    key: value
                    for key, value in tensor.items()
                    if key
                    not in {
                        "storage_id",
                        "storage_offset_bytes",
                        "storage_size_bytes",
                        "view_span_bytes",
                    }
                }
                for tensor in event_tensors
            ]
        semantic_events.append(semantic_event)
    identity = {
        "framework": dict(framework),
        "model": dict(model),
        "request": dict(request),
        "device": dict(_require_mapping(raw.get("device"), "device")),
        "events": semantic_events,
        "module_sequence": module_sequence,
        "result": dict(_require_mapping(raw.get("result"), "result")),
    }
    memory_map = {
        "memory_space_id": "shared0.dram3d",
        "global_pa_base": global_pa_base,
        "capacity_bytes": capacity_bytes,
        "alignment_bytes": alignment_bytes,
        "storages": [
            {
                "storage_id": storage_id,
                "global_address": storage_bases[storage_id],
                "size_bytes": storage_sizes[storage_id],
            }
            for storage_id in sorted(storage_sizes)
        ],
    }
    payload = {
        "schema_version": "hetero-huggingface-runtime-normalized/v1",
        "identity": identity,
        "identity_sha256": _digest(identity),
        "runtime_events": normalized_events,
        "allocation_observation_sha256": _digest(normalized_events),
        "memory_map": memory_map,
        "memory_map_sha256": _digest(memory_map),
        "tensor_global_pa_bindings": tensor_bindings,
        "alias_groups": alias_groups,
        "actual_runtime_callbacks_observed": True,
        "raw_cuda_addresses_are_identity": False,
        "framework_remains_authoritative": True,
        "performance_claim_allowed": False,
    }
    return {**payload, "record_sha256": _digest(payload)}


def bind_huggingface_runtime_to_export(
    framework_export: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Bind one runtime-observed request to the matching P28 export identity."""

    if framework_export.get("schema_version") != "hetero-framework-export/v1":
        raise FrameworkIntegrationError("invalid framework export")
    if runtime.get("schema_version") != "hetero-huggingface-runtime-normalized/v1":
        raise FrameworkIntegrationError("invalid normalized Hugging Face runtime")
    manifests = _require_mapping(framework_export.get("manifests"), "manifests")
    execution = _require_mapping(manifests.get("execution"), "execution manifest")
    model_manifest = _require_mapping(manifests.get("model"), "model manifest")
    request_manifest = _require_mapping(manifests.get("requests"), "request manifest")
    runtime_identity = _require_mapping(runtime.get("identity"), "runtime identity")
    runtime_framework = _require_mapping(runtime_identity.get("framework"), "framework")
    runtime_model = _require_mapping(runtime_identity.get("model"), "model")
    runtime_request = _require_mapping(runtime_identity.get("request"), "request")
    exported_model = _require_mapping(model_manifest.get("model"), "exported model")
    requests = request_manifest.get("requests")
    if not isinstance(requests, Sequence) or len(requests) != 1:
        raise FrameworkIntegrationError(
            "P30 binding requires exactly one exported request"
        )
    exported_request = _require_mapping(requests[0], "exported request")
    required_equal = (
        (execution.get("framework"), "huggingface", "framework"),
        (
            execution.get("framework_version"),
            runtime_framework.get("version"),
            "version",
        ),
        (exported_model.get("name"), runtime_model.get("name"), "model name"),
        (
            exported_model.get("checkpoint_revision"),
            runtime_model.get("revision"),
            "checkpoint revision",
        ),
        (
            exported_request.get("request_id"),
            runtime_request.get("request_id"),
            "request ID",
        ),
        (
            exported_request.get("prompt_length"),
            runtime_request.get("prompt_tokens"),
            "prompt length",
        ),
        (
            exported_request.get("output_length"),
            runtime_request.get("decode_tokens"),
            "output length",
        ),
    )
    for expected, observed, name in required_equal:
        if expected != observed:
            raise FrameworkIntegrationError(f"P30 runtime/export mismatch: {name}")
    identity = {
        "framework_export_sha256": framework_export.get("identity_sha256"),
        "runtime_identity_sha256": runtime.get("identity_sha256"),
        "adapter_version": "heterosim-huggingface-runtime-adapter/v1",
    }
    return {
        "schema_version": "hetero-huggingface-online-shadow-binding/v1",
        "identity": identity,
        "simulation_key": _digest(identity),
        "runtime_memory_map_sha256": runtime.get("memory_map_sha256"),
        "runtime_tensor_global_pa_bindings": runtime["tensor_global_pa_bindings"],
        "framework_token_source": "huggingface",
        "simulator_timing_source": "gpu-atlas-heterosim",
        "feedback_to_framework": False,
        "online_runtime_observation_connected": True,
        "automatic_sass_capture_qualified": False,
        "performance_claim_allowed": False,
    }


def compile_atlas_executable_artifact(
    tensor_ir_compile: Mapping[str, object],
    *,
    tensor_global_pa: Mapping[str, int],
    compiler_version: str,
    request_bytes: int = 64,
) -> dict[str, object]:
    """Turn a validated P28 projection lowering into an executable stage program."""

    if tensor_ir_compile.get("schema_version") != "hetero-atlas-tensor-ir-compile/v1":
        raise FrameworkIntegrationError("invalid ATLAS Tensor IR compile record")
    if not tensor_ir_compile.get("full_atlas_bundle_ready") or not compiler_version:
        raise FrameworkIntegrationError("ATLAS Tensor IR is not executable")
    if request_bytes <= 0 or request_bytes & (request_bytes - 1):
        raise FrameworkIntegrationError("ATLAS request bytes must be a power of two")
    if set(tensor_global_pa) != {"activation", "weight", "output"}:
        raise FrameworkIntegrationError("ATLAS Global PA bindings must be exact")
    if any(
        int(value) < 0 or int(value) % request_bytes
        for value in tensor_global_pa.values()
    ):
        raise FrameworkIntegrationError("ATLAS Global PA bindings are not aligned")
    source = _require_mapping(tensor_ir_compile.get("source_tensor_ir"), "Tensor IR")
    inputs = source.get("inputs")
    outputs = source.get("outputs")
    if not isinstance(inputs, Sequence) or not isinstance(outputs, Sequence):
        raise FrameworkIntegrationError("ATLAS Tensor IR tensors are incomplete")
    activation_shape = [int(item) for item in inputs[0]["shape"]]  # type: ignore[index]
    weight_shape = [int(item) for item in inputs[1]["shape"]]  # type: ignore[index]
    output_shape = [int(item) for item in outputs[0]["shape"]]  # type: ignore[index]
    if len(activation_shape) != 2 or len(weight_shape) != 2 or len(output_shape) != 2:
        raise FrameworkIntegrationError("ATLAS executable only supports rank-2 GEMM")
    m_dim, k_dim = activation_shape
    weight_k, n_dim = weight_shape
    if weight_k != k_dim or output_shape != [m_dim, n_dim]:
        raise FrameworkIntegrationError("ATLAS GEMM tensor shapes disagree")
    dtype = str(inputs[0]["dtype"])  # type: ignore[index]
    bytes_per_element = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype)
    if bytes_per_element is None:
        raise FrameworkIntegrationError("unsupported ATLAS executable dtype")
    tile = _require_mapping(tensor_ir_compile.get("tile"), "tile")
    tile_m = int(tile["M"])
    tile_k = int(tile["K"])
    tile_n = int(tile["N"])
    per_core = _require_mapping(
        tensor_ir_compile.get("data_placement"), "data placement"
    ).get("per_core")
    if not isinstance(per_core, Sequence) or not per_core:
        raise FrameworkIntegrationError("ATLAS core placement is empty")
    stage_programs = []
    total_tiles = 0
    for raw_core in per_core:
        core = _require_mapping(raw_core, "core placement")
        n_begin = int(core["n_begin"])
        n_end = int(core["n_end"])
        n_tiles = (n_end - n_begin) // tile_n
        m_tiles = m_dim // tile_m
        k_tiles = k_dim // tile_k
        total_tiles += m_tiles * n_tiles * k_tiles
        stage_programs.append(
            {
                "core_id": int(core["core_id"]),
                "loop_nest": {
                    "m_tiles": m_tiles,
                    "n_begin": n_begin,
                    "n_end": n_end,
                    "n_tiles": n_tiles,
                    "k_tiles": k_tiles,
                },
                "stages": [
                    {
                        "stage": "load_activation",
                        "tensor": "activation",
                        "access": "read",
                    },
                    {"stage": "load_weight", "tensor": "weight", "access": "read"},
                    {"stage": "matrix_mac", "accumulate": True},
                    {"stage": "store_output", "tensor": "output", "access": "write"},
                ],
            }
        )
    logical_bytes = {
        "activation_read": total_tiles * tile_m * tile_k * bytes_per_element,
        "weight_read": total_tiles * tile_k * tile_n * bytes_per_element,
        "output_write": (m_dim // tile_m)
        * (n_dim // tile_n)
        * tile_m
        * tile_n
        * bytes_per_element,
    }
    identity = {
        "tensor_ir_compile_sha256": tensor_ir_compile.get("compile_sha256"),
        "compiler_version": compiler_version,
        "request_bytes": request_bytes,
        "tile": dict(tile),
        "core_count": len(per_core),
        "tensor_global_pa": {
            key: int(value) for key, value in sorted(tensor_global_pa.items())
        },
        "address_mapping": "row_major_global_pa_before_dram_decode",
    }
    artifact = {
        "schema_version": "hetero-atlas-executable-artifact/v1",
        "identity": identity,
        "artifact_key": _digest(identity),
        "source_tensor_ir": source,
        "stage_programs": stage_programs,
        "trace_descriptor": {
            "format": "hetero-atlas-memory-trace-jsonl/v1",
            "request_bytes": request_bytes,
            "ordering": "core,m_tile,n_tile,k_tile,stage,address",
        },
        "logical_bytes": logical_bytes,
        "simulation_executed": False,
        "global_pa_bound": True,
        "performance_eligible": False,
    }
    return {**artifact, "artifact_sha256": _digest(artifact)}


def _transactions(begin: int, size: int, request_bytes: int) -> range:
    first = begin // request_bytes * request_bytes
    end = _align_up(begin + size, request_bytes)
    return range(first, end, request_bytes)


def iter_atlas_memory_trace(
    artifact: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    """Yield the exact tiled Global-PA request stream for an ATLAS Artifact."""

    if artifact.get("schema_version") != "hetero-atlas-executable-artifact/v1":
        raise FrameworkIntegrationError("invalid ATLAS executable Artifact")
    identity = _require_mapping(artifact.get("identity"), "ATLAS identity")
    bindings = _require_mapping(identity.get("tensor_global_pa"), "ATLAS bindings")
    source = _require_mapping(artifact.get("source_tensor_ir"), "ATLAS Tensor IR")
    inputs = source["inputs"]  # type: ignore[index]
    activation_shape = [int(item) for item in inputs[0]["shape"]]  # type: ignore[index]
    weight_shape = [int(item) for item in inputs[1]["shape"]]  # type: ignore[index]
    m_dim, k_dim = activation_shape
    _, n_dim = weight_shape
    dtype = str(inputs[0]["dtype"])  # type: ignore[index]
    bytes_per_element = {"fp16": 2, "bf16": 2, "fp32": 4}[dtype]
    tile = _require_mapping(identity.get("tile"), "tile")
    tile_m, tile_k, tile_n = int(tile["M"]), int(tile["K"]), int(tile["N"])
    request_bytes = int(identity["request_bytes"])
    sequence = 0
    for raw_program in artifact["stage_programs"]:  # type: ignore[index]
        program = _require_mapping(raw_program, "stage program")
        loop = _require_mapping(program["loop_nest"], "loop nest")
        core_id = int(program["core_id"])
        for m_begin in range(0, m_dim, tile_m):
            for n_begin in range(int(loop["n_begin"]), int(loop["n_end"]), tile_n):
                for k_begin in range(0, k_dim, tile_k):
                    for row in range(m_begin, m_begin + tile_m):
                        begin = (
                            int(bindings["activation"])
                            + (row * k_dim + k_begin) * bytes_per_element
                        )
                        for address in _transactions(
                            begin, tile_k * bytes_per_element, request_bytes
                        ):
                            yield {
                                "sequence": sequence,
                                "core_id": core_id,
                                "command": "read",
                                "tensor": "activation",
                                "global_address": address,
                                "size_bytes": request_bytes,
                            }
                            sequence += 1
                    for row in range(k_begin, k_begin + tile_k):
                        begin = (
                            int(bindings["weight"])
                            + (row * n_dim + n_begin) * bytes_per_element
                        )
                        for address in _transactions(
                            begin, tile_n * bytes_per_element, request_bytes
                        ):
                            yield {
                                "sequence": sequence,
                                "core_id": core_id,
                                "command": "read",
                                "tensor": "weight",
                                "global_address": address,
                                "size_bytes": request_bytes,
                            }
                            sequence += 1
                for row in range(m_begin, m_begin + tile_m):
                    begin = (
                        int(bindings["output"])
                        + (row * n_dim + n_begin) * bytes_per_element
                    )
                    for address in _transactions(
                        begin, tile_n * bytes_per_element, request_bytes
                    ):
                        yield {
                            "sequence": sequence,
                            "core_id": core_id,
                            "command": "write",
                            "tensor": "output",
                            "global_address": address,
                            "size_bytes": request_bytes,
                        }
                        sequence += 1


def summarize_atlas_memory_trace(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    """Count requests and enforce sequence/alignment/range invariants."""

    identity = _require_mapping(artifact.get("identity"), "ATLAS identity")
    request_bytes = int(identity["request_bytes"])
    count = 0
    reads = 0
    writes = 0
    by_tensor: dict[str, int] = {}
    digest = hashlib.sha256()
    for expected, item in enumerate(iter_atlas_memory_trace(artifact)):
        if (
            int(item["sequence"]) != expected
            or int(item["global_address"]) % request_bytes
        ):
            raise FrameworkIntegrationError("ATLAS memory trace is not canonical")
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
        digest.update(encoded.encode("utf-8"))
        count += 1
        if item["command"] == "read":
            reads += 1
        else:
            writes += 1
        tensor = str(item["tensor"])
        by_tensor[tensor] = by_tensor.get(tensor, 0) + 1
    return {
        "request_count": count,
        "read_requests": reads,
        "write_requests": writes,
        "requests_by_tensor": dict(sorted(by_tensor.items())),
        "trace_sha256": digest.hexdigest(),
        "request_bytes": request_bytes,
        "parent_child_conservation_expected": True,
    }
