"""P23 exact-BS=2, one-layer Decode Trace timeline construction.

The scheduler-facing batch contains two logical requests, while the device
graph contains one fused BS=2 kernel instance per operator.  GPU durations and
memory stalls are obtained by replaying the exact P23 traces after binding
them into one Global-PA map.  Control, allocator and KV-append work uses the
existing live Ramulator2 runtime-task path and remains performance uncalibrated.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from .global_memory_map import GlobalAllocation, build_global_memory_map
from .ir import ModelGraph, ModelNode, NodeKind, Phase, StorageClass, Value
from .model_graph import ModelSpec, RequestSpec, graph_to_dict, model_spec_from_config
from .online_operator_runtime import OnlineDispatchSpec, run_online_operator_dag
from .operator_artifact import OperatorArtifactManifest
from .operator_event import BackendTaskResult, OperatorEventDispatcher
from .placement import place_nodes
from .runner import _execution_graph, _git_revision, _materialize_residency, _write_json

CHECKPOINT = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
BATCH_REQUEST_ID = "P23-BS2-DECODE-R0-R1"
MEMBER_REQUEST_IDS = ("R0", "R1")
GPU_OPERATORS = (
    "token_embedding",
    "attention_norm",
    "qkv_projection",
    "rope",
    "causal_attention",
    "output_projection",
    "residual_add",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
)
RUNTIME_OPERATORS = (
    "request_start",
    "kv_allocate",
    "kv_append",
    "request_finish",
    "kv_release",
)


class P23TimelineError(RuntimeError):
    """Raised when the exact batched timeline contract is incomplete."""


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P23TimelineError(f"{path} must contain a JSON object")
    return payload


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (project_root / path).resolve()


def _value(
    value_id: str,
    shape: tuple[int | str, ...],
    storage: StorageClass,
    *,
    dtype: str = "fp16",
    mutable: bool = False,
) -> Value:
    return Value(
        value_id=value_id,
        shape_expr=shape,
        dtype=dtype,
        layout="row_major",
        storage_class=storage,
        mutable=mutable,
        lifetime="static" if storage is StorageClass.PARAMETER else "request",
    )


def build_graph(model: ModelSpec) -> ModelGraph:
    """Build the exact fused-BS=2, one-layer, one-step logical graph."""

    if (
        model.name != "TinyLlama-1.1B"
        or model.checkpoint_revision != CHECKPOINT
        or model.num_layers != 1
        or model.hidden_size != 2048
        or model.intermediate_size != 5632
        or model.num_kv_heads != 4
        or model.head_dim != 64
        or model.vocab_size != 32000
        or model.dtype != "fp16"
    ):
        raise P23TimelineError("P23 graph requires the exact TinyLlama one-layer model")

    batch = 2
    q_len = 1
    past_len = 16
    kv_len = 17
    h = model.hidden_size
    i = model.intermediate_size
    kv_width = model.num_kv_heads * model.head_dim
    prefix = BATCH_REQUEST_ID
    common = {
        "batch_size": batch,
        "context_length": past_len,
        "q_len": q_len,
        "past_kv_len": past_len,
        "attention_kv_len": kv_len,
        "batch_member_request_ids": list(MEMBER_REQUEST_IDS),
    }

    values = [
        _value(
            "model.tok_embeddings.weight", (model.vocab_size, h), StorageClass.PARAMETER
        ),
        _value("model.layers.0.attention_norm.weight", (h,), StorageClass.PARAMETER),
        _value("model.layers.0.q.weight", (h, h), StorageClass.PARAMETER),
        _value("model.layers.0.k.weight", (kv_width, h), StorageClass.PARAMETER),
        _value("model.layers.0.v.weight", (kv_width, h), StorageClass.PARAMETER),
        _value("model.layers.0.output.weight", (h, h), StorageClass.PARAMETER),
        _value("model.layers.0.mlp_norm.weight", (h,), StorageClass.PARAMETER),
        _value("model.layers.0.gate.weight", (i, h), StorageClass.PARAMETER),
        _value("model.layers.0.up.weight", (i, h), StorageClass.PARAMETER),
        _value("model.layers.0.down.weight", (h, i), StorageClass.PARAMETER),
        _value("model.final_norm.weight", (h,), StorageClass.PARAMETER),
        _value("model.lm_head.weight", (model.vocab_size, h), StorageClass.PARAMETER),
        _value(
            f"{prefix}.token_ids", (batch, q_len), StorageClass.METADATA, dtype="int64"
        ),
        _value(
            f"{prefix}.hidden.embedding", (batch, q_len, h), StorageClass.ACTIVATION
        ),
        _value(
            f"{prefix}.hidden.attention_norm",
            (batch, q_len, h),
            StorageClass.ACTIVATION,
        ),
        _value(f"{prefix}.q", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(f"{prefix}.k_new", (batch, q_len, kv_width), StorageClass.ACTIVATION),
        _value(f"{prefix}.v_new", (batch, q_len, kv_width), StorageClass.ACTIVATION),
        _value(f"{prefix}.q_rot", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(f"{prefix}.k_rot", (batch, q_len, kv_width), StorageClass.ACTIVATION),
        _value(
            f"{prefix}.kv.l0.k",
            (batch, "kv_tokens", model.num_kv_heads, model.head_dim),
            StorageClass.KV_CACHE,
            mutable=True,
        ),
        _value(
            f"{prefix}.kv.l0.v",
            (batch, "kv_tokens", model.num_kv_heads, model.head_dim),
            StorageClass.KV_CACHE,
            mutable=True,
        ),
        _value(f"{prefix}.query", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(
            f"{prefix}.attention.context", (batch, q_len, h), StorageClass.ACTIVATION
        ),
        _value(
            f"{prefix}.attention.projected", (batch, q_len, h), StorageClass.ACTIVATION
        ),
        _value(
            f"{prefix}.attention.residual", (batch, q_len, h), StorageClass.ACTIVATION
        ),
        _value(f"{prefix}.hidden.mlp_norm", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(f"{prefix}.mlp.gate", (batch, q_len, i), StorageClass.ACTIVATION),
        _value(f"{prefix}.mlp.up", (batch, q_len, i), StorageClass.ACTIVATION),
        _value(f"{prefix}.mlp.activated", (batch, q_len, i), StorageClass.ACTIVATION),
        _value(f"{prefix}.mlp.down", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(f"{prefix}.layer.output", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(f"{prefix}.final_norm", (batch, q_len, h), StorageClass.ACTIVATION),
        _value(
            f"{prefix}.logits",
            (batch, q_len, model.vocab_size),
            StorageClass.ACTIVATION,
        ),
        _value(f"{prefix}.token", (batch,), StorageClass.METADATA, dtype="int64"),
    ]

    nodes: list[ModelNode] = []
    previous: str | None = None

    def add(
        suffix: str,
        kind: NodeKind,
        op: str,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        *,
        layer_id: int | None = None,
        operator_group: str = "runtime",
    ) -> None:
        nonlocal previous
        node_id = f"{prefix}.{suffix}"
        nodes.append(
            ModelNode(
                node_id=node_id,
                kind=kind,
                op=op,
                phase=Phase.DECODE
                if op in GPU_OPERATORS or op == "kv_append"
                else Phase.CONTROL,
                layer_id=layer_id,
                step_id=0,
                dependencies=() if previous is None else (previous,),
                read_values=reads,
                write_values=writes,
                attributes={"operator_group": operator_group, **common},
            )
        )
        previous = node_id

    k_cache = f"{prefix}.kv.l0.k"
    v_cache = f"{prefix}.kv.l0.v"
    add("request_start", NodeKind.CONTROL, "request_start")
    add("kv_allocate", NodeKind.STATE, "kv_allocate", writes=(k_cache, v_cache))
    add(
        "token_embedding",
        NodeKind.COMPUTE,
        "token_embedding",
        (f"{prefix}.token_ids", "model.tok_embeddings.weight"),
        (f"{prefix}.hidden.embedding",),
        operator_group="embedding",
    )
    add(
        "l0.attention_norm",
        NodeKind.COMPUTE,
        "attention_norm",
        (f"{prefix}.hidden.embedding", "model.layers.0.attention_norm.weight"),
        (f"{prefix}.hidden.attention_norm",),
        layer_id=0,
        operator_group="norm",
    )
    add(
        "l0.qkv_projection",
        NodeKind.COMPUTE,
        "qkv_projection",
        (
            f"{prefix}.hidden.attention_norm",
            "model.layers.0.q.weight",
            "model.layers.0.k.weight",
            "model.layers.0.v.weight",
        ),
        (f"{prefix}.q", f"{prefix}.k_new", f"{prefix}.v_new"),
        layer_id=0,
        operator_group="attention",
    )
    add(
        "l0.rope",
        NodeKind.COMPUTE,
        "rope",
        (f"{prefix}.q", f"{prefix}.k_new"),
        (f"{prefix}.q_rot", f"{prefix}.k_rot"),
        layer_id=0,
        operator_group="attention",
    )
    add(
        "l0.kv_append",
        NodeKind.STATE,
        "kv_append",
        (f"{prefix}.q_rot", f"{prefix}.k_rot", f"{prefix}.v_new", k_cache, v_cache),
        (f"{prefix}.query", k_cache, v_cache),
        layer_id=0,
        operator_group="attention",
    )
    add(
        "l0.causal_attention",
        NodeKind.COMPUTE,
        "causal_attention",
        (f"{prefix}.query", k_cache, v_cache),
        (f"{prefix}.attention.context",),
        layer_id=0,
        operator_group="attention",
    )
    add(
        "l0.output_projection",
        NodeKind.COMPUTE,
        "output_projection",
        (f"{prefix}.attention.context", "model.layers.0.output.weight"),
        (f"{prefix}.attention.projected",),
        layer_id=0,
        operator_group="attention",
    )
    add(
        "l0.attention_residual",
        NodeKind.COMPUTE,
        "residual_add",
        (f"{prefix}.hidden.embedding", f"{prefix}.attention.projected"),
        (f"{prefix}.attention.residual",),
        layer_id=0,
        operator_group="attention",
    )
    add(
        "l0.mlp_norm",
        NodeKind.COMPUTE,
        "mlp_norm",
        (f"{prefix}.attention.residual", "model.layers.0.mlp_norm.weight"),
        (f"{prefix}.hidden.mlp_norm",),
        layer_id=0,
        operator_group="mlp",
    )
    add(
        "l0.gate_up_projection",
        NodeKind.COMPUTE,
        "gate_up_projection",
        (
            f"{prefix}.hidden.mlp_norm",
            "model.layers.0.gate.weight",
            "model.layers.0.up.weight",
        ),
        (f"{prefix}.mlp.gate", f"{prefix}.mlp.up"),
        layer_id=0,
        operator_group="mlp",
    )
    add(
        "l0.silu_multiply",
        NodeKind.COMPUTE,
        "silu_multiply",
        (f"{prefix}.mlp.gate", f"{prefix}.mlp.up"),
        (f"{prefix}.mlp.activated",),
        layer_id=0,
        operator_group="mlp",
    )
    add(
        "l0.down_projection",
        NodeKind.COMPUTE,
        "down_projection",
        (f"{prefix}.mlp.activated", "model.layers.0.down.weight"),
        (f"{prefix}.mlp.down",),
        layer_id=0,
        operator_group="mlp",
    )
    add(
        "l0.mlp_residual",
        NodeKind.COMPUTE,
        "residual_add",
        (f"{prefix}.attention.residual", f"{prefix}.mlp.down"),
        (f"{prefix}.layer.output",),
        layer_id=0,
        operator_group="mlp",
    )
    add(
        "final_norm",
        NodeKind.COMPUTE,
        "final_norm",
        (f"{prefix}.layer.output", "model.final_norm.weight"),
        (f"{prefix}.final_norm",),
        operator_group="finalize",
    )
    add(
        "lm_head",
        NodeKind.COMPUTE,
        "lm_head",
        (f"{prefix}.final_norm", "model.lm_head.weight"),
        (f"{prefix}.logits",),
        operator_group="finalize",
    )
    add(
        "sampling",
        NodeKind.COMPUTE,
        "sampling",
        (f"{prefix}.logits",),
        (f"{prefix}.token",),
        operator_group="finalize",
    )
    add("request_finish", NodeKind.CONTROL, "request_finish")
    add("kv_release", NodeKind.STATE, "kv_release")
    graph = ModelGraph("hetero-model-graph/v2", tuple(values), tuple(nodes))
    graph.validate()
    return graph


def _binding(
    tensor_id: str,
    source: str,
    index: int,
    offset: int = 0,
    *,
    shadow: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "tensor_id": tensor_id,
        "source": source,
        "index": index,
        "value_offset_bytes": offset,
    }
    if shadow:
        result["binding_mode"] = "external_input_widened_shadow"
    return result


def value_bindings(operator: str) -> list[dict[str, object]]:
    prefix = f"tinyllama.decode.kv17.layer0.{operator}"
    if operator == "token_embedding":
        return [
            _binding(f"{prefix}.token_ids", "input", 0, shadow=True),
            _binding(f"{prefix}.weight", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator in {"attention_norm", "mlp_norm", "final_norm"}:
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.weight", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "qkv_projection":
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.q_weight", "input", 1),
            _binding(f"{prefix}.k_weight", "input", 2),
            _binding(f"{prefix}.v_weight", "input", 3),
            _binding(f"{prefix}.query", "output", 0),
            _binding(f"{prefix}.key", "output", 1),
            _binding(f"{prefix}.value", "output", 2),
        ]
    if operator == "rope":
        return [
            _binding(f"{prefix}.query", "input", 0),
            _binding(f"{prefix}.key", "input", 1),
            _binding(f"{prefix}.query_output", "output", 0),
            _binding(f"{prefix}.key_output", "output", 1),
        ]
    if operator == "causal_attention":
        return [
            _binding(f"{prefix}.query", "input", 0),
            _binding(f"{prefix}.key", "input", 1, shadow=True),
            _binding(f"{prefix}.value", "input", 2, shadow=True),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator in {"output_projection", "down_projection", "lm_head"}:
        output_name = "logits" if operator == "lm_head" else "output"
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.weight", "input", 1),
            _binding(f"{prefix}.{output_name}", "output", 0),
        ]
    if operator == "residual_add":
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.residual", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "gate_up_projection":
        return [
            _binding(f"{prefix}.input", "input", 0),
            _binding(f"{prefix}.gate_weight", "input", 1),
            _binding(f"{prefix}.up_weight", "input", 2),
            _binding(f"{prefix}.gate", "output", 0),
            _binding(f"{prefix}.up", "output", 1),
        ]
    if operator == "silu_multiply":
        return [
            _binding(f"{prefix}.gate", "input", 0),
            _binding(f"{prefix}.up", "input", 1),
            _binding(f"{prefix}.output", "output", 0),
        ]
    if operator == "sampling":
        return [
            _binding(f"{prefix}.logits", "input", 0),
            _binding(f"{prefix}.token", "output", 0),
        ]
    raise P23TimelineError(f"P23 value bindings are not defined for {operator}")


def _artifact_name(operator: str, *, coupled: bool) -> str:
    suffix = "_shared_hbdram_range_rebase" if coupled else "_trace"
    return f"tinyllama_decode_bs2_ctx16_kv17_{operator}_sm89{suffix}.json"


def build_backends(
    config: Mapping[str, object], project_root: Path
) -> dict[str, object]:
    execution = config["execution"]
    if not isinstance(execution, Mapping):
        raise P23TimelineError("execution must be an object")
    source_root = _resolve(project_root, execution["source_root"])
    coupled_root = _resolve(project_root, execution["coupled_root"])
    bindings = []
    for operator in GPU_OPERATORS:
        bindings.append(
            {
                "selector": {"phase": "decode", "op": operator, "step_id": 0},
                "trace_manifest": str(
                    source_root / _artifact_name(operator, coupled=False)
                ),
                "operator_artifact": str(
                    coupled_root / _artifact_name(operator, coupled=True)
                ),
                "compatibility": "exact_operator",
                "contract_overrides": {"layer_id": 0},
                "value_bindings": value_bindings(operator),
            }
        )
    return {
        "gpu": {
            "kind": "accel_sim",
            "requested_timing_mode": "coupled",
            "config_ref": str(execution["gpu_backend_ref"]),
            "require_request_cycle_ready": True,
            "resource_bindings": {
                "gpu_core": "gpu0.core",
                "gpu_l1": "gpu0.l1",
                "gpu_l2": "gpu0.l2",
                "gpu_noc": "gpu0.noc",
                "shared_3d_dram": "shared0.dram3d",
            },
            "trace_bindings": bindings,
            "fallback_kind": "none",
            "runtime_task_model_ref": str(execution["runtime_task_model_ref"]),
            "runtime_task_operators": list(RUNTIME_OPERATORS),
            "parameter_source": "P23 remote RTX4090 exact BS2 SM89 capture, replayed as SM86",
        },
        "atlas": {"kind": "none"},
        "host": {"kind": "none"},
    }


def validate_ready_catalog(config: Mapping[str, object], project_root: Path) -> None:
    execution = config["execution"]
    if not isinstance(execution, Mapping):
        raise P23TimelineError("execution must be an object")
    catalog = _load(_resolve(project_root, execution["ready_catalog_ref"]))
    records = catalog.get("records")
    if (
        catalog.get("schema_version") != "hetero-p23-bs2-decode-ready-catalog/v1"
        or not isinstance(records, list)
        or len(records) != len(GPU_OPERATORS)
    ):
        raise P23TimelineError("P23 Ready Catalog is incomplete")
    by_operator = {str(item["operator_type"]): item for item in records}
    if set(by_operator) != set(GPU_OPERATORS):
        raise P23TimelineError("P23 Ready Catalog operator set is not exact")
    coupled_root = _resolve(project_root, execution["coupled_root"])
    source_root = _resolve(project_root, execution["source_root"])
    for operator in GPU_OPERATORS:
        record = by_operator[operator]
        if any(
            record.get(key) is not expected
            for key, expected in (
                ("double_run_qualified", True),
                ("range_rebase_ready", True),
                ("global_pa_binding_ready", True),
                ("request_cycle_ready", True),
                ("performance_eligible", False),
            )
        ):
            raise P23TimelineError(f"{operator} is not request-cycle ready")
        artifact_path = coupled_root / _artifact_name(operator, coupled=True)
        trace_path = source_root / _artifact_name(operator, coupled=False)
        if not artifact_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError(
                artifact_path if not artifact_path.is_file() else trace_path
            )
        artifact = OperatorArtifactManifest.load(artifact_path)
        key = artifact.compatibility_key
        if (
            not artifact.request_cycle_ready
            or key.operator != operator
            or key.batch_size != 2
            or key.context_length != 16
            or key.q_len != 1
            or key.kv_length != 17
        ):
            raise P23TimelineError(f"{operator} Artifact identity is not exact")


def _member_kv_slices(
    allocations: Mapping[str, GlobalAllocation], model: ModelSpec
) -> list[dict[str, object]]:
    per_member = 17 * model.num_kv_heads * model.head_dim * model.bytes_per_element
    result = []
    for kind in ("k", "v"):
        value_id = f"{BATCH_REQUEST_ID}.kv.l0.{kind}"
        allocation = allocations[value_id]
        if allocation.size_bytes != per_member * len(MEMBER_REQUEST_IDS):
            raise P23TimelineError(
                f"unexpected batched KV allocation size for {value_id}"
            )
        for member_index, request_id in enumerate(MEMBER_REQUEST_IDS):
            base = allocation.base_address + member_index * per_member
            result.append(
                {
                    "request_id": request_id,
                    "value_id": value_id,
                    "kind": kind,
                    "member_index": member_index,
                    "base_address": base,
                    "end_address_exclusive": base + per_member,
                    "size_bytes": per_member,
                    "initial_kv_length": 16,
                    "committed_kv_length": 17,
                }
            )
    return result


def _task_by_op(
    execution_graph: Mapping[str, object], op: str, occurrence: int = 0
) -> Mapping[str, object]:
    tasks = [item for item in execution_graph["tasks"] if item["op"] == op]  # type: ignore[index]
    return tasks[occurrence]


def _request_lifecycle(execution_graph: Mapping[str, object]) -> dict[str, object]:
    start = _task_by_op(execution_graph, "request_start")["timing"]
    allocate = _task_by_op(execution_graph, "kv_allocate")["timing"]
    append = _task_by_op(execution_graph, "kv_append")["timing"]
    sampling = _task_by_op(execution_graph, "sampling")["timing"]
    finish = _task_by_op(execution_graph, "request_finish")["timing"]
    release = _task_by_op(execution_graph, "kv_release")["timing"]
    records = []
    events = []
    for request_id in MEMBER_REQUEST_IDS:
        records.append(
            {
                "request_id": request_id,
                "final_state": "FINISHED",
                "arrival_time_fs": 0,
                "admission_time_fs": int(start["start_time_fs"]),
                "allocation_complete_time_fs": int(allocate["completion_time_fs"]),
                "kv_append_complete_time_fs": int(append["completion_time_fs"]),
                "token_ready_time_fs": int(sampling["completion_time_fs"]),
                "finish_time_fs": int(finish["completion_time_fs"]),
                "retire_time_fs": int(release["completion_time_fs"]),
                "initial_kv_length": 16,
                "final_committed_kv_length": 17,
                "initial_kv_version": 0,
                "final_kv_version": 1,
                "generated_tokens": 1,
            }
        )
        for event, timing in (
            ("ADMIT", start),
            ("ALLOCATE_KV", allocate),
            ("COMMIT_KV_APPEND", append),
            ("TOKEN_READY", sampling),
            ("FINISH", finish),
            ("RETIRE_AND_RELEASE", release),
        ):
            events.append(
                {
                    "request_id": request_id,
                    "event": event,
                    "time_fs": int(
                        timing["start_time_fs"]
                        if event == "ADMIT"
                        else timing["completion_time_fs"]
                    ),
                }
            )
    events.sort(key=lambda item: (int(item["time_fs"]), str(item["request_id"])))
    return {
        "schema_version": "hetero-p23-bs2-request-lifecycle/v1",
        "requests": records,
        "events": events,
        "version_commits": [
            {
                "request_id": request_id,
                "from_version": 0,
                "to_version": 1,
                "committed_kv_length": 17,
                "cause_task_id": _task_by_op(execution_graph, "kv_append")["task_id"],
                "commit_time_fs": int(append["completion_time_fs"]),
            }
            for request_id in MEMBER_REQUEST_IDS
        ],
        "zero_in_flight": True,
    }


def _prepare_timeline(config_path: Path, output_dir: Path) -> tuple[object, ...]:
    project_root = Path(__file__).resolve().parents[2]
    config = _load(config_path.resolve())
    if config.get("schema_version") != "hetero-p23-bs2-decode-timeline/v1":
        raise P23TimelineError("invalid P23 timeline config schema")
    validate_ready_catalog(config, project_root)
    model_ref = _resolve(project_root, config["model_ref"])
    model = model_spec_from_config(_load(model_ref))
    graph = build_graph(model)
    placement = _load(_resolve(project_root, config["placement_ref"]))
    address = _load(_resolve(project_root, config["address_ref"]))
    backends = build_backends(config, project_root)
    request = RequestSpec(
        BATCH_REQUEST_ID,
        16,
        1,
        execution_scope="decode_step",
        initial_kv_length=16,
    )
    decisions = place_nodes(graph.nodes, placement, active_batch=2)
    execution_graph, _runtime_tasks, specs = _execution_graph(
        [(graph, decisions, request)],
        "model3_gpu_native_3ddram",
        "shared",
        model,
        backends,
        {},
        "operator_event",
    )
    for task in execution_graph["tasks"]:  # type: ignore[index]
        task["batch_member_request_ids"] = list(MEMBER_REQUEST_IDS)
        task["fused_batch_size"] = 2
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatcher = OperatorEventDispatcher(
        project_root, output_dir / "backend_runs", backends
    )
    capacity = int(address["global_pa_capacity_bytes"])
    alignment = int(address["allocation_alignment_bytes"])
    allocations, memory_map = build_global_memory_map(
        execution_graph, "shared0.dram3d", capacity, alignment
    )
    memory_map["batch_member_kv_slices"] = _member_kv_slices(allocations, model)
    dispatcher.configure_global_pa_bindings(
        allocations,
        specs,
        memory_map,
        capacity_bytes=capacity,
        alignment_bytes=alignment,
    )
    return (
        project_root,
        config,
        model,
        graph,
        execution_graph,
        specs,
        dispatcher,
        memory_map,
        output_dir,
    )


def preflight_timeline(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Validate every Artifact, tensor binding and Global-PA range without replay."""

    (
        _project_root,
        _config,
        _model,
        graph,
        execution_graph,
        specs,
        _dispatcher,
        memory_map,
        _output_dir,
    ) = _prepare_timeline(config_path, output_dir)
    return {
        "schema_version": "hetero-p23-bs2-decode-timeline-preflight/v1",
        "status": "passed",
        **graph_summary(graph),
        "dispatch_spec_count": len(specs),
        "global_pa_allocation_count": int(memory_map["allocation_count"]),
        "request_cycle_binding_count": len(memory_map["request_cycle_bindings"]),
        "runtime_task_binding_count": len(memory_map["runtime_task_bindings"]),
        "batch_member_kv_slice_count": len(memory_map["batch_member_kv_slices"]),
        "performance_claim_allowed": False,
        "all_tasks_are_fused_bs2": all(
            int(task["fused_batch_size"]) == 2
            for task in execution_graph["tasks"]  # type: ignore[index]
        ),
    }


def run_timeline(config_path: Path, output_dir: Path) -> Path:
    (
        project_root,
        config,
        _model,
        graph,
        execution_graph,
        specs,
        dispatcher,
        memory_map,
        output_dir,
    ) = _prepare_timeline(config_path, output_dir)

    progress_path = output_dir / "progress.json"
    completed: list[dict[str, object]] = []

    def dispatch(spec: OnlineDispatchSpec) -> BackendTaskResult:
        _write_json(
            progress_path,
            {
                "schema_version": "hetero-p23-bs2-timeline-progress/v1",
                "status": "running",
                "current_task_id": spec.task_id,
                "current_operator": spec.node.op,
                "completed_task_count": len(completed),
                "total_task_count": len(specs),
                "completed_tasks": completed,
            },
        )
        result = dispatcher.dispatch(
            spec.backend_key, spec.node, spec.model, spec.device_id, spec.task_id
        )
        completed.append(
            {
                "task_id": spec.task_id,
                "operator": spec.node.op,
                "duration_fs": result.duration_fs,
                "cycles": int(result.statistics.get("cycles", 0)),
            }
        )
        return result

    runtime = run_online_operator_dag(execution_graph, specs, dispatch)
    timing_by_id = {str(item["task_id"]): item for item in runtime["tasks"]}
    for task in execution_graph["tasks"]:  # type: ignore[index]
        timing = timing_by_id[str(task["task_id"])]
        task["timing"] = timing
        task["effective_duration_fs"] = int(timing["completion_time_fs"]) - int(
            timing["start_time_fs"]
        )
    residency = _materialize_residency(execution_graph, timing_by_id)
    lifecycle = _request_lifecycle(execution_graph)
    gpu_tasks = [
        task for task in execution_graph["tasks"] if task["op"] in GPU_OPERATORS
    ]  # type: ignore[index]
    runtime_tasks = [
        task for task in execution_graph["tasks"] if task["op"] in RUNTIME_OPERATORS
    ]  # type: ignore[index]
    metrics = {
        "schema_version": "hetero-p23-bs2-decode-timeline-metrics/v1",
        "run_status": "request_cycle_deployment",
        "implementation_status": "implemented_unqualified",
        "performance_claim_allowed": False,
        "batch_size": 2,
        "request_count": 2,
        "layer_count": 1,
        "decode_steps": 1,
        "gpu_task_instances": len(gpu_tasks),
        "gpu_operator_types": len({str(task["op"]) for task in gpu_tasks}),
        "runtime_task_instances": len(runtime_tasks),
        "makespan_fs": runtime["makespan_fs"],
        "trace_coverage": len(gpu_tasks) / len(execution_graph["tasks"]),  # type: ignore[arg-type]
        "qualification_boundary": (
            "Exact BS=2 operator instruction/request timing plus live KV/runtime "
            "memory is functionally composed. Cross-kernel persistent-memory-state "
            "and device performance calibration remain closed."
        ),
    }
    _write_json(output_dir / "resolved_timeline_config.json", config)
    _write_json(output_dir / "model_graph.json", graph_to_dict(graph))
    _write_json(output_dir / "execution_graph.json", execution_graph)
    _write_json(output_dir / "runtime_result.json", runtime)
    _write_json(output_dir / "global_memory_map.json", memory_map)
    _write_json(output_dir / "residency.json", residency)
    _write_json(output_dir / "request_lifecycle.json", lifecycle)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "trace_bundle.json", dispatcher.trace_bundle())
    _write_json(output_dir / "backend_provenance.json", dispatcher.provenance())
    _write_json(
        output_dir / "provenance.json",
        {
            "schema_version": "hetero-p23-bs2-decode-timeline-provenance/v1",
            "simulator_revision": _git_revision(project_root),
            "physical_capture_device": "remote RTX4090",
            "capture_sm": 89,
            "replay_target_sm": 86,
            "local_rtx3070_sass_used": False,
            "batch_member_request_ids": list(MEMBER_REQUEST_IDS),
        },
    )
    _write_json(
        progress_path,
        {
            "schema_version": "hetero-p23-bs2-timeline-progress/v1",
            "status": "complete",
            "current_task_id": None,
            "current_operator": None,
            "completed_task_count": len(completed),
            "total_task_count": len(specs),
            "completed_tasks": completed,
        },
    )
    return output_dir


def graph_summary(graph: ModelGraph) -> dict[str, object]:
    counts = Counter(node.op for node in graph.nodes)
    return {
        "task_count": len(graph.nodes),
        "gpu_task_count": sum(counts[operator] for operator in GPU_OPERATORS),
        "gpu_operator_type_count": len(GPU_OPERATORS),
        "runtime_task_count": sum(counts[operator] for operator in RUNTIME_OPERATORS),
        "operator_counts": dict(sorted(counts.items())),
    }
