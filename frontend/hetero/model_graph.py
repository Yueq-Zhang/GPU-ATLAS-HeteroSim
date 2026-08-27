"""Canonical decoder-only LLM request graph construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .ir import ModelGraph, ModelNode, NodeKind, Phase, StorageClass, Value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    dtype: str = "fp16"
    bytes_per_element: int = 2
    architecture: str = "llama"
    mlp_type: str = "swiglu"
    position_encoding: str = "rope"
    tied_embeddings: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            self.vocab_size,
            self.bytes_per_element,
        )
        if not self.name or any(value <= 0 for value in integer_fields):
            raise ValueError("model fields must be non-empty positive values")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_attention_heads * head_dim must equal hidden_size")
        if self.mlp_type not in {"swiglu", "dense_gelu"}:
            raise ValueError("mlp_type must be swiglu or dense_gelu")
        if self.position_encoding not in {"rope", "learned_absolute"}:
            raise ValueError("unsupported position_encoding")


@dataclass(frozen=True, slots=True)
class RequestSpec:
    request_id: str
    prompt_length: int
    output_length: int
    arrival_time_fs: int = 0
    priority: int = 0
    execution_scope: str = "full_request"
    initial_kv_length: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.prompt_length <= 0 or self.output_length <= 0:
            raise ValueError("prompt_length and output_length must be positive")
        if self.arrival_time_fs < 0:
            raise ValueError("arrival_time_fs must be unsigned")
        if self.execution_scope not in {"full_request", "decode_step"}:
            raise ValueError("execution_scope must be full_request or decode_step")
        if self.initial_kv_length < 0:
            raise ValueError("initial_kv_length must be unsigned")
        if self.execution_scope == "decode_step" and self.initial_kv_length <= 0:
            raise ValueError("decode_step requires positive initial_kv_length")


@dataclass(frozen=True, slots=True)
class GraphCounters:
    prefill_forwards: int
    decode_forwards: int
    lm_head: int
    sampling: int
    kv_append_pairs: int
    kv_range_writes: int
    final_committed_kv_len: int


def _layer_ops(model: ModelSpec) -> Iterable[tuple[str, str]]:
    attention = (
        ("norm.attention", "attention_norm"),
        ("attention.projection", "qkv_projection"),
        (
            "attention.rope" if model.position_encoding == "rope" else "attention.position",
            "rope" if model.position_encoding == "rope" else "position_add",
        ),
        ("attention.kv_append", "kv_append"),
        ("attention.core", "causal_attention"),
        ("attention.output", "output_projection"),
        ("attention.residual", "residual_add"),
        ("mlp.norm", "mlp_norm"),
    )
    mlp = (
        (
            ("mlp.gate_up", "gate_up_projection"),
            ("mlp.activation", "silu_multiply"),
        )
        if model.mlp_type == "swiglu"
        else (
            ("mlp.fc1", "fc1_projection"),
            ("mlp.activation", "gelu"),
        )
    )
    return (*attention, *mlp, ("mlp.down", "down_projection"), ("mlp.residual", "residual_add"))


def build_request_graph(model: ModelSpec, request: RequestSpec) -> ModelGraph:
    """Build one complete full-request logical graph.

    Prefill produces output token zero.  Decode forward ``j`` consumes output
    token ``j-1`` and therefore appears exactly ``output_length - 1`` times.
    """

    values: list[Value] = []
    nodes: list[ModelNode] = []
    value_ids: set[str] = set()
    previous_node: str | None = None

    def add_value(value_id: str, storage: StorageClass, shape: tuple[int | str, ...]) -> None:
        if value_id in value_ids:
            return
        value_ids.add(value_id)
        values.append(
            Value(
                value_id=value_id,
                shape_expr=shape,
                dtype=model.dtype,
                layout="row_major",
                storage_class=storage,
                mutable=storage is StorageClass.KV_CACHE,
                lifetime="request" if storage is not StorageClass.PARAMETER else "static",
            )
        )

    def add_node(
        node_id: str,
        kind: NodeKind,
        op: str,
        phase: Phase,
        step_id: int,
        layer_id: int | None = None,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        attributes: dict[str, object] | None = None,
    ) -> None:
        nonlocal previous_node
        dependencies = () if previous_node is None else (previous_node,)
        nodes.append(
            ModelNode(
                node_id=node_id,
                kind=kind,
                op=op,
                phase=phase,
                layer_id=layer_id,
                step_id=step_id,
                dependencies=dependencies,
                read_values=reads,
                write_values=writes,
                attributes=attributes or {},
            )
        )
        previous_node = node_id

    prefix = request.request_id
    prompt_value = f"{prefix}.prompt"
    if request.execution_scope == "full_request":
        add_value(
            prompt_value,
            StorageClass.ACTIVATION,
            (request.prompt_length, model.hidden_size),
        )
    else:
        prompt_value = f"{prefix}.decode_input"
        add_value(prompt_value, StorageClass.ACTIVATION, (1, model.hidden_size))
    for layer_id in range(model.num_layers):
        for kind in ("k", "v"):
            add_value(
                f"{prefix}.kv.l{layer_id}.{kind}",
                StorageClass.KV_CACHE,
                ("kv_tokens", model.num_kv_heads, model.head_dim),
            )

    add_node(f"{prefix}.request_start", NodeKind.CONTROL, "request_start", Phase.CONTROL, 0)
    add_node(f"{prefix}.kv_allocate", NodeKind.STATE, "kv_allocate", Phase.CONTROL, 0)

    forward_steps = (
        [
            (Phase.PREFILL, 0, request.prompt_length, 0),
            *[
                (Phase.DECODE, step, 1, request.prompt_length + step - 1)
                for step in range(1, request.output_length)
            ],
        ]
        if request.execution_scope == "full_request"
        else [(Phase.DECODE, 0, 1, request.initial_kv_length)]
    )
    for phase, step_id, q_len, past_len in forward_steps:
        phase_key = f"{phase.value}.s{step_id}"
        hidden = (
            prompt_value
            if phase is Phase.PREFILL or request.execution_scope == "decode_step"
            else f"{prefix}.token.{step_id - 1}"
        )
        if phase is Phase.DECODE and request.execution_scope == "full_request":
            add_value(hidden, StorageClass.ACTIVATION, (1, model.hidden_size))
        for layer_id in range(model.num_layers):
            for group, op in _layer_ops(model):
                output = f"{prefix}.{phase_key}.l{layer_id}.{group}.out"
                add_value(output, StorageClass.ACTIVATION, (q_len, model.hidden_size))
                reads = [hidden]
                writes = [output]
                kind = NodeKind.COMPUTE
                if op == "kv_append":
                    kind = NodeKind.STATE
                    writes.extend(
                        [
                            f"{prefix}.kv.l{layer_id}.k",
                            f"{prefix}.kv.l{layer_id}.v",
                        ]
                    )
                if op == "causal_attention":
                    reads.extend(
                        [
                            f"{prefix}.kv.l{layer_id}.k",
                            f"{prefix}.kv.l{layer_id}.v",
                        ]
                    )
                add_node(
                    f"{prefix}.{phase_key}.l{layer_id}.{group}",
                    kind,
                    op,
                    phase,
                    step_id,
                    layer_id,
                    tuple(reads),
                    tuple(writes),
                    {
                        "operator_group": group.split(".", 1)[0],
                        "q_len": q_len,
                        "past_kv_len": past_len,
                        "attention_kv_len": past_len + q_len,
                    },
                )
                hidden = output

        norm = f"{prefix}.{phase_key}.final_norm.out"
        logits = f"{prefix}.{phase_key}.logits"
        token = f"{prefix}.token.{step_id}"
        add_value(norm, StorageClass.ACTIVATION, (q_len, model.hidden_size))
        add_value(logits, StorageClass.ACTIVATION, (1, model.vocab_size))
        add_value(token, StorageClass.METADATA, (1,))
        add_node(f"{prefix}.{phase_key}.final_norm", NodeKind.COMPUTE, "final_norm", phase, step_id, reads=(hidden,), writes=(norm,))
        add_node(f"{prefix}.{phase_key}.lm_head", NodeKind.COMPUTE, "lm_head", phase, step_id, reads=(norm,), writes=(logits,))
        add_node(f"{prefix}.{phase_key}.sampling", NodeKind.CONTROL, "sampling", phase, step_id, reads=(logits,), writes=(token,))

    add_node(f"{prefix}.request_finish", NodeKind.CONTROL, "request_finish", Phase.CONTROL, request.output_length)
    add_node(f"{prefix}.kv_release", NodeKind.STATE, "kv_release", Phase.CONTROL, request.output_length)
    graph = ModelGraph("hetero-model-graph/v1", tuple(values), tuple(nodes))
    graph.validate()
    return graph


def graph_counters(model: ModelSpec, request: RequestSpec) -> GraphCounters:
    if request.execution_scope == "decode_step":
        return GraphCounters(
            prefill_forwards=0,
            decode_forwards=1,
            lm_head=1,
            sampling=1,
            kv_append_pairs=model.num_layers,
            kv_range_writes=model.num_layers * 2,
            final_committed_kv_len=request.initial_kv_length + 1,
        )
    decode = request.output_length - 1
    written_tokens = request.prompt_length + decode
    return GraphCounters(
        prefill_forwards=1,
        decode_forwards=decode,
        lm_head=request.output_length,
        sampling=request.output_length,
        kv_append_pairs=written_tokens * model.num_layers,
        kv_range_writes=written_tokens * model.num_layers * 2,
        final_committed_kv_len=written_tokens,
    )


def model_spec_from_config(config: dict[str, object]) -> ModelSpec:
    return ModelSpec(**config)  # type: ignore[arg-type]


def request_specs_from_config(requests: list[dict[str, object]]) -> list[RequestSpec]:
    return [RequestSpec(**request) for request in requests]  # type: ignore[arg-type]


def graph_to_dict(graph: ModelGraph) -> dict[str, object]:
    return asdict(graph)
