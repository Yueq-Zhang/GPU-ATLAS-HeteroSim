"""Uncalibrated integer-only Roofline estimates for M2 execution previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .ir import ModelNode
from .model_graph import ModelSpec

FS_PER_SECOND = 10**15


@dataclass(frozen=True, slots=True)
class AnalyticalTaskCost:
    flops: int
    read_bytes: int
    write_bytes: int
    compute_time_fs: int
    memory_time_fs: int
    duration_fs: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("analytical denominator must be positive")
    return (numerator + denominator - 1) // denominator


def estimate_node_cost(
    node: ModelNode,
    model: ModelSpec,
    backend: Mapping[str, object],
) -> AnalyticalTaskCost:
    """Estimate one node without pretending to model caches or DRAM timing.

    Projection weights are counted once per node execution.  The estimate is
    intentionally conservative and deterministic; it is a preview input to
    the global event runtime, not a calibrated performance claim.
    """

    compute_rate = int(backend["effective_compute_flops_per_s"])
    memory_rate = int(backend["effective_memory_bandwidth_Bps"])
    if compute_rate <= 0 or memory_rate <= 0:
        raise ValueError("effective compute and memory rates must be positive")

    m = int(node.attributes.get("q_len", 1))
    kv_len = int(node.attributes.get("attention_kv_len", m))
    h = model.hidden_size
    i = model.intermediate_size
    v = model.vocab_size
    b = model.bytes_per_element
    activation = m * h * b

    flops = 0
    read_bytes = activation
    write_bytes = activation
    if node.op == "qkv_projection":
        flops = 6 * m * h * h
        read_bytes += 3 * h * h * b
    elif node.op == "output_projection":
        flops = 2 * m * h * h
        read_bytes += h * h * b
    elif node.op == "gate_up_projection":
        flops = 4 * m * h * i
        read_bytes += 2 * h * i * b
        write_bytes = 2 * m * i * b
    elif node.op == "down_projection":
        flops = 2 * m * i * h
        read_bytes = m * i * b + i * h * b
    elif node.op == "lm_head":
        flops = 2 * m * h * v
        read_bytes += h * v * b
        write_bytes = m * v * b
    elif node.op == "causal_attention":
        flops = 4 * m * kv_len * h
        kv_bytes = 2 * kv_len * model.num_kv_heads * model.head_dim * b
        read_bytes += kv_bytes
    elif node.op == "kv_append":
        kv_write = 2 * m * model.num_kv_heads * model.head_dim * b
        write_bytes += kv_write
    elif node.op in {
        "attention_norm",
        "rope",
        "residual_add",
        "mlp_norm",
        "silu_multiply",
        "final_norm",
    }:
        flops = 5 * m * h
    elif node.op == "sampling":
        flops = v
        read_bytes = v * b
        write_bytes = 8
    else:
        # State and control nodes still occupy one femtosecond so that the C++
        # runtime can preserve their ordering with a strictly positive duration.
        read_bytes = 0
        write_bytes = 0

    compute_time = _ceil_div(flops * FS_PER_SECOND, compute_rate) if flops else 0
    memory_time = _ceil_div(
        (read_bytes + write_bytes) * FS_PER_SECOND, memory_rate
    ) if read_bytes or write_bytes else 0
    return AnalyticalTaskCost(
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        compute_time_fs=compute_time,
        memory_time_fs=memory_time,
        duration_fs=max(1, compute_time, memory_time),
    )


def estimate_link_duration_fs(payload_bytes: int, link: Mapping[str, object]) -> int:
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be unsigned")
    bandwidth = int(link["wire_bandwidth_Bps"])
    latency = int(link.get("latency_fs", 0))
    header = int(link.get("header_bytes", 0))
    if bandwidth <= 0 or latency < 0 or header < 0:
        raise ValueError("invalid analytical link parameters")
    serialization = _ceil_div((payload_bytes + header) * FS_PER_SECOND, bandwidth)
    return max(1, latency + serialization)
