from frontend.hetero.analytical import (
    estimate_link_duration_fs,
    estimate_node_cost,
)
from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec


def _model() -> ModelSpec:
    return ModelSpec("tiny", 128, 256, 2, 4, 2, 32, 256)


def test_projection_cost_uses_integer_roofline_maximum() -> None:
    node = ModelNode(
        "qkv",
        NodeKind.COMPUTE,
        "qkv_projection",
        Phase.PREFILL,
        0,
        0,
        attributes={"q_len": 16},
    )
    cost = estimate_node_cost(
        node,
        _model(),
        {
            "effective_compute_flops_per_s": 10**12,
            "effective_memory_bandwidth_Bps": 10**9,
        },
    )
    assert cost.flops == 6 * 16 * 128 * 128
    assert cost.duration_fs == cost.memory_time_fs
    assert cost.duration_fs > cost.compute_time_fs


def test_link_cost_counts_header_as_wire_bytes() -> None:
    duration = estimate_link_duration_fs(
        64,
        {
            "wire_bandwidth_Bps": 10**12,
            "latency_fs": 11,
            "header_bytes": 16,
        },
    )
    assert duration == 80_011
