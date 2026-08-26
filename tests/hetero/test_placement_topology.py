import pytest

from frontend.hetero.model_graph import ModelSpec, RequestSpec, build_request_graph
from frontend.hetero.placement import place_nodes
from frontend.hetero.topology import (
    LoweringKind,
    lower_cross_device_dependency,
    primary_3ddram,
)


@pytest.mark.parametrize(
    ("profile", "memory_space"),
    [
        ("model1_atlas_native", "atlas0.dram3d"),
        ("model2_host_memory_pcie", "host0.dram3d"),
        ("model3_gpu_native_3ddram", "shared0.dram3d"),
        ("model4_cxl_memory_tier", "cxl0.dram3d"),
    ],
)
def test_primary_3ddram_role_is_profile_specific(profile: str, memory_space: str) -> None:
    assert primary_3ddram(profile) == memory_space


def test_model3_cross_device_edge_is_sync_not_dma() -> None:
    decision = lower_cross_device_dependency(
        "model3_gpu_native_3ddram", "gpu0", "atlas0.compute"
    )
    assert decision.kind is LoweringKind.SYNCHRONIZATION
    assert decision.source_space == decision.destination_space == "shared0.dram3d"
    assert decision.actions == (
        "writeback",
        "release_fence",
        "invalidate",
        "acquire_fence",
    )


def test_model2_cross_device_edge_uses_pcie() -> None:
    decision = lower_cross_device_dependency(
        "model2_host_memory_pcie", "gpu0", "atlas0.compute"
    )
    assert decision.kind is LoweringKind.TRANSFER
    assert decision.route_id == "pcie0.dma"


def test_rule_placement_is_yaml_order_first_match() -> None:
    model = ModelSpec("tiny", 128, 256, 1, 4, 2, 32, 256)
    graph = build_request_graph(model, RequestSpec("R0", 16, 2))
    decisions = place_nodes(
        graph.nodes,
        {
            "mode": "rule_based",
            "default_target": "gpu0",
            "rules": [
                {"match": {"phase": "decode", "operator_group": "attention"}, "target": "atlas0.compute"},
                {"match": {"phase": "decode"}, "target": "gpu0"},
            ],
        },
    )
    by_id = {decision.node_id: decision for decision in decisions}
    attention = by_id["R0.decode.s1.l0.attention.core"]
    mlp = by_id["R0.decode.s1.l0.mlp.down"]
    assert attention.target_device == "atlas0.compute"
    assert attention.matched_rule == 0
    assert mlp.target_device == "gpu0"
    assert mlp.matched_rule == 1
