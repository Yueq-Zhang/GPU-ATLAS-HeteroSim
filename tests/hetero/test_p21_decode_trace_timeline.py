from collections import Counter
from pathlib import Path

from scripts.build_p21_decode_trace_timeline import (
    OPERATORS,
    build_capability,
    build_experiment,
)
from scripts.qualify_p21_decode_trace_timeline import _expected_counts


def test_p21_one_layer_builds_exact_step_specific_trace_bindings() -> None:
    experiment = build_experiment(1, Path("source"), Path("coupled"))
    gpu = experiment["backends"]["gpu"]
    bindings = gpu["trace_bindings"]
    assert len(bindings) == 56
    assert gpu["fallback_kind"] == "none"
    assert gpu["require_request_cycle_ready"] is True
    assert {item["selector"]["step_id"] for item in bindings} == {0, 1, 2, 3}
    assert {item["selector"]["op"] for item in bindings} == set(OPERATORS)
    assert all(item["contract_overrides"] == {"layer_id": 0} for item in bindings)
    for item in bindings:
        kv_length = 17 + item["selector"]["step_id"]
        assert f"kv{kv_length}_" in item["trace_manifest"]
        assert f"kv{kv_length}_" in item["operator_artifact"]


def test_p21_capability_and_expected_graph_scale_to_22_layers() -> None:
    capability = build_capability(22, Path("coupled"))
    operators = {item["operator_type"]: item for item in capability["operator_types"]}
    expected = _expected_counts(22)
    assert sum(expected.values()) == 1076
    assert expected["residual_add"] == 176
    assert expected["kv_append"] == 88
    assert operators["qkv_projection"]["instances_in_reference_graph"] == 88
    assert operators["qkv_projection"]["request_cycle_ready"] is True
    assert operators["kv_append"]["request_cycle_ready"] is False
    assert capability["claim_boundary"]["performance_claim_allowed"] is False


def test_p21_expected_operator_counts_cover_all_19_types() -> None:
    counts = _expected_counts(1)
    assert len(counts) == 19
    assert counts == Counter(
        {
            "request_start": 1,
            "kv_allocate": 1,
            "request_finish": 1,
            "kv_release": 1,
            "token_embedding": 4,
            "attention_norm": 4,
            "qkv_projection": 4,
            "rope": 4,
            "kv_append": 4,
            "causal_attention": 4,
            "output_projection": 4,
            "residual_add": 8,
            "mlp_norm": 4,
            "gate_up_projection": 4,
            "silu_multiply": 4,
            "down_projection": 4,
            "final_norm": 4,
            "lm_head": 4,
            "sampling": 4,
        }
    )
