from frontend.hetero.model_graph import (
    ModelSpec,
    RequestSpec,
    build_request_graph,
    graph_counters,
)


def test_decode_step_scope_builds_one_decode_without_prefill() -> None:
    model = ModelSpec(
        name="tiny",
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=256,
    )
    request = RequestSpec(
        "decode-only",
        prompt_length=1024,
        output_length=1,
        execution_scope="decode_step",
        initial_kv_length=1024,
    )
    graph = build_request_graph(model, request)
    counters = graph_counters(model, request)
    phases = {node.phase.value for node in graph.nodes}
    assert "prefill" not in phases
    assert "decode" in phases
    assert counters.prefill_forwards == 0
    assert counters.decode_forwards == 1
    assert counters.final_committed_kv_len == 1025
    attention = next(node for node in graph.nodes if node.op == "causal_attention")
    assert attention.attributes["attention_kv_len"] == 1025


TINY_MODEL = ModelSpec(
    name="tiny_llama_2layer",
    hidden_size=128,
    intermediate_size=256,
    num_layers=2,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=32,
    vocab_size=256,
)


def test_tiny_full_request_has_frozen_token_semantics() -> None:
    request = RequestSpec("R0", prompt_length=16, output_length=3)
    graph = build_request_graph(TINY_MODEL, request)
    counters = graph_counters(TINY_MODEL, request)

    assert counters.decode_forwards == 2
    assert counters.lm_head == 3
    assert counters.sampling == 3
    assert counters.final_committed_kv_len == 18
    assert counters.kv_append_pairs == 36
    assert counters.kv_range_writes == 72
    assert sum(node.op == "kv_append" for node in graph.nodes) == 6
    assert sum(node.op == "lm_head" for node in graph.nodes) == 3
    assert graph.nodes[-2].op == "request_finish"
    assert graph.nodes[-1].op == "kv_release"


def test_output_length_one_has_no_decode_forward() -> None:
    request = RequestSpec("R1", prompt_length=8, output_length=1)
    graph = build_request_graph(TINY_MODEL, request)
    assert graph_counters(TINY_MODEL, request).decode_forwards == 0
    assert not any(node.phase.value == "decode" for node in graph.nodes)
    assert sum(node.op == "sampling" for node in graph.nodes) == 1
