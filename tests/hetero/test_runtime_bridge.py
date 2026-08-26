from frontend.hetero.runtime_bridge import (
    allocate_paged_kv,
    ideal_link_completion_fs,
    simulate_token_barrier,
)


def test_cpp_scheduler_matches_frozen_epoch_table() -> None:
    result = simulate_token_barrier(
        [
            {"request_id": "R0", "arrival_time_fs": 0, "prompt_length": 4, "output_length": 2},
            {"request_id": "R1", "arrival_time_fs": 0, "prompt_length": 2, "output_length": 1},
            {"request_id": "R2", "arrival_time_fs": 1500, "prompt_length": 3, "output_length": 2},
        ],
        {
            "max_num_sequences": 2,
            "max_batched_tokens": 4,
            "prefill_chunk_tokens": 2,
            "max_prefill_wait_epochs": 8,
            "epoch_duration_fs": 1000,
        },
    )
    selections = [
        [(item["request_id"], item["phase"], item["token_begin"], item["token_count"]) for item in epoch["selections"]]
        for epoch in result["epochs"]
    ]
    assert selections == [
        [("R0", "prefill", 0, 2), ("R1", "prefill", 0, 2)],
        [("R0", "prefill", 2, 2)],
        [("R0", "decode", 4, 1), ("R2", "prefill", 0, 2)],
        [("R2", "prefill", 2, 1)],
        [("R2", "decode", 3, 1)],
    ]


def test_cpp_paged_kv_matches_tiny_golden_values() -> None:
    result = allocate_paged_kv(
        [{"request_id": "R0", "prompt_length": 16, "output_length": 3}],
        {
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 32,
            "bytes_per_element": 2,
        },
        {"page_tokens": 16, "kv_capacity_bytes": 1 << 30},
        "shared0.dram3d",
    )
    allocation = result["allocations"][0]
    assert allocation["final_committed_tokens"] == 18
    assert allocation["allocated_blocks"] == 8
    assert allocation["bytes_per_block"] == 2048
    assert allocation["logical_bytes"] == 9216
    assert allocation["allocated_bytes"] == 16384


def test_cpp_ideal_link_formula_uses_wire_bytes() -> None:
    assert ideal_link_completion_fs(7, 11, 64, 16, 10**12) == 80018
