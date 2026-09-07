from frontend.hetero.runtime_bridge import (
    allocate_paged_kv,
    ideal_link_completion_fs,
    run_task_dag,
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
    assert result["epochs"][0]["admitted_request_ids"] == ["R0", "R1"]
    assert result["epochs"][0]["retired_request_ids"] == ["R1"]


def test_cpp_scheduler_re_admits_after_kv_capacity_is_released() -> None:
    result = simulate_token_barrier(
        [
            {
                "request_id": "K0",
                "arrival_time_fs": 0,
                "prompt_length": 1,
                "output_length": 1,
                "kv_reservation_bytes": 64,
            },
            {
                "request_id": "K1",
                "arrival_time_fs": 0,
                "prompt_length": 1,
                "output_length": 1,
                "kv_reservation_bytes": 64,
            },
        ],
        {
            "max_num_sequences": 2,
            "max_batched_tokens": 2,
            "prefill_chunk_tokens": 1,
            "max_prefill_wait_epochs": 8,
            "epoch_duration_fs": 1000,
            "kv_capacity_bytes": 64,
        },
    )
    assert [epoch["admitted_request_ids"] for epoch in result["epochs"]] == [
        ["K0"],
        ["K1"],
    ]


def test_cpp_scheduler_applies_eos_max_length_and_barrier_cancellation() -> None:
    result = simulate_token_barrier(
        [
            {
                "request_id": "E",
                "prompt_length": 1,
                "output_length": 8,
                "execution_scope": "decode_loop",
                "initial_kv_length": 16,
                "kv_reservation_bytes": 64,
                "eos_after_generated_tokens": 2,
            },
            {
                "request_id": "M",
                "prompt_length": 1,
                "output_length": 8,
                "execution_scope": "decode_loop",
                "initial_kv_length": 16,
                "kv_reservation_bytes": 64,
                "max_output_tokens": 3,
            },
            {
                "request_id": "C",
                "prompt_length": 1,
                "output_length": 8,
                "execution_scope": "decode_loop",
                "initial_kv_length": 16,
                "kv_reservation_bytes": 64,
                "cancel_time_fs": 1000,
            },
        ],
        {
            "max_num_sequences": 3,
            "max_batched_tokens": 3,
            "prefill_chunk_tokens": 1,
            "epoch_duration_fs": 1000,
            "kv_capacity_bytes": 192,
        },
    )
    by_id = {item["request_id"]: item for item in result["requests"]}
    assert (by_id["E"]["generated_length"], by_id["E"]["termination_reason"]) == (
        2,
        "eos",
    )
    assert (by_id["M"]["generated_length"], by_id["M"]["termination_reason"]) == (
        3,
        "max_length",
    )
    assert (by_id["C"]["generated_length"], by_id["C"]["termination_reason"]) == (
        1,
        "cancelled",
    )
    assert result["epochs"][1]["cancelled_request_ids"] == ["C"]


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


def test_global_task_dag_runtime_serializes_resources() -> None:
    result = run_task_dag(
        [
            {
                "task_id": "gpu.a",
                "resource_id": "gpu0.compute",
                "dependencies": [],
                "duration_fs": 10,
            },
            {
                "task_id": "gpu.b",
                "resource_id": "gpu0.compute",
                "dependencies": [],
                "duration_fs": 5,
            },
            {
                "task_id": "atlas.c",
                "resource_id": "atlas0.compute",
                "dependencies": ["gpu.a"],
                "duration_fs": 7,
            },
        ]
    )
    assert result["makespan_fs"] == 17
    assert result["tasks"][1]["start_time_fs"] == 10
    assert result["tasks"][2]["start_time_fs"] == 10
