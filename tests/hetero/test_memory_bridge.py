import json
from pathlib import Path

from frontend.hetero.backends.memory_bridge import run_jsonl_bridge


def test_jsonl_bridge_normalizes_before_shared_memory(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "hetero-trace-manifest/v1",
        "trace_id": "bridge-test",
        "trace_semantics": "functional",
        "replay_safe": False,
        "qualification_record": None,
        "kernels_list": "kernelslist.g",
        "capture": {"gpu_sm": 89},
        "compilation": {"dtype": "fp16"},
        "address_ranges": [
            {
                "capture_allocation_id": "capture.A",
                "trace_base": "0x1000",
                "size_bytes": 256,
                "tensor_id": "A",
                "tensor_offset_bytes": 0,
                "capture_epoch": 1,
                "backing_allocation_id": "capture.A",
                "view_offset_bytes": 0,
                "alignment_bytes": 256,
                "shape": [128],
                "layout": "linear",
            }
        ],
    }
    bindings = {
        "schema_version": "hetero-simulation-buffer-bindings/v1",
        "bindings": [
            {
                "tensor_id": "A",
                "tensor_offset_bytes": 0,
                "size_bytes": 256,
                "memory_space_id": "shared0.dram3d",
                "physical_offset_bytes": 8192,
            }
        ],
    }
    memory = {
        "memory_space_id": "shared0.dram3d",
        "initiator_order": ["gpu0", "atlas0.compute"],
        "channel_count": 2,
        "banks_per_channel": 4,
        "transaction_bytes": 64,
        "queue_depth_per_initiator": 8,
        "fixed_latency_fs": 100,
        "channel_injection_interval_fs": 10,
        "bank_busy_time_fs": 20,
    }
    paths = {
        "manifest": tmp_path / "manifest.json",
        "bindings": tmp_path / "bindings.json",
        "memory": tmp_path / "memory.json",
        "requests": tmp_path / "requests.jsonl",
        "responses": tmp_path / "responses.jsonl",
    }
    paths["manifest"].write_text(json.dumps(manifest))
    paths["bindings"].write_text(json.dumps(bindings))
    paths["memory"].write_text(json.dumps(memory))
    paths["requests"].write_text(
        json.dumps(
            {
                "type": "memory_request",
                "request_id": 1,
                "parent_task_id": 7,
                "initiator_id": "gpu0",
                "trace_address": "0x1040",
                "size_bytes": 64,
                "operation": "read",
                "issue_time_fs": 0,
            }
        )
        + "\n"
    )
    result = run_jsonl_bridge(
        paths["manifest"],
        paths["bindings"],
        paths["memory"],
        paths["requests"],
        paths["responses"],
    )
    assert result["submitted_bytes"] == 64
    assert result["child_records"][0]["offset_bytes"] == 8256
    lines = [json.loads(line) for line in paths["responses"].read_text().splitlines()]
    assert lines[0]["type"] == "bridge_header"
    assert lines[1]["type"] == "memory_response"
