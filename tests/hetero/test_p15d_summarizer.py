from __future__ import annotations

import json
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys


FULL_OPERATORS = [
    "attention_norm",
    "qkv_projection",
    "rope",
    "kv_append",
    "causal_attention",
    "output_projection",
    "mlp_norm",
    "gate_up_projection",
    "silu_multiply",
    "down_projection",
    "final_norm",
    "lm_head",
    "sampling",
]


def _write_run(path: Path, *, streamed: bool = False) -> None:
    path.mkdir()
    memory = {
        "accepted_parent_ids": 5,
        "observed_completion_ids": 5,
        "completed": 5,
        "durable_completed": 5,
        "full_traffic_parents": 4,
        "sampled_traffic_parents": 1,
        "children_sent": 6,
        "children_completed": 6,
        "reads": 4,
        "writes": 1,
        "logical_bytes": 257,
        "internal_bytes": 320,
        "gpu_cycles": 30,
        "clock": 10,
        "instances": 1,
        "one_live_timing_owner": True,
        "outstanding": 0,
        "initiators": {
            "gpu0": {"parents": 5, "completed": 5, "children": 6},
            "atlas0.compute": {"parents": 0, "completed": 0, "children": 0},
        },
    }
    coverage = {
        "expected_tasks": 20,
        "covered_tasks": 20,
        "full_traffic_tasks": 13,
        "sampled_traffic_tasks": 7,
        "full_traffic_by_operator": {name: 1 for name in FULL_OPERATORS},
    }
    metrics = {"makespan_fs": 25_000, "performance_claim_allowed": False}
    trace_reference: object = [{"request_id": 1}]
    if streamed:
        stream_path = path / "request_cycle_trace.jsonl.gz"
        with stream_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(b'{"event":"header"}\n')
        trace_reference = {
            "encoding": "canonical_jsonl_gzip",
            "path": stream_path.name,
            "compressed_bytes": stream_path.stat().st_size,
            "compressed_sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest(),
        }
    payloads = {
        "request_cycle_trace.json": {"memory_trace": trace_reference},
        "memory_statistics.json": memory,
        "prefill_artifact_coverage.json": coverage,
        "metrics.json": metrics,
        "execution_graph.json": {"tasks": []},
        "global_memory_map.json": {"ranges": []},
        "trace_manifest.json": {"trace_semantics": "test"},
    }
    for name, payload in payloads.items():
        (path / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (path / "event_log.jsonl").write_text('{"event":"done"}\n', encoding="utf-8")


def test_p15d_thirteen_full_traffic_summary(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    _write_run(run1)
    _write_run(run2)
    output = tmp_path / "qualification_record.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "summarize_p15d_thirteen_full_traffic.py"),
            "--run1",
            str(run1),
            "--run2",
            str(run2),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["memory"]["parents"] == 5
    assert record["invariants"]["one_live_ramulator2"] is True
    assert record["claim_boundary"]["performance_claim_allowed"] is False


def test_p15d_summary_validates_stream_reference(tmp_path: Path) -> None:
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    _write_run(run1, streamed=True)
    _write_run(run2, streamed=True)
    output = tmp_path / "qualification_record.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "summarize_p15d_thirteen_full_traffic.py"),
            "--run1",
            str(run1),
            "--run2",
            str(run2),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert "request_cycle_trace.jsonl.gz" in record["determinism"]["exact_files"]
