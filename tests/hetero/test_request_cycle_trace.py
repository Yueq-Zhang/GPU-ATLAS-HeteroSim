import gzip
import json
from pathlib import Path

import pytest

from frontend.hetero.request_cycle_trace import (
    RequestCycleTraceError,
    RequestCycleTraceRecorder,
)


def _record(parent_id: int) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "issue_cycle": parent_id,
        "issue_time_fs": parent_id * 10,
        "task_id": "task.gpu",
        "device_id": "gpu0",
        "value_id": "x",
        "version": 1,
        "operation": "read",
        "global_address": 4096 + parent_id * 64,
        "size_bytes": 64,
        "represented_bytes": 64,
        "sample_index": parent_id - 1,
        "sample_count": 2,
        "logical_value_bytes": 128,
        "traffic_mode": "full",
    }


def _write(path: Path) -> dict[str, object]:
    recorder = RequestCycleTraceRecorder(path)
    for parent_id in (1, 2):
        recorder.issue(_record(parent_id))
        recorder.complete(
            parent_id,
            {
                "parent_id": parent_id,
                "completion_cycle": parent_id + 2,
                "completion_time_fs": (parent_id + 2) * 10,
            },
        )
    result = recorder.finalize()
    assert isinstance(result, dict)
    return result


def test_stream_trace_is_deterministic_compressed_and_conserved(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.jsonl.gz")
    second = _write(tmp_path / "second.jsonl.gz")
    assert first["compressed_sha256"] == second["compressed_sha256"]
    assert first["uncompressed_sha256"] == second["uncompressed_sha256"]
    assert first["statistics"] == second["statistics"]
    assert first["statistics"]["issued_parents"] == 2
    assert first["statistics"]["outstanding_parents"] == 0
    with gzip.open(tmp_path / "first.jsonl.gz", "rt", encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    assert [item["event"] for item in events] == [
        "header",
        "request_issue",
        "request_completion",
        "request_issue",
        "request_completion",
        "footer",
    ]


def test_stream_trace_removes_partial_file_on_abort(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl.gz"
    recorder = RequestCycleTraceRecorder(path)
    recorder.issue(_record(1))
    with pytest.raises(RequestCycleTraceError, match="outstanding"):
        recorder.finalize()
    recorder.abort()
    assert not path.exists()
    assert not path.with_name(path.name + ".partial").exists()


def test_trace_rejects_invalid_issue_and_mismatched_completion() -> None:
    recorder = RequestCycleTraceRecorder()
    invalid = _record(1)
    invalid["operation"] = "atomic"
    with pytest.raises(RequestCycleTraceError, match="unsupported operation"):
        recorder.issue(invalid)
    recorder.issue(_record(1))
    with pytest.raises(RequestCycleTraceError, match="parent mismatch"):
        recorder.complete(1, {"parent_id": 2, "completion_cycle": 3})
    recorder.complete(1, {"completion_cycle": 3, "completion_time_fs": 30})
    result = recorder.finalize()
    assert isinstance(result, list)
    assert result[0]["parent_id"] == 1
