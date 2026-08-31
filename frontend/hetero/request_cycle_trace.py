"""Bounded-memory request-cycle trace recording.

Production runs append canonical JSON Lines to a deterministic gzip stream.  Unit
tests may omit ``path`` and retain the legacy merged in-memory representation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping


class RequestCycleTraceError(RuntimeError):
    """Raised when issue/completion conservation is violated."""


def _canonical_line(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RequestCycleTraceRecorder:
    """Record request lifecycle events without retaining the complete trace."""

    def __init__(self, path: Path | None = None, *, compresslevel: int = 6) -> None:
        self.path = path
        self.partial_path = (
            path.with_name(path.name + ".partial") if path is not None else None
        )
        self._raw = None
        self._gzip = None
        self._logical_digest = hashlib.sha256()
        self._logical_bytes = 0
        self._event_count = 0
        self._issued: dict[int, dict[str, object] | None] = {}
        self._records: list[dict[str, object]] = []
        self._completed = 0
        self._operation_counts = {"read": 0, "write": 0}
        self._traffic_counts = {"full": 0, "sampled": 0, "coherence_probe": 0}
        self._represented_bytes = {"read": 0, "write": 0}
        self._payload_bytes = 0
        self._closed = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.partial_path is None:
                raise AssertionError("partial path missing")
            self.partial_path.unlink(missing_ok=True)
            self._raw = self.partial_path.open("wb")
            self._gzip = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compresslevel,
                fileobj=self._raw,
                mtime=0,
            )
            self._write_event(
                {
                    "event": "header",
                    "schema_version": "hetero-request-cycle-event-stream/v1",
                    "ordering": "issue_and_completion_observation_order",
                }
            )

    @property
    def issued_count(self) -> int:
        return sum(self._operation_counts.values())

    def _write_event(self, payload: Mapping[str, object]) -> None:
        line = _canonical_line(payload)
        self._logical_digest.update(line)
        self._logical_bytes += len(line)
        self._event_count += 1
        if self._gzip is not None:
            self._gzip.write(line)

    def issue(self, record: Mapping[str, object]) -> None:
        if self._closed:
            raise RequestCycleTraceError("cannot issue after trace finalization")
        parent_id = int(record["parent_id"])
        if parent_id in self._issued:
            raise RequestCycleTraceError(f"duplicate request issue {parent_id}")
        item = dict(record)
        operation = str(item["operation"])
        traffic_mode = str(item.get("traffic_mode", ""))
        if operation not in self._operation_counts:
            raise RequestCycleTraceError(f"unsupported operation {operation}")
        self._issued[parent_id] = item if self.path is None else None
        if self.path is None:
            self._records.append(item)
        else:
            self._write_event({"event": "request_issue", **item})
        self._operation_counts[operation] += 1
        if traffic_mode in self._traffic_counts:
            self._traffic_counts[traffic_mode] += 1
        self._represented_bytes[operation] += int(item.get("represented_bytes", 0))
        self._payload_bytes += int(item["size_bytes"])

    def complete(self, parent_id: int, completion: Mapping[str, object]) -> None:
        if self._closed:
            raise RequestCycleTraceError("cannot complete after trace finalization")
        if parent_id not in self._issued:
            raise RequestCycleTraceError(f"completion without issue {parent_id}")
        completion_item = dict(completion)
        if (
            "parent_id" in completion_item
            and int(completion_item["parent_id"]) != parent_id
        ):
            raise RequestCycleTraceError(
                f"completion parent mismatch: expected {parent_id}, "
                f"observed {completion_item['parent_id']}"
            )
        completion_item["parent_id"] = parent_id
        record = self._issued[parent_id]
        if record is None:
            self._write_event({"event": "request_completion", **completion_item})
        else:
            record.update(completion_item)
        del self._issued[parent_id]
        self._completed += 1

    def statistics(self) -> dict[str, int]:
        issued = sum(self._operation_counts.values())
        return {
            "issued_parents": issued,
            "completed_parents": self._completed,
            "outstanding_parents": len(self._issued),
            "reads": self._operation_counts["read"],
            "writes": self._operation_counts["write"],
            "full_traffic_parents": self._traffic_counts["full"],
            "sampled_traffic_parents": self._traffic_counts["sampled"],
            "coherence_probe_parents": self._traffic_counts["coherence_probe"],
            "represented_read_bytes": self._represented_bytes["read"],
            "represented_write_bytes": self._represented_bytes["write"],
            "simulated_parent_payload_bytes": self._payload_bytes,
        }

    def finalize(self) -> list[dict[str, object]] | dict[str, object]:
        if self._closed:
            raise RequestCycleTraceError("trace was already finalized")
        stats = self.statistics()
        if stats["outstanding_parents"] != 0:
            raise RequestCycleTraceError(
                f"trace has {stats['outstanding_parents']} outstanding parents"
            )
        if stats["issued_parents"] != stats["completed_parents"]:
            raise RequestCycleTraceError("request issue/completion count mismatch")
        if self.path is None:
            self._closed = True
            return self._records
        self._write_event({"event": "footer", "statistics": stats})
        assert self._gzip is not None and self._raw is not None
        self._gzip.close()
        self._raw.close()
        assert self.partial_path is not None and self.path is not None
        self.partial_path.replace(self.path)
        self._closed = True
        return {
            "schema_version": "hetero-request-cycle-stream-reference/v1",
            "encoding": "canonical_jsonl_gzip",
            "path": self.path.name,
            "compressed_bytes": self.path.stat().st_size,
            "compressed_sha256": _sha256_file(self.path),
            "uncompressed_bytes": self._logical_bytes,
            "uncompressed_sha256": self._logical_digest.hexdigest(),
            "event_count": self._event_count,
            "statistics": stats,
        }

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._gzip is not None:
            self._gzip.close()
        if self._raw is not None and not self._raw.closed:
            self._raw.close()
        if self.partial_path is not None:
            self.partial_path.unlink(missing_ok=True)
