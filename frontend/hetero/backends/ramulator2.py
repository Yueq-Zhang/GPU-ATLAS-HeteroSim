"""Standalone Ramulator2 replay adapter.

The live library callback path uses the same canonical request fields but is
qualified separately.  This adapter materializes a LoadStoreTrace and parses
Ramulator2's YAML statistics, providing a reproducible offline baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

class Ramulator2BackendError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Ramulator2BackendConfig:
    executable: Path
    config_template: Path
    frequency_hz: int
    transaction_bytes: int
    timeout_seconds: int
    expected_commit: str

    @classmethod
    def load(cls, path: Path) -> "Ramulator2BackendConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "hetero-ramulator2-backend/v1":
            raise Ramulator2BackendError("invalid Ramulator2 backend schema")
        base = path.resolve().parent

        def resolve(value: object) -> Path:
            candidate = Path(str(value))
            return candidate if candidate.is_absolute() else (base / candidate).resolve()

        result = cls(
            resolve(payload["executable"]),
            resolve(payload["config_template"]),
            int(payload["frequency_hz"]),
            int(payload.get("transaction_bytes", 64)),
            int(payload.get("timeout_seconds", 3600)),
            str(payload["expected_commit"]),
        )
        if min(
            result.frequency_hz,
            result.transaction_bytes,
            result.timeout_seconds,
        ) <= 0 or not result.expected_commit:
            raise Ramulator2BackendError("invalid Ramulator2 backend values")
        return result


@dataclass(frozen=True, slots=True)
class Ramulator2RunResult:
    memory_system_cycles: int
    duration_fs: int
    parent_requests: int
    trace_transactions: int
    logical_bytes: int
    trace_sha256: str
    stdout_path: str
    stderr_path: str


class Ramulator2Backend:
    def __init__(self, config: Ramulator2BackendConfig) -> None:
        self.config = config

    def run(
        self,
        requests: Sequence[Mapping[str, object]],
        output_dir: Path,
    ) -> Ramulator2RunResult:
        if not requests:
            raise Ramulator2BackendError("Ramulator2 replay requires requests")
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_lines: list[str] = []
        logical_bytes = 0
        for request in requests:
            operation = str(request.get("operation", "read"))
            if operation not in {"read", "write"}:
                raise Ramulator2BackendError(f"invalid memory operation: {operation}")
            address = int(request["offset_bytes"])
            size = int(request["size_bytes"])
            if address < 0 or size <= 0:
                raise Ramulator2BackendError("invalid request address or size")
            logical_bytes += size
            cursor = 0
            while cursor < size:
                trace_lines.append(
                    f"{'ST' if operation == 'write' else 'LD'} 0x{address + cursor:x}"
                )
                cursor += self.config.transaction_bytes
        trace_text = "\n".join(trace_lines) + "\n"
        trace_path = output_dir / "loadstore.trace"
        trace_path.write_text(trace_text, encoding="utf-8")
        template = self.config.config_template.read_text(encoding="utf-8")
        if "{{TRACE_PATH}}" not in template:
            raise Ramulator2BackendError(
                "Ramulator2 config template must contain {{TRACE_PATH}}"
            )
        materialized = template.replace("{{TRACE_PATH}}", str(trace_path.resolve()))
        config_path = output_dir / "ramulator2.resolved.yaml"
        config_path.write_text(materialized, encoding="utf-8")
        completed = subprocess.run(
            [str(self.config.executable), "-f", str(config_path)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        stdout_path = output_dir / "ramulator2.stdout.log"
        stderr_path = output_dir / "ramulator2.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise Ramulator2BackendError(
                f"Ramulator2 exited with {completed.returncode}; see {stderr_path}"
            )
        match = re.search(r"memory_system_cycles\s*:\s*(\d+)", completed.stdout)
        if match is None:
            raise Ramulator2BackendError("Ramulator2 output lacks memory_system_cycles")
        cycles = int(match.group(1))
        duration_fs = (
            cycles * 10**15 + self.config.frequency_hz - 1
        ) // self.config.frequency_hz
        return Ramulator2RunResult(
            cycles,
            duration_fs,
            len(requests),
            len(trace_lines),
            logical_bytes,
            hashlib.sha256(trace_text.encode("utf-8")).hexdigest(),
            str(stdout_path),
            str(stderr_path),
        )

    def qualify(
        self,
        requests: Sequence[Mapping[str, object]],
        output_dir: Path,
    ) -> Path:
        first = self.run(requests, output_dir / "run1")
        second = self.run(requests, output_dir / "run2")
        passed = (
            first.memory_system_cycles == second.memory_system_cycles
            and first.duration_fs == second.duration_fs
            and first.parent_requests == second.parent_requests
            and first.trace_transactions == second.trace_transactions
            and first.logical_bytes == second.logical_bytes
            and first.trace_sha256 == second.trace_sha256
        )
        record = {
            "schema_version": "hetero-ramulator2-qualification/v1",
            "status": "passed" if passed else "failed",
            "expected_commit": self.config.expected_commit,
            "run1": asdict(first),
            "run2": asdict(second),
            "deterministic_equivalence": passed,
        }
        path = output_dir / "qualification_record.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        if not passed:
            raise Ramulator2BackendError("Ramulator2 qualification mismatch")
        return path
