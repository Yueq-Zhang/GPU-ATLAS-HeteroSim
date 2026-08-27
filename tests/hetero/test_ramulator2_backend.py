import json
from pathlib import Path

from frontend.hetero.backends.ramulator2 import (
    Ramulator2Backend,
    Ramulator2BackendConfig,
)


def test_standalone_ramulator_adapter_materializes_and_qualifies(tmp_path: Path) -> None:
    template = tmp_path / "template.yaml"
    template.write_text(
        "# {{TRACE_PATH}}\nprintf 'memory_system_cycles: 123\\n'\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "backend.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "hetero-ramulator2-backend/v1",
                "executable": "/bin/sh",
                "config_template": str(template),
                "frequency_hz": 1000000000,
                "transaction_bytes": 64,
                "timeout_seconds": 10,
                "expected_commit": "test-commit",
            }
        ),
        encoding="utf-8",
    )
    requests = [
        {"offset_bytes": 0, "size_bytes": 128, "operation": "read"},
        {"offset_bytes": 256, "size_bytes": 64, "operation": "write"},
    ]
    backend = Ramulator2Backend(Ramulator2BackendConfig.load(config_path))
    result = backend.run(requests, tmp_path / "single")
    assert result.memory_system_cycles == 123
    assert result.duration_fs == 123000000
    assert result.trace_transactions == 3
    record = backend.qualify(requests, tmp_path / "qualification")
    payload = json.loads(record.read_text())
    assert payload["status"] == "passed"
    assert payload["deterministic_equivalence"] is True
