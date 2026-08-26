import json
import math
import os

import pytest

from frontend.hetero.backends.accel_sim import (
    AccelSimBackend,
    AccelSimBackendConfig,
    parse_accel_sim_stats,
)
from frontend.hetero.trace_manifest import TraceManifest


def _files(tmp_path):
    executable = tmp_path / "fake-accel-sim"
    executable.write_text(
        "#!/usr/bin/env sh\n"
        "printf 'gpu_tot_sim_cycle = 1132\\n'\n"
        "printf 'gpu_tot_sim_insn = 4096\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    gpgpu = tmp_path / "gpgpusim.config"
    trace_config = tmp_path / "trace.config"
    kernels = tmp_path / "kernelslist.g"
    for path in (gpgpu, trace_config, kernels):
        path.write_text("test\n", encoding="utf-8")
    config = tmp_path / "backend.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-backend/v1",
                "backend_id": "test.accel_sim",
                "executable": executable.name,
                "gpgpu_config": gpgpu.name,
                "trace_config": trace_config.name,
                "target_gpu": "test",
                "target_sm": 86,
                "core_frequency_hz": 1_132_000_000,
                "timeout_seconds": 10,
                "dependency_commits": {"accel_sim": "c5296df"},
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": "test",
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": kernels.name,
            "capture": {"tool": "test"},
            "compilation": {"target_sm": 86},
            "address_ranges": [],
        }),
        encoding="utf-8",
    )
    manifest = TraceManifest.load(manifest_path)
    return config, manifest


def test_stat_parser_accepts_integer_float_and_scientific_values() -> None:
    stats = parse_accel_sim_stats(
        "gpu_tot_sim_cycle = 1132\n"
        "gpu_tot_sim_insn = 4096\n"
        "gpu_ipc = 3.618\n"
        "L2_BW = 1.25e+02\n"
    )
    assert stats == {
        "gpu_tot_sim_cycle": 1132,
        "gpu_tot_sim_insn": 4096,
        "gpu_ipc": 3.618,
        "L2_BW": 125.0,
    }


def test_command_preserves_native_accel_sim_contract(tmp_path) -> None:
    config_path, manifest = _files(tmp_path)
    backend = AccelSimBackend(AccelSimBackendConfig.load(config_path))
    command = backend.command(manifest)
    assert command[1:3] == ("-trace", str(manifest.kernels_list))
    assert command.count("-config") == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX test executable")
def test_qualification_requires_exact_native_adapter_match(tmp_path) -> None:
    config_path, manifest = _files(tmp_path)
    backend = AccelSimBackend(AccelSimBackendConfig.load(config_path))
    record_path = backend.qualify(manifest, tmp_path / "qualification")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["comparison"]["gpu_tot_sim_cycle"] == [1132, 1132]
    assert record["timing_ownership"]["external_ramulator2"] is False
    qualified = TraceManifest.load(
        tmp_path / "qualification" / "qualified_trace_manifest.json"
    )
    assert qualified.replay_safe is True
    stats = json.loads(
        (tmp_path / "qualification" / "adapter" / "stats.json").read_text()
    )
    assert stats["duration_fs"] == math.ceil(1132 * 10**15 / 1_132_000_000)
