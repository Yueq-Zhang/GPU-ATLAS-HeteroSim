import json
import os
from pathlib import Path

import pytest

from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config, validate_config


def _fake_accel_sim_files(tmp_path: Path) -> tuple[Path, Path]:
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
    backend_config = tmp_path / "backend.json"
    backend_config.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-backend/v1",
                "backend_id": "gpu.accel_sim.fake",
                "executable": executable.name,
                "gpgpu_config": gpgpu.name,
                "trace_config": trace_config.name,
                "target_gpu": "fake",
                "target_sm": 86,
                "core_frequency_hz": 1_132_000_000,
                "timeout_seconds": 10,
                "dependency_commits": {"accel_sim": "c5296df"},
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "hetero-trace-manifest/v1",
                "trace_id": "fake.qkv",
                "trace_semantics": "functional",
                "replay_safe": False,
                "qualification_record": None,
                "kernels_list": kernels.name,
                "capture": {"tool": "test"},
                "compilation": {"target_sm": 86},
                "address_ranges": [],
            }
        ),
        encoding="utf-8",
    )
    return backend_config, manifest


def _operator_config(tmp_path: Path) -> dict[str, object]:
    backend_config, manifest = _fake_accel_sim_files(tmp_path)
    config = load_and_validate_config(
        "configs/hetero/experiments/m2_model1_analytical_preview.json"
    )
    config["experiment"]["name"] = "operator_event_fake"  # type: ignore[index]
    config["simulation"] = {  # type: ignore[index]
        "coupling": "operator_event",
        "execution_mode": "operator_event",
    }
    config["backends"]["gpu"] = {  # type: ignore[index]
        "kind": "accel_sim",
        "requested_timing_mode": "total",
        "config_ref": str(backend_config),
        "resource_bindings": {
            "gpu_core": "gpu0.core",
            "gpu_l1": "gpu0.l1",
            "gpu_l2": "gpu0.l2",
            "gpu_noc": "gpu0.noc",
            "gpu_local_dram": "gpu0.hbm",
        },
        "trace_bindings": [
            {
                "selector": {
                    "phase": "prefill",
                    "op": "qkv_projection",
                    "layer_id": 0,
                    "step_id": 0,
                },
                "trace_manifest": str(manifest),
                "compatibility": "exact_operator",
            }
        ],
        "fallback_kind": "analytical",
        "effective_compute_flops_per_s": 100_000_000_000_000,
        "effective_memory_bandwidth_Bps": 1_000_000_000_000,
        "parameter_source": "unit_test",
    }
    validate_config(config)
    return config


@pytest.mark.skipif(os.name == "nt", reason="POSIX fake Accel-Sim executable")
def test_operator_event_run_dispatches_trace_and_reuses_simulation_cache(
    tmp_path: Path,
) -> None:
    config = _operator_config(tmp_path)
    run_dir = execute_run(config, Path.cwd(), tmp_path / "runs")
    execution = json.loads((run_dir / "execution_graph.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    provenance = json.loads((run_dir / "provenance.json").read_text())
    trace_bundle = json.loads((run_dir / "trace_manifest.json").read_text())

    traced = [
        task for task in execution["tasks"] if task["fidelity"]["trace_coverage"] == 1.0
    ]
    assert len(traced) == 1
    assert traced[0]["backend_statistics"]["cycles"] == 1132
    assert traced[0]["backend_statistics"]["instructions"] == 4096
    assert traced[0]["backend_statistics"]["cache_hit"] is False
    assert traced[0]["timing_contract"]["duration_semantics"] == "total"
    assert traced[0]["timing_contract"]["exports"] == []
    assert metrics["run_status"] == "operator_event"
    assert 0 < metrics["fidelity"]["trace_coverage"] < 1
    assert metrics["performance_claim_allowed"] is False
    assert provenance["runtime_owner"] == "cpp.GlobalEventRuntime"
    assert provenance["backend_dispatch"]["timing_owners"]["gpu0.hbm"] == (
        "gpu.accel_sim.fake"
    )
    assert trace_bundle["schema_version"] == "hetero-run-trace-bundle/v1"
    assert trace_bundle["captures"][0]["task_ids"] == [
        "R0.prefill.s0.l0.attention.projection"
    ]

    second_run = execute_run(config, Path.cwd(), tmp_path / "runs")
    second = json.loads((second_run / "execution_graph.json").read_text())
    second_traced = [
        task for task in second["tasks"] if task["fidelity"]["trace_coverage"] == 1.0
    ]
    assert second_traced[0]["backend_statistics"]["cache_hit"] is True
