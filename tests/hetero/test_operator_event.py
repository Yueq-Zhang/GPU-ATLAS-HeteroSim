import json
import os
from pathlib import Path

import pytest

from frontend.hetero.global_memory_map import GlobalAllocation
from frontend.hetero.ir import ModelNode, NodeKind, Phase
from frontend.hetero.model_graph import ModelSpec
from frontend.hetero.operator_artifact import (
    OperatorArtifactManifest,
    OperatorCompatibilityKey,
)
from frontend.hetero.operator_event import (
    OperatorEventDispatcher,
    OperatorEventError,
    TraceBinding,
    _reserved_allocation_extents,
)
from frontend.hetero.runner import execute_run
from frontend.hetero.schema import load_and_validate_config, validate_config
from frontend.hetero.trace_manifest import TraceManifest


def test_global_pa_reservation_exposes_alignment_padding() -> None:
    allocations = {
        "token": GlobalAllocation(
            "token", "shared0.dram3d", 0, 8, 64, "activation", "int64"
        ),
        "next": GlobalAllocation(
            "next", "shared0.dram3d", 64, 16, 64, "activation", "fp16"
        ),
    }
    assert _reserved_allocation_extents(allocations, 128) == {
        "token": 64,
        "next": 64,
    }


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


def test_duplicate_operator_contention_backend_is_rejected_by_dispatcher(
    tmp_path: Path,
) -> None:
    backends = {
        "gpu": {
            "kind": "accel_sim",
            "config_ref": (
                "configs/hetero/backends/"
                "gpu_accelsim_rtx3070_full_atlas_chip_shared_hbdram_edge_16ch.json"
            ),
        },
        "atlas": {"kind": "none"},
        "host": {"kind": "none"},
    }
    with pytest.raises(
        OperatorEventError, match="duplicate-operator contention stress"
    ):
        OperatorEventDispatcher(Path.cwd(), tmp_path / "backend_runs", backends)


def test_runtime_cycle_fallback_dispatches_shape_locked_control_task(
    tmp_path: Path,
) -> None:
    config = _operator_config(tmp_path)
    gpu = config["backends"]["gpu"]  # type: ignore[index]
    gpu["fallback_kind"] = "runtime_cycle"
    gpu["runtime_task_model_ref"] = str(
        Path(
            "configs/hetero/runtime_tasks/"
            "tinyllama_prefill_layer0_bs1_ctx16_uncalibrated.json"
        ).resolve()
    )
    validate_config(config)
    dispatcher = OperatorEventDispatcher(
        Path.cwd(), tmp_path / "backend_runs", config["backends"]  # type: ignore[arg-type]
    )
    model = ModelSpec(
        "TinyLlama-1.1B",
        2048,
        5632,
        1,
        32,
        4,
        64,
        32000,
        dtype="fp16",
        checkpoint_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    )
    node = ModelNode(
        "request.start",
        NodeKind.CONTROL,
        "request_start",
        Phase.CONTROL,
        0,
        0,
        attributes={
            "batch_size": 1,
            "context_length": 16,
            "q_len": 16,
            "attention_kv_len": 16,
        },
    )
    result = dispatcher.dispatch("gpu", node, model, "gpu0")
    assert result.statistics["cycles"] == 1
    assert result.fidelity["compute_fidelity"] == (
        "host_control_boundary_uncalibrated"
    )
    assert result.fidelity["device_performance_included"] is False
    assert result.fidelity["performance_eligible"] is False


def test_runtime_cycle_operator_allowlist_precedes_analytical_fallback(
    tmp_path: Path,
) -> None:
    config = _operator_config(tmp_path)
    gpu = config["backends"]["gpu"]  # type: ignore[index]
    gpu["fallback_kind"] = "analytical"
    gpu["runtime_task_model_ref"] = str(
        Path(
            "configs/hetero/runtime_tasks/"
            "tinyllama_prefill_layer0_bs1_ctx16_uncalibrated.json"
        ).resolve()
    )
    gpu["runtime_task_operators"] = ["request_start"]
    validate_config(config)
    dispatcher = OperatorEventDispatcher(
        Path.cwd(), tmp_path / "backend_runs", config["backends"]  # type: ignore[arg-type]
    )
    model = ModelSpec(
        "TinyLlama-1.1B",
        2048,
        5632,
        1,
        32,
        4,
        64,
        32000,
        dtype="fp16",
        checkpoint_revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    )
    node = ModelNode(
        "request.start",
        NodeKind.CONTROL,
        "request_start",
        Phase.CONTROL,
        0,
        0,
        attributes={"batch_size": 1, "context_length": 16},
    )
    result = dispatcher.dispatch("gpu", node, model, "gpu0")
    assert result.artifact["kind"] == "host_control_event"
    assert dispatcher.provenance()["runtime_task_operators"] == ["request_start"]


def test_shape_locked_trace_binding_rejects_wrong_prefill_length() -> None:
    key = OperatorCompatibilityKey(
        model_spec_name="TinyLlama-1.1B",
        checkpoint_revision="fe8a4e",
        operator="attention_norm",
        phase="prefill",
        layer_id=0,
        batch_size=1,
        context_length=16,
        q_len=16,
        kv_length=16,
        dtype="fp16",
    )
    artifact = OperatorArtifactManifest(
        Path("artifact.json"),
        "fake.norm",
        key,
        {"backend": {"kind": "accel_sim"}},
        "hash",
    )
    manifest = TraceManifest.from_dict(
        {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": "fake.norm",
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": "kernelslist.g",
            "capture": {},
            "compilation": {},
            "address_ranges": [],
        }
    )
    binding = TraceBinding(
        {"op": "attention_norm"}, manifest, "exact_operator", artifact
    )
    node = ModelNode(
        "n0",
        NodeKind.COMPUTE,
        "attention_norm",
        Phase.PREFILL,
        0,
        0,
        attributes={"q_len": 32, "attention_kv_len": 32},
    )
    model = ModelSpec(
        "TinyLlama-1.1B", 2048, 5632, 1, 32, 4, 64, 32000, dtype="fp16"
    )
    with pytest.raises(OperatorEventError, match="q_len"):
        binding.validate_exact_contract(node, model)


def test_shape_locked_trace_binding_accepts_explicit_layer_contract_override() -> None:
    key = OperatorCompatibilityKey(
        model_spec_name="TinyLlama-1.1B",
        checkpoint_revision="fe8a4e",
        operator="final_norm",
        phase="prefill",
        layer_id=0,
        batch_size=1,
        context_length=16,
        q_len=16,
        kv_length=16,
        dtype="fp16",
    )
    artifact = OperatorArtifactManifest(
        Path("artifact.json"),
        "fake.final_norm",
        key,
        {"backend": {"kind": "accel_sim"}},
        "hash",
    )
    manifest = TraceManifest.from_dict(
        {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": "fake.final_norm",
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": "kernelslist.g",
            "capture": {},
            "compilation": {},
            "address_ranges": [],
        }
    )
    binding = TraceBinding(
        {"op": "final_norm"},
        manifest,
        "exact_operator",
        artifact,
        contract_overrides={"layer_id": 0},
    )
    node = ModelNode(
        "n0",
        NodeKind.COMPUTE,
        "final_norm",
        Phase.PREFILL,
        None,
        0,
        attributes={"q_len": 16, "attention_kv_len": 16},
    )
    model = ModelSpec(
        "TinyLlama-1.1B", 2048, 5632, 1, 32, 4, 64, 32000, dtype="fp16"
    )
    binding.validate_exact_contract(node, model)


def test_shape_locked_trace_binding_rejects_batch_context_and_revision_changes() -> None:
    key = OperatorCompatibilityKey(
        model_spec_name="TinyLlama-1.1B",
        checkpoint_revision="qualified-revision",
        operator="attention_norm",
        phase="prefill",
        layer_id=0,
        batch_size=1,
        context_length=16,
        q_len=16,
        kv_length=16,
        dtype="fp16",
    )
    artifact = OperatorArtifactManifest(
        Path("artifact.json"),
        "fake.norm",
        key,
        {"backend": {"kind": "accel_sim"}},
        "hash",
    )
    manifest = TraceManifest.from_dict(
        {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": "fake.norm",
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": "kernelslist.g",
            "capture": {},
            "compilation": {},
            "address_ranges": [],
        }
    )
    binding = TraceBinding(
        {"op": "attention_norm"}, manifest, "exact_operator", artifact
    )
    node = ModelNode(
        "n0",
        NodeKind.COMPUTE,
        "attention_norm",
        Phase.PREFILL,
        0,
        0,
        attributes={
            "batch_size": 2,
            "context_length": 32,
            "q_len": 16,
            "attention_kv_len": 16,
        },
    )
    model = ModelSpec(
        "TinyLlama-1.1B",
        2048,
        5632,
        1,
        32,
        4,
        64,
        32000,
        dtype="fp16",
        checkpoint_revision="different-revision",
    )
    with pytest.raises(OperatorEventError) as error:
        binding.validate_exact_contract(node, model)
    message = str(error.value)
    assert "batch_size" in message
    assert "context_length" in message
    assert "checkpoint_revision" in message


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
    online = json.loads((run_dir / "online_dispatch.json").read_text())

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
    assert provenance["runtime_owner"] == "python.OnlineOperatorRuntime"
    assert execution["placement_contract"]["backend_dispatch_count"] == len(
        execution["tasks"]
    )
    assert execution["placement_contract"]["online_dispatch_gate"][
        "backend_launches_after_dependencies"
    ] is True
    assert online["backend_dispatch_count"] == len(execution["tasks"])
    timing = {
        record["task_id"]: record["timing"]
        for record in [*execution["tasks"], *execution["routes"]]
    }
    for task in execution["tasks"]:
        if task["dependencies"]:
            assert task["backend_launch_time_fs"] >= max(
                timing[dependency]["completion_time_fs"]
                for dependency in task["dependencies"]
            )
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
