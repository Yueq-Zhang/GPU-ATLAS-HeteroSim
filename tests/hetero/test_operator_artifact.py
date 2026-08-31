import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from frontend.hetero.operator_artifact import (
    OperatorArtifactCatalog,
    OperatorArtifactError,
    OperatorArtifactManifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path, operator: str = "attention_norm") -> Path:
    trace = tmp_path / f"{operator}.tracez"
    trace.write_bytes(b"trace")
    payload = {
        "schema_version": "hetero-operator-artifact/v1",
        "artifact_id": f"tinyllama.{operator}.prefill.ctx16",
        "source_contract": {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_spec_name": "TinyLlama-1.1B",
            "checkpoint_revision": "fe8a4e",
            "operator": operator,
            "implementation": "unit_test",
            "phase": "prefill",
            "layer_id": 0,
            "batch_size": 1,
            "context_length": 16,
            "q_len": 16,
            "kv_length": 16,
            "dtype": "fp16",
        },
        "backend": {"kind": "accel_sim", "target_sm": 86},
        "execution_contract": {
            "trace_semantics": "functional",
            "memory_traffic": "not_extracted",
            "supports_stall_resume": False,
            "compute_memory_coupled": False,
            "global_pa_binding_ready": False,
            "request_cycle_ready": False,
            "replay_safe_across_memory_candidates": False,
        },
        "address_contract": {
            "capture_address": "trace_address",
            "normalized_address": "tensor_id_plus_offset",
            "global_pa_binding": "required_at_simulation",
            "virtual_memory_mode": "identity_untranslated",
            "dram_mapping": "candidate_specific_after_global_pa",
        },
        "qualification": {
            "status": "capture_only_pending_cycle_qualification",
            "performance_eligible": False,
            "qualification_record": None,
        },
        "tensors": [
            {
                "tensor_id": f"tinyllama.layer0.{operator}.input",
                "role": "input",
                "trace_base": 4096,
                "size_bytes": 64,
                "shape": [16, 2],
                "strides": [2, 1],
                "dtype": "float16",
                "layout": "strided",
                "alignment_bytes": 256,
            }
        ],
        "files": [
            {
                "kind": "instruction_trace",
                "path": trace.name,
                "sha256": _sha(trace),
                "size_bytes": trace.stat().st_size,
            }
        ],
    }
    path = tmp_path / f"{operator}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_operator_artifact_validates_shape_address_and_hash(tmp_path: Path) -> None:
    artifact = OperatorArtifactManifest.load(_artifact(tmp_path))
    assert artifact.compatibility_key.operator == "attention_norm"
    assert artifact.compatibility_key.context_length == 16
    assert artifact.request_cycle_ready is False


def test_operator_artifact_rejects_mutated_file(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    (tmp_path / "attention_norm.tracez").write_bytes(b"mutated")
    with pytest.raises(OperatorArtifactError, match="hash mismatch"):
        OperatorArtifactManifest.load(path)


def test_request_cycle_ready_requires_full_trace_and_stall_resume(
    tmp_path: Path,
) -> None:
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload["execution_contract"]["request_cycle_ready"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OperatorArtifactError, match="Global PA binding"):
        OperatorArtifactManifest.load(path)


def test_compute_memory_coupled_is_separate_from_global_pa_readiness(
    tmp_path: Path,
) -> None:
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    execution = payload["execution_contract"]
    execution.update(
        {
            "memory_traffic": "full_instruction_trace",
            "supports_stall_resume": True,
            "compute_memory_coupled": True,
            "global_pa_binding_ready": False,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = OperatorArtifactManifest.load(path)
    assert artifact.compute_memory_coupled is True
    assert artifact.request_cycle_ready is False

    execution["request_cycle_ready"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OperatorArtifactError, match="Global PA binding"):
        OperatorArtifactManifest.load(path)

    execution["global_pa_binding_ready"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert OperatorArtifactManifest.load(path).request_cycle_ready is True


def test_optional_allocator_coverage_is_strict(tmp_path: Path) -> None:
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload["address_contract"]["capture_allocator_coverage"] = (
        "target_window_pytorch_allocator"
    )
    path.write_text(json.dumps(payload))
    OperatorArtifactManifest.load(path)
    payload["address_contract"]["capture_allocator_coverage"] = "guessed_workspace"
    path.write_text(json.dumps(payload))
    with pytest.raises(OperatorArtifactError, match="allocator"):
        OperatorArtifactManifest.load(path)


def test_gpu_artifact_builder_includes_observed_opaque_workspaces(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "heterosim-exact-llm-operator/v2",
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "model_spec_name": "TinyLlama-1.1B",
                "revision": "fe8a4e",
                "operator": "attention_norm",
                "phase": "prefill",
                "layer_id": 0,
                "batch_size": 1,
                "context_length": 16,
                "q_len": 16,
                "kv_length": 16,
                "dtype": "fp16",
                "implementation": "unit_test",
                "scope": "one_shape_locked_operator_not_end_to_end",
                "tensors": [
                    {
                        "tensor_id": "input",
                        "role": "input",
                        "address": 0x1000,
                        "size_bytes": 0x80,
                        "shape": [64],
                        "strides": [1],
                        "dtype": "float16",
                        "layout": "strided",
                        "alignment_bytes": 256,
                    },
                    {
                        "tensor_id": "output",
                        "role": "output",
                        "address": 0x1200,
                        "size_bytes": 8,
                        "shape": [1],
                        "strides": [1],
                        "dtype": "float16",
                        "layout": "strided",
                        "alignment_bytes": 256,
                    },
                ],
                "capture_allocator": {
                    "source": (
                        "pytorch_cuda_caching_allocator_target_window_plus_"
                        "tensor_segments"
                    ),
                    "device": 0,
                    "ranges": [{"address": 0x1000, "size_bytes": 0x400}],
                },
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "kernel.tracez"
    trace.write_bytes(b"trace")
    kernels = tmp_path / "kernelslist.g"
    kernels.write_text(trace.name + "\n", encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    manifest = tmp_path / "trace_manifest.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_gpu_operator_artifact.py"),
            "--metadata",
            str(metadata),
            "--kernels-list",
            str(kernels),
            "--output",
            str(artifact),
            "--trace-manifest-output",
            str(manifest),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    ranges = json.loads(manifest.read_text())["address_ranges"]
    workspaces = [item for item in ranges if item["layout"] == "opaque_allocator_range"]
    output_range = next(item for item in ranges if item["tensor_id"] == "output")
    assert output_range["size_bytes"] == 32
    assert [
        (item["trace_base"], item["size_bytes"], item["alignment_bytes"])
        for item in workspaces
    ] == [
        (0x1080, 0x180, 128),
        (0x1220, 0x1E0, 32),
    ]
    loaded = OperatorArtifactManifest.load(artifact)
    assert loaded.payload["address_contract"]["capture_allocator_coverage"] == (
        "target_window_pytorch_allocator_plus_tensor_segments"
    )


def test_catalog_reports_registration_separately_from_cycle_readiness(
    tmp_path: Path,
) -> None:
    norm = _artifact(tmp_path, "attention_norm")
    qkv = _artifact(tmp_path, "qkv_projection")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "hetero-operator-artifact-catalog/v1",
                "required_operators": [
                    "attention_norm",
                    "qkv_projection",
                    "rope",
                ],
                "zero_fallback_required": True,
                "artifacts": [norm.name, qkv.name],
            }
        ),
        encoding="utf-8",
    )
    coverage = OperatorArtifactCatalog.load(catalog_path).coverage()
    assert coverage["missing_operators"] == ["rope"]
    assert coverage["registration_complete"] is False
    assert coverage["request_cycle_coverage_complete"] is False


def test_catalog_match_includes_batch_and_context(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "hetero-operator-artifact-catalog/v1",
                "required_operators": ["attention_norm"],
                "zero_fallback_required": True,
                "artifacts": [artifact_path.name],
            }
        ),
        encoding="utf-8",
    )
    catalog = OperatorArtifactCatalog.load(catalog_path)
    common = {
        "model_spec_name": "TinyLlama-1.1B",
        "operator": "attention_norm",
        "phase": "prefill",
        "layer_id": 0,
        "q_len": 16,
        "kv_length": 16,
        "dtype": "fp16",
        "backend_kinds": {"accel_sim"},
    }
    assert catalog.match(batch_size=1, context_length=16, **common) is not None
    assert catalog.match(batch_size=2, context_length=16, **common) is None
    assert catalog.match(batch_size=1, context_length=32, **common) is None


def test_runtime_state_cannot_claim_instruction_trace_addresses(tmp_path: Path) -> None:
    path = _artifact(tmp_path, "kv_append")
    payload = json.loads(path.read_text())
    payload["backend"]["kind"] = "runtime_state"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)
    with pytest.raises(OperatorArtifactError, match="must not claim"):
        OperatorArtifactManifest.load(path)


def test_coupled_artifact_builder_preserves_global_pa_gate(tmp_path: Path) -> None:
    source = _artifact(tmp_path)
    source_payload = json.loads(source.read_text())
    backend = tmp_path / "backend.json"
    backend.write_text("{}\n", encoding="utf-8")
    memory = {
        "instances": 1,
        "reads": 4,
        "writes": 0,
        "completed": 4,
        "durable_completed": 4,
        "gpu_parents": 4,
        "gpu_completed": 4,
        "children_sent": 4,
        "children_completed": 4,
        "gpu_children": 4,
        "atlas_parents": 0,
        "atlas_completed": 0,
        "outstanding": 0,
    }
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-qualification/v1",
                "status": "passed",
                "trace_id": source_payload["artifact_id"],
                "backend_id": "gpu.accel_sim.coupled.test",
                "qualified_scopes": ["cycle_coupled_request_response"],
                "timing_ownership": {
                    "duration_mode": "coupled",
                    "external_ramulator2": "shared3d.ramulator2",
                    "gpu_local_dram": None,
                },
                "comparison": {
                    "gpu_tot_sim_cycle": [100, 100],
                    "gpu_tot_sim_insn": [200, 200],
                    "external_memory_stats": [memory, memory],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "coupled.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_coupled_gpu_operator_artifact.py"),
            "--source-artifact",
            str(source),
            "--backend-config",
            str(backend),
            "--qualification-record",
            str(qualification),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = OperatorArtifactManifest.load(output)
    assert artifact.compute_memory_coupled is True
    assert artifact.request_cycle_ready is False
    execution = artifact.payload["execution_contract"]
    assert execution["global_pa_binding_ready"] is False


def test_coupled_artifact_builder_promotes_qualified_range_rebase(
    tmp_path: Path,
) -> None:
    source = _artifact(tmp_path)
    source_payload = json.loads(source.read_text())
    source_payload["address_contract"]["capture_allocator_coverage"] = (
        "target_window_pytorch_allocator"
    )
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    backend = tmp_path / "backend.json"
    backend.write_text("{}\n", encoding="utf-8")
    memory = {
        "instances": 1,
        "reads": 4,
        "writes": 0,
        "completed": 4,
        "durable_completed": 4,
        "gpu_parents": 4,
        "gpu_completed": 4,
        "children_sent": 4,
        "children_completed": 4,
        "gpu_children": 4,
        "atlas_parents": 0,
        "atlas_completed": 0,
        "outstanding": 0,
        "address_translated": 10,
        "address_already_global": 0,
        "address_unmapped": 0,
        "address_binding_ranges": 3,
    }
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-qualification/v1",
                "status": "passed",
                "trace_id": source_payload["artifact_id"],
                "backend_id": "gpu.accel_sim.coupled.range_rebase.test",
                "qualified_scopes": ["cycle_coupled_request_response"],
                "timing_ownership": {
                    "duration_mode": "coupled",
                    "external_ramulator2": "shared3d.ramulator2",
                    "gpu_local_dram": None,
                },
                "comparison": {
                    "gpu_tot_sim_cycle": [100, 100],
                    "gpu_tot_sim_insn": [200, 200],
                    "external_memory_stats": [memory, memory],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "coupled_range_rebase.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_coupled_gpu_operator_artifact.py"),
            "--source-artifact",
            str(source),
            "--backend-config",
            str(backend),
            "--qualification-record",
            str(qualification),
            "--address-mode",
            "range_rebase",
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = OperatorArtifactManifest.load(output)
    assert artifact.compute_memory_coupled is True
    assert artifact.request_cycle_ready is True
    assert artifact.payload["execution_contract"]["global_pa_binding_ready"] is True
    assert artifact.payload["address_contract"]["virtual_memory_mode"] == "range_rebase"


def test_generic_coupled_summary_uses_catalog_and_qualification_map(
    tmp_path: Path,
) -> None:
    artifact_path = _artifact(tmp_path, "output_projection")
    payload = json.loads(artifact_path.read_text())
    source_id = payload["artifact_id"]
    payload["artifact_id"] = source_id + ".shared_hbdram_identity_v1"
    payload["execution_contract"].update(
        {
            "memory_traffic": "full_instruction_trace",
            "supports_stall_resume": True,
            "compute_memory_coupled": True,
        }
    )
    payload["qualification"]["status"] = (
        "cycle_coupled_identity_untranslated_pending_global_pa_binding"
    )
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "hetero-operator-artifact-catalog/v1",
                "required_operators": ["output_projection"],
                "zero_fallback_required": True,
                "artifacts": [artifact_path.name],
            }
        ),
        encoding="utf-8",
    )
    memory = {
        "instances": 1,
        "reads": 3,
        "writes": 1,
        "completed": 4,
        "durable_completed": 4,
        "gpu_parents": 4,
        "gpu_completed": 4,
        "children_sent": 5,
        "children_completed": 5,
        "gpu_children": 5,
        "atlas_parents": 0,
        "atlas_completed": 0,
        "atlas_children": 0,
        "logical_bytes": 256,
        "internal_bytes": 320,
        "cycles": 12,
        "link_cycles": 7,
        "rejected": 2,
        "outstanding": 0,
    }
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-qualification/v1",
                "status": "passed",
                "trace_id": source_id,
                "replay_safety_qualified": False,
                "qualified_scopes": ["cycle_coupled_request_response"],
                "timing_ownership": {
                    "duration_mode": "coupled",
                    "external_ramulator2": "shared3d.ramulator2",
                    "gpu_local_dram": None,
                },
                "comparison": {
                    "gpu_tot_sim_cycle": [100, 100],
                    "gpu_tot_sim_insn": [200, 200],
                    "external_memory_stats": [memory, memory],
                },
            }
        ),
        encoding="utf-8",
    )
    qualification_map = tmp_path / "qualification_map.json"
    qualification_map.write_text(
        json.dumps(
            {
                "schema_version": "hetero-coupled-qualification-map/v1",
                "records": {"output_projection": qualification.name},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "scripts"
                / "summarize_coupled_gpu_operator_artifacts.py"
            ),
            "--catalog",
            str(catalog),
            "--qualification-map",
            str(qualification_map),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(output.read_text())
    assert summary["status"] == "passed"
    assert summary["aggregate"] == {
        "operator_count": 1,
        "gpu_parents": 4,
        "gpu_children": 5,
    }
    assert summary["claim_boundary"]["request_cycle_ready"] is False


def test_range_rebased_summary_requires_translation_and_conservation(
    tmp_path: Path,
) -> None:
    artifact_path = _artifact(tmp_path, "qkv_projection")
    payload = json.loads(artifact_path.read_text())
    source_id = payload["artifact_id"]
    payload["artifact_id"] = source_id + ".shared_hbdram_range_rebase_v1"
    payload["execution_contract"].update(
        {
            "memory_traffic": "full_instruction_trace",
            "supports_stall_resume": True,
            "compute_memory_coupled": True,
            "global_pa_binding_ready": True,
            "request_cycle_ready": True,
        }
    )
    payload["address_contract"].update(
        {
            "virtual_memory_mode": "range_rebase",
            "global_pa_binding": "required_at_simulation",
            "capture_allocator_coverage": (
                "target_window_pytorch_allocator_plus_tensor_segments"
            ),
        }
    )
    payload["qualification"].update(
        {
            "status": "cycle_coupled_range_rebased_pending_global_timeline",
            "performance_eligible": False,
        }
    )
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "hetero-operator-artifact-catalog/v1",
                "required_operators": ["qkv_projection"],
                "zero_fallback_required": True,
                "artifacts": [artifact_path.name],
            }
        ),
        encoding="utf-8",
    )
    memory = {
        "instances": 1,
        "reads": 3,
        "writes": 1,
        "completed": 4,
        "durable_completed": 4,
        "gpu_parents": 4,
        "gpu_completed": 4,
        "children_sent": 5,
        "children_completed": 5,
        "gpu_children": 5,
        "atlas_parents": 0,
        "atlas_completed": 0,
        "atlas_children": 0,
        "logical_bytes": 256,
        "internal_bytes": 320,
        "cycles": 12,
        "link_cycles": 7,
        "rejected": 2,
        "outstanding": 0,
        "address_translated": 10,
        "address_already_global": 0,
        "address_unmapped": 0,
        "address_binding_ranges": 3,
    }
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-qualification/v1",
                "status": "passed",
                "trace_id": source_id,
                "replay_safety_qualified": False,
                "qualified_scopes": ["cycle_coupled_request_response"],
                "timing_ownership": {
                    "duration_mode": "coupled",
                    "external_ramulator2": "shared3d.ramulator2",
                    "gpu_local_dram": None,
                },
                "comparison": {
                    "gpu_tot_sim_cycle": [100, 100],
                    "gpu_tot_sim_insn": [200, 200],
                    "external_memory_stats": [memory, memory],
                },
            }
        ),
        encoding="utf-8",
    )
    qualification_map = tmp_path / "qualification_map.json"
    qualification_map.write_text(
        json.dumps(
            {
                "schema_version": "hetero-coupled-qualification-map/v1",
                "records": {"qkv_projection": qualification.name},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    project_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "scripts"
                / "summarize_range_rebased_gpu_operator_artifacts.py"
            ),
            "--catalog",
            str(catalog),
            "--qualification-map",
            str(qualification_map),
            "--output",
            str(output),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(output.read_text())
    assert summary["status"] == "passed"
    assert summary["aggregate"] == {
        "operator_count": 1,
        "gpu_parents": 4,
        "gpu_children": 5,
        "address_translated": 10,
        "logical_bytes": 256,
        "internal_bytes": 320,
    }
    assert summary["claim_boundary"]["request_cycle_ready"] is True
