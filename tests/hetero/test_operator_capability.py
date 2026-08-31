import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from frontend.hetero.model_graph import ModelSpec, RequestSpec, build_request_graph
from frontend.hetero.operator_capability import (
    OperatorCapabilityCatalog,
    OperatorCapabilityError,
)
from frontend.hetero.operator_artifact import OperatorArtifactManifest


CATALOG = Path(
    "configs/hetero/operator_capabilities/"
    "tinyllama_prefill_layer0_bs1_ctx16.json"
)


def _model(**overrides: object) -> ModelSpec:
    values: dict[str, object] = {
        "name": "TinyLlama-1.1B",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_layers": 1,
        "num_attention_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 64,
        "vocab_size": 32000,
        "dtype": "fp16",
        "materialize_parameters": True,
        "checkpoint_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    }
    values.update(overrides)
    return ModelSpec(**values)  # type: ignore[arg-type]


def test_reference_capability_catalog_is_complete_and_truthful() -> None:
    catalog = OperatorCapabilityCatalog.load(CATALOG)
    summary = catalog.summary()
    assert summary["operator_type_count"] == 19
    assert summary["reference_graph_instance_count"] == 20
    assert summary["request_cycle_ready_operator_types"] == 17
    assert summary["performance_eligible_operator_types"] == 0
    assert catalog.operators["residual_add"].instances_in_reference_graph == 2
    assert catalog.operators["kv_append"].backend_kind == "runtime_live_ramulator2"
    assert catalog.operators["kv_append"].test_status == "request_cycle_qualified"
    assert catalog.operators["kv_append"].request_cycle_ready is True
    assert catalog.operators["request_start"].backend_kind == "event_marker"
    assert catalog.operators["request_start"].cycle_fidelity == "event_only"
    assert catalog.operators["token_embedding"].request_cycle_ready is True

    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    project_root = Path.cwd()
    for item in payload["operator_types"]:
        for ref in item["artifact_refs"]:
            assert (project_root / ref).is_file(), ref


def test_capability_catalog_exactly_covers_reference_graph_instances() -> None:
    catalog = OperatorCapabilityCatalog.load(CATALOG)
    graph = build_request_graph(_model(), RequestSpec("R0", 16, 1))
    expected = Counter(node.op for node in graph.nodes)
    recorded = {
        name: item.instances_in_reference_graph
        for name, item in catalog.operators.items()
    }
    assert recorded == dict(expected)


def test_exact_shape_contract_accepts_only_qualified_shape() -> None:
    catalog = OperatorCapabilityCatalog.load(CATALOG)
    capability = catalog.require_exact_shape(
        "attention_norm",
        _model(),
        phase="prefill",
        layer_id=0,
        batch_size=1,
        context_length=16,
        q_len=16,
        kv_length=16,
        require_request_cycle_ready=True,
    )
    assert capability.request_cycle_ready is True

    with pytest.raises(OperatorCapabilityError, match="no exact tested shape"):
        catalog.require_exact_shape(
            "attention_norm",
            _model(),
            phase="prefill",
            layer_id=0,
            batch_size=1,
            context_length=32,
            q_len=32,
            kv_length=32,
            require_request_cycle_ready=True,
        )


def test_model_dimensions_and_revision_are_part_of_coverage_identity() -> None:
    catalog = OperatorCapabilityCatalog.load(CATALOG)
    common = {
        "phase": "prefill",
        "layer_id": 0,
        "batch_size": 1,
        "context_length": 16,
        "q_len": 16,
        "kv_length": 16,
        "require_request_cycle_ready": True,
    }
    with pytest.raises(OperatorCapabilityError, match="hidden_size"):
        catalog.require_exact_shape(
            "attention_norm",
            _model(
                hidden_size=4096,
                num_attention_heads=64,
            ),
            **common,
        )
    with pytest.raises(OperatorCapabilityError, match="checkpoint_revision"):
        catalog.require_exact_shape(
            "attention_norm",
            _model(checkpoint_revision="different-revision"),
            **common,
        )


def test_host_control_event_cannot_pass_request_cycle_gate() -> None:
    catalog = OperatorCapabilityCatalog.load(CATALOG)
    with pytest.raises(OperatorCapabilityError, match="not request-cycle qualified"):
        catalog.require_exact_shape(
            "request_start",
            _model(),
            phase="control",
            layer_id=0,
            batch_size=1,
            context_length=16,
            q_len=16,
            kv_length=16,
            require_request_cycle_ready=True,
        )


@pytest.mark.parametrize(
    ("operator", "cycles", "instructions", "translated", "parents"),
    (
        ("token_embedding", 6691, 851_968, 5120, 132),
        ("residual_add", 28_772, 491_520, 6144, 4096),
    ),
)
def test_p16_simple_gpu_operators_have_deterministic_range_rebase_evidence(
    operator: str,
    cycles: int,
    instructions: int,
    translated: int,
    parents: int,
) -> None:
    root = Path("configs/hetero/operator_artifacts/p16")
    artifact = OperatorArtifactManifest.load(
        root
        / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_"
        "shared_hbdram_range_rebase.json"
    )
    assert artifact.request_cycle_ready is True
    trace = json.loads(
        (
            root
            / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["compilation"]["framework"] == "standalone_cuda"
    assert trace["compilation"]["cuda_toolkit"] == "11.8"
    record = json.loads(
        (root / "qualification_records" / f"{operator}_range_rebase.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "passed"
    assert record["comparison"]["gpu_tot_sim_cycle"] == [cycles, cycles]
    assert record["comparison"]["gpu_tot_sim_insn"] == [instructions, instructions]
    first, second = record["comparison"]["external_memory_stats"]
    assert first == second
    assert first["address_translated"] == translated
    assert first["address_unmapped"] == 0
    assert first["gpu_parents"] == parents
    assert first["completed"] == parents
    assert first["durable_completed"] == parents
    assert first["children_sent"] == first["children_completed"]
    assert first["atlas_parents"] == 0
    assert first["instances"] == 1
    assert first["outstanding"] == 0


def test_p16_experiment_assigns_all_twenty_tasks_to_explicit_models() -> None:
    config = json.loads(
        Path(
            "configs/hetero/experiments/"
            "p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu.json"
        ).read_text(encoding="utf-8")
    )
    gpu = config["backends"]["gpu"]
    graph = build_request_graph(_model(), RequestSpec("R0", 16, 1))
    runtime_ops = set(gpu["runtime_task_operators"])
    trace_tasks = 0
    runtime_tasks = 0
    for node in graph.nodes:
        matches = []
        for binding in gpu["trace_bindings"]:
            selector = binding["selector"]
            actual = {
                "node_id": node.node_id,
                "phase": node.phase.value,
                "op": node.op,
                "layer_id": node.layer_id,
                "step_id": node.step_id,
                "operator_group": node.attributes.get("operator_group"),
            }
            if all(actual.get(key) == value for key, value in selector.items()):
                matches.append(binding)
        assert len(matches) <= 1, node.node_id
        if matches:
            trace_tasks += 1
        else:
            assert node.op in runtime_ops, node.node_id
            runtime_tasks += 1
    assert trace_tasks == 15
    assert runtime_tasks == 5
    assert gpu["fallback_kind"] == "none"


def test_p16_repository_local_artifact_files_match_recorded_hashes() -> None:
    config = json.loads(
        Path(
            "configs/hetero/experiments/"
            "p16_tinyllama_prefill_1layer_ctx16_full_task_models_gpu.json"
        ).read_text(encoding="utf-8")
    )
    checked: set[Path] = set()
    for binding in config["backends"]["gpu"]["trace_bindings"]:
        artifact_path = Path(binding["operator_artifact"])
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        for record in payload["files"]:
            recorded_path = Path(record["path"])
            if recorded_path.is_absolute():
                continue
            local_path = (artifact_path.parent / recorded_path).resolve()
            if local_path in checked:
                continue
            data = local_path.read_bytes()
            assert len(data) == record["size_bytes"], local_path
            assert hashlib.sha256(data).hexdigest() == record["sha256"], local_path
            checked.add(local_path)
    assert checked
