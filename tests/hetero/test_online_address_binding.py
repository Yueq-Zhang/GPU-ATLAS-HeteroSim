from pathlib import Path

import pytest

from frontend.hetero.online_address_binding import (
    OnlineAddressBindingError,
    PackedRangeRebasePolicy,
    materialize_explicit_online_address_bindings,
    materialize_online_address_bindings,
)
from frontend.hetero.trace_manifest import SimulationBufferBinding, TraceManifest


def _manifest(trace_base: int = 0x100000) -> TraceManifest:
    return TraceManifest.from_dict(
        {
            "schema_version": "hetero-trace-manifest/v1",
            "trace_id": "online-binding-test",
            "trace_semantics": "functional",
            "replay_safe": False,
            "qualification_record": None,
            "kernels_list": "kernelslist.g",
            "capture": {"tool": "test"},
            "compilation": {"target_sm": 86},
            "address_ranges": [
                {
                    "capture_allocation_id": "capture.A",
                    "trace_base": trace_base,
                    "size_bytes": 256,
                    "tensor_id": "A",
                    "tensor_offset_bytes": 0,
                    "capture_epoch": 0,
                    "backing_allocation_id": "capture.A",
                    "view_offset_bytes": 0,
                    "alignment_bytes": 256,
                    "shape": [128],
                    "layout": "linear",
                },
                {
                    "capture_allocation_id": "capture.B",
                    "trace_base": trace_base + 0x1000,
                    "size_bytes": 128,
                    "tensor_id": "B",
                    "tensor_offset_bytes": 64,
                    "capture_epoch": 0,
                    "backing_allocation_id": "capture.B",
                    "view_offset_bytes": 64,
                    "alignment_bytes": 256,
                    "shape": [64],
                    "layout": "view",
                },
            ],
        }
    )


def _policy(**overrides: object) -> PackedRangeRebasePolicy:
    payload: dict[str, object] = {
        "mode": "range_rebase_packed_manifest",
        "memory_space_id": "shared0.dram3d",
        "physical_base_bytes": 0,
        "capacity_bytes": 4096,
        "alignment_bytes": 64,
        "require_nonzero_translations": True,
    }
    payload.update(overrides)
    return PackedRangeRebasePolicy.load(payload)


def test_materialized_binding_is_deterministic_and_preserves_tensor_offset(
    tmp_path: Path,
) -> None:
    first = materialize_online_address_bindings(_manifest(), _policy(), tmp_path / "a")
    second = materialize_online_address_bindings(_manifest(), _policy(), tmp_path / "b")
    assert first["table_sha256"] == second["table_sha256"]
    assert first["translation_point"] == "mem_fetch_before_gpu_cache_lookup"
    assert first["dram_tuple_mapping"].startswith("deferred_to_single_ramulator2")
    rows = (tmp_path / "a" / "online_address_bindings.tsv").read_text().splitlines()
    assert rows[-2] == "range\t1048576\t1048832\t0\t256"
    assert rows[-1] == "range\t1052672\t1052800\t320\t448"


def test_materialized_binding_rejects_capacity_and_ambiguous_address_spaces(
    tmp_path: Path,
) -> None:
    with pytest.raises(OnlineAddressBindingError, match="capacity"):
        materialize_online_address_bindings(
            _manifest(), _policy(capacity_bytes=128), tmp_path / "small"
        )
    with pytest.raises(OnlineAddressBindingError, match="overlap"):
        materialize_online_address_bindings(
            _manifest(0), _policy(), tmp_path / "overlap"
        )


def test_explicit_binding_preserves_global_timeline_addresses(tmp_path: Path) -> None:
    payload = materialize_explicit_online_address_bindings(
        _manifest(),
        _policy(),
        (
            SimulationBufferBinding("A", 0, 256, "shared0.dram3d", 512),
            SimulationBufferBinding("B", 0, 192, "shared0.dram3d", 1024),
        ),
        tmp_path,
    )
    rows = (tmp_path / "online_address_bindings.tsv").read_text().splitlines()
    assert payload["allocation_owner"] == "prefill_global_timeline"
    assert rows[-2] == "range\t1048576\t1048832\t512\t768"
    assert rows[-1] == "range\t1052672\t1052800\t1088\t1216"


def test_explicit_binding_rejects_overlap(tmp_path: Path) -> None:
    with pytest.raises(OnlineAddressBindingError, match="overlap"):
        materialize_explicit_online_address_bindings(
            _manifest(),
            _policy(),
            (
                SimulationBufferBinding("A", 0, 256, "shared0.dram3d", 512),
                SimulationBufferBinding("B", 0, 192, "shared0.dram3d", 640),
            ),
            tmp_path,
        )
