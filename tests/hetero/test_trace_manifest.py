import json

import pytest

from frontend.hetero.trace_manifest import (
    SimulationBufferBinding,
    TraceManifest,
    TraceManifestError,
)


def _manifest(tmp_path, *, replay_safe=False, qualification_record=None, ranges=None):
    kernels = tmp_path / "kernelslist.g"
    kernels.write_text("kernel-1.traceg\n", encoding="utf-8")
    return {
        "schema_version": "hetero-trace-manifest/v1",
        "trace_id": "vector_add.sm86",
        "trace_semantics": "functional",
        "replay_safe": replay_safe,
        "qualification_record": qualification_record,
        "kernels_list": "kernelslist.g",
        "capture": {"tool": "nvbit", "version": "1.7.3"},
        "compilation": {"target_sm": 86, "compiler": "CUDA 11.8"},
        "address_ranges": ranges or [],
    }


def _range(tensor_id, trace_base, size, tensor_offset=0):
    return {
        "capture_allocation_id": f"capture.{tensor_id}",
        "trace_base": trace_base,
        "size_bytes": size,
        "tensor_id": tensor_id,
        "tensor_offset_bytes": tensor_offset,
        "capture_epoch": 0,
        "backing_allocation_id": f"backing.{tensor_id}",
        "view_offset_bytes": 0,
        "alignment_bytes": 256,
        "shape": [size // 4],
        "layout": "contiguous_fp32",
    }


def test_capture_and_simulation_bindings_are_separate(tmp_path) -> None:
    payload = _manifest(
        tmp_path,
        ranges=[
            _range("A", "0x7f2000000000", 4096, tensor_offset=1024),
            _range("B", "0x7f3000000000", 8192),
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = TraceManifest.load(path)
    assert len(manifest.trace_key()) == 64
    normalized = manifest.normalize(0x7F2000000180)
    assert normalized.tensor_id == "A"
    assert normalized.tensor_offset == 1024 + 0x180
    translated = manifest.translate(
        0x7F2000000180,
        (
            SimulationBufferBinding(
                tensor_id="A",
                tensor_offset_bytes=0,
                size_bytes=8192,
                memory_space_id="3ddram0",
                physical_offset_bytes=0x40000000,
            ),
        ),
    )
    assert translated.memory_space_id == "3ddram0"
    assert translated.offset_bytes == 0x40000000 + 1024 + 0x180


def test_trace_and_global_pa_ranges_must_not_overlap(tmp_path) -> None:
    payload = _manifest(
        tmp_path,
        ranges=[
            _range("A", 0x1000, 0x200),
            _range("B", 0x1100, 0x100),
        ],
    )
    with pytest.raises(TraceManifestError, match="trace ranges overlap"):
        TraceManifest.from_dict(payload, tmp_path / "manifest.json")


def test_replay_safe_defaults_to_false_and_requires_qualification(tmp_path) -> None:
    manifest = TraceManifest.from_dict(_manifest(tmp_path), tmp_path / "manifest.json")
    assert manifest.replay_safe is False
    with pytest.raises(TraceManifestError, match="requires a qualification_record"):
        TraceManifest.from_dict(
            _manifest(tmp_path, replay_safe=True), tmp_path / "manifest.json"
        )


def test_trace_key_excludes_qualification_and_replay_decision(tmp_path) -> None:
    first = TraceManifest.from_dict(_manifest(tmp_path), tmp_path / "manifest.json")
    qualification = tmp_path / "qualification_record.json"
    qualification.write_text(
        json.dumps(
            {
                "schema_version": "hetero-accel-sim-qualification/v1",
                "status": "passed",
                "replay_safety_qualified": True,
                "trace_key": first.trace_key(),
            }
        ),
        encoding="utf-8",
    )
    second = TraceManifest.from_dict(
        _manifest(
            tmp_path,
            replay_safe=True,
            qualification_record=qualification.name,
        ),
        tmp_path / "manifest.json",
    )
    assert first.trace_key() == second.trace_key()
