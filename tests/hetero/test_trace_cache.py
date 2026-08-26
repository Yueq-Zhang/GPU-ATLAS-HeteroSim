import json

from frontend.hetero.trace_cache import TraceCache


def test_trace_cache_is_content_addressed_and_keeps_trace_files(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel-1.traceg").write_text("trace\n", encoding="utf-8")
    (source / "kernelslist.g").write_text("kernel-1.traceg\n", encoding="utf-8")
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "hetero-trace-manifest/v1",
                "trace_id": "vector-add",
                "trace_semantics": "functional",
                "replay_safe": False,
                "qualification_record": None,
                "kernels_list": "kernelslist.g",
                "capture": {"tool": "nvbit", "version": "1.7.3"},
                "compilation": {"target_sm": 86},
                "address_ranges": [],
            }
        ),
        encoding="utf-8",
    )
    cache = TraceCache(tmp_path / "cache")
    cached = cache.register(manifest_path)
    assert cached.kernels_list.is_file()
    assert (cached.kernels_list.parent / "kernel-1.traceg").is_file()
    assert cache.lookup(cached.trace_key()).trace_id == "vector-add"
