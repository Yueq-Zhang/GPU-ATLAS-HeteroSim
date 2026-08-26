"""Content-addressed storage for replayable GPU traces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .trace_manifest import TraceManifest, TraceManifestError


class TraceCache:
    """Store trace files by program/capture identity, independent of DRAM DSE."""

    def __init__(self, root: Path):
        self.root = root

    def entry(self, trace_key: str) -> Path:
        if len(trace_key) != 64 or any(c not in "0123456789abcdef" for c in trace_key):
            raise TraceManifestError("trace_key must be a lowercase SHA-256 string")
        return self.root / trace_key

    def lookup(self, trace_key: str) -> TraceManifest | None:
        manifest_path = self.entry(trace_key) / "trace_manifest.json"
        return TraceManifest.load(manifest_path) if manifest_path.is_file() else None

    def register(self, manifest_path: Path) -> TraceManifest:
        manifest = TraceManifest.load(manifest_path)
        if manifest.kernels_list is None:
            raise TraceManifestError("cannot cache a manifest without a captured trace")
        destination = self.entry(manifest.trace_key())
        if destination.exists():
            cached = TraceManifest.load(destination / "trace_manifest.json")
            if cached.trace_key() != manifest.trace_key():
                raise TraceManifestError("trace cache key collision")
            return cached
        destination.mkdir(parents=True)
        kernels_destination = destination / manifest.kernels_list.name
        shutil.copy2(manifest.kernels_list, kernels_destination)
        for trace in manifest.kernels_list.parent.glob("*.traceg"):
            shutil.copy2(trace, destination / trace.name)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["kernels_list"] = kernels_destination.name
        (destination / "trace_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return TraceManifest.load(destination / "trace_manifest.json")
