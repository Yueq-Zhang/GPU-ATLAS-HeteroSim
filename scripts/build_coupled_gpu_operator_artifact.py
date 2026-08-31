#!/usr/bin/env python3
"""Promote a standalone GPU Artifact with cycle-coupled memory evidence.

The promoted Artifact deliberately remains request-cycle-unready until a
runtime TensorID+offset to Global-PA binding has been applied to every traced
memory request.  Existing P9b's identity-untranslated address path is useful
coupling evidence, but it is not that final address contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from frontend.hetero.operator_artifact import OperatorArtifactManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact file path must be a non-empty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _file(path: Path, base: Path, kind: str) -> dict[str, object]:
    path = path.resolve()
    try:
        rendered = str(path.relative_to(base))
    except ValueError:
        rendered = str(path)
    return {
        "kind": kind,
        "path": rendered,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--qualification-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--address-mode",
        choices=("identity_untranslated", "range_rebase"),
        default="identity_untranslated",
    )
    args = parser.parse_args()

    source_path = args.source_artifact.resolve()
    backend_path = args.backend_config.resolve()
    qualification_path = args.qualification_record.resolve()
    output = args.output.resolve()
    source = OperatorArtifactManifest.load(source_path)
    source_backend = _mapping(source.payload["backend"], "backend")
    if source_backend.get("kind") != "accel_sim":
        raise ValueError("source Artifact must use the accel_sim backend")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (
        qualification.get("schema_version")
        != "hetero-accel-sim-qualification/v1"
        or qualification.get("status") != "passed"
        or qualification.get("trace_id") != source.artifact_id
        or "cycle_coupled_request_response"
        not in qualification.get("qualified_scopes", [])
    ):
        raise ValueError("qualification does not cover this cycle-coupled Artifact")
    ownership = _mapping(qualification.get("timing_ownership"), "timing_ownership")
    if (
        ownership.get("duration_mode") != "coupled"
        or ownership.get("external_ramulator2") != "shared3d.ramulator2"
        or ownership.get("gpu_local_dram") is not None
    ):
        raise ValueError("qualification has incompatible timing ownership")
    comparison = _mapping(qualification.get("comparison"), "comparison")
    cycles = comparison.get("gpu_tot_sim_cycle")
    instructions = comparison.get("gpu_tot_sim_insn")
    memories = comparison.get("external_memory_stats")
    if (
        not isinstance(cycles, list)
        or len(cycles) != 2
        or len(set(cycles)) != 1
        or not isinstance(instructions, list)
        or len(instructions) != 2
        or len(set(instructions)) != 1
        or not isinstance(memories, list)
        or len(memories) != 2
        or memories[0] != memories[1]
    ):
        raise ValueError("cycle-coupled qualification is not deterministic")
    memory = _mapping(memories[0], "external_memory_stats[0]")
    accepted = int(memory.get("reads", 0)) + int(memory.get("writes", 0))
    if (
        int(memory.get("instances", 0)) != 1
        or accepted <= 0
        or int(memory.get("completed", -1)) != accepted
        or int(memory.get("durable_completed", accepted)) != accepted
        or int(memory.get("gpu_parents", -1)) != accepted
        or int(memory.get("gpu_completed", accepted)) != accepted
        or int(memory.get("atlas_parents", -1)) != 0
        or int(memory.get("atlas_completed", 0)) != 0
        or int(memory.get("children_sent", -1)) <= 0
        or int(memory.get("children_completed", -1))
        != int(memory.get("children_sent", -2))
        or int(memory.get("gpu_children", memory.get("children_sent", -1)))
        != int(memory.get("children_sent", -2))
        or int(memory.get("outstanding", -1)) != 0
    ):
        raise ValueError("cycle-coupled memory conservation failed")
    range_rebased = args.address_mode == "range_rebase"
    if range_rebased and (
        int(memory.get("address_translated", 0)) <= 0
        or int(memory.get("address_unmapped", -1)) != 0
        or int(memory.get("address_binding_ranges", 0)) <= 0
    ):
        raise ValueError(
            "range_rebase promotion requires translated requests, binding ranges, "
            "and zero unmapped requests"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent
    source_files = source.payload["files"]
    assert isinstance(source_files, list)
    files: list[dict[str, object]] = []
    for index, item in enumerate(source_files):
        mapping = _mapping(item, f"files[{index}]")
        files.append(
            _file(
                _resolve(source_path.parent, mapping.get("path")),
                base,
                str(mapping.get("kind", "source_file")),
            )
        )
    files.extend(
        (
            _file(source_path, base, "standalone_operator_artifact"),
            _file(backend_path, base, "coupled_accel_sim_backend"),
            _file(qualification_path, base, "cycle_coupled_qualification_record"),
        )
    )
    payload = {
        "schema_version": "hetero-operator-artifact/v1",
        "artifact_id": source.artifact_id
        + (
            ".shared_hbdram_range_rebase_v1"
            if range_rebased
            else ".shared_hbdram_identity_v1"
        ),
        "source_contract": dict(
            _mapping(source.payload["source_contract"], "source_contract")
        ),
        "backend": {
            **dict(source_backend),
            "backend_id": qualification["backend_id"],
            "memory_coupling": "ramulator2_in_process",
        },
        "execution_contract": {
            "trace_semantics": "functional",
            "memory_traffic": "full_instruction_trace",
            "supports_stall_resume": True,
            "compute_memory_coupled": True,
            "global_pa_binding_ready": range_rebased,
            "request_cycle_ready": range_rebased,
            "replay_safe_across_memory_candidates": False,
        },
        "address_contract": {
            **dict(_mapping(source.payload["address_contract"], "address_contract")),
            "virtual_memory_mode": (
                "range_rebase" if range_rebased else "identity_untranslated"
            ),
        },
        "qualification": {
            "status": (
                "cycle_coupled_range_rebased_pending_global_timeline"
                if range_rebased
                else "cycle_coupled_identity_untranslated_pending_global_pa_binding"
            ),
            "performance_eligible": False,
            "qualification_record": str(qualification_path),
            "cycles": int(cycles[0]),
            "instructions": int(instructions[0]),
            "gpu_parents": accepted,
            "ramulator2_instances": 1,
            "outstanding": 0,
        },
        "tensors": list(source.payload["tensors"]),
        "files": files,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    promoted = OperatorArtifactManifest.load(output)
    if (
        not promoted.compute_memory_coupled
        or promoted.request_cycle_ready is not range_rebased
    ):
        raise ValueError("promoted Artifact has the wrong readiness state")
    print(
        json.dumps(
            {
                "artifact_id": promoted.artifact_id,
                "compute_memory_coupled": promoted.compute_memory_coupled,
                "global_pa_binding_ready": range_rebased,
                "request_cycle_ready": promoted.request_cycle_ready,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
