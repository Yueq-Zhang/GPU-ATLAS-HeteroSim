#!/usr/bin/env python3
"""Build P17 trace/native execution-program identities from sealed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hetero.gpu_execution_identity import (  # noqa: E402
    EXECUTION_IDENTITY_CATALOG_SCHEMA,
    EXECUTION_IDENTITY_SCHEMA,
    canonical_sha256,
    trace_kernel_sequence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _repository_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recapture-record",
        type=Path,
        default=ROOT / "validation/p17/sm86_sealed_recapture/recapture_record.json",
    )
    parser.add_argument(
        "--sealed-manifest-root",
        type=Path,
        default=ROOT / "configs/hetero/operator_artifacts/p17_sealed",
    )
    parser.add_argument(
        "--portable-evidence-root",
        type=Path,
        default=ROOT / "configs/hetero/operator_artifacts/p16/evidence",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    record_path = args.recapture_record.resolve()
    recapture = _json(record_path)
    target = recapture.get("target_binary")
    operators = recapture.get("operators")
    if not isinstance(target, Mapping) or not isinstance(operators, Mapping):
        raise ValueError("recapture record is missing target_binary or operators")
    native_measurements = recapture.get("native_measurements")
    if native_measurements is not None and not isinstance(native_measurements, Mapping):
        raise ValueError("recapture native_measurements must be an object")
    root_binary_sha256 = str(target.get("sha256", ""))
    if len(root_binary_sha256) != 64:
        raise ValueError("recapture target binary SHA-256 is invalid")

    identities: list[dict[str, object]] = []
    for operator in sorted(operators):
        evidence = operators[operator]
        if not isinstance(evidence, Mapping):
            raise ValueError(f"operator evidence must be an object: {operator}")
        operator_target = evidence.get("target_binary", target)
        if not isinstance(operator_target, Mapping):
            raise ValueError(f"operator target_binary must be an object: {operator}")
        binary_sha256 = str(operator_target.get("sha256", ""))
        if len(binary_sha256) != 64:
            raise ValueError(f"operator executable SHA-256 is invalid: {operator}")
        manifest_path = (
            args.sealed_manifest_root.resolve()
            / f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_trace.json"
        )
        portable = args.portable_evidence_root.resolve() / operator
        metadata = portable / "operator_metadata.json"
        kernels = portable / "traces/kernelslist.g"
        trace_files = [
            (kernels.parent / line.strip()).resolve()
            for line in kernels.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not trace_files:
            raise ValueError(f"expected at least one sealed trace for {operator}")
        trace_evidence = [
            {
                "name": trace.relative_to(kernels.parent).as_posix(),
                "sha256": _sha256(trace),
            }
            for trace in trace_files
        ]
        expected = {
            "metadata_sha256": _sha256(metadata),
            "kernels_list_sha256": _sha256(kernels),
            "trace_manifest_sha256": _sha256(manifest_path),
        }
        for field, actual in expected.items():
            if evidence.get(field) != actual:
                raise ValueError(f"sealed recapture mismatch: {operator}.{field}")
        recorded_trace_evidence = evidence.get("trace_files")
        if recorded_trace_evidence is not None:
            if recorded_trace_evidence != trace_evidence:
                raise ValueError(f"sealed recapture trace-set mismatch: {operator}")
        elif len(trace_evidence) == 1:
            if evidence.get("trace_sha256") != trace_evidence[0]["sha256"]:
                raise ValueError(f"sealed recapture mismatch: {operator}.trace_sha256")
        else:
            raise ValueError(f"multi-Trace recapture set is absent: {operator}")
        manifest = _json(manifest_path)
        compilation = manifest.get("compilation")
        capture = manifest.get("capture")
        if not isinstance(compilation, Mapping) or not isinstance(capture, Mapping):
            raise ValueError(f"sealed manifest contract is incomplete: {operator}")
        if operator_target.get("kind") == "python_pytorch_launch_program":
            captured_program = compilation.get("launch_program")
            if (
                not isinstance(captured_program, Mapping)
                or captured_program.get("kind")
                != "python_pytorch_launch_program"
                or captured_program.get("sha256") != binary_sha256
                or captured_program.get("components")
                != operator_target.get("components")
            ):
                raise ValueError(f"captured launch-program mismatch for {operator}")
        sequence_sha256, launches = trace_kernel_sequence(kernels)
        launch_contract = {
            "operator_type": operator,
            "trace_id": manifest.get("trace_id"),
            "implementation": compilation.get("implementation"),
            "target_sm": compilation.get("target_sm"),
            "shape": capture.get("shape"),
            "dtype": capture.get("dtype"),
            "kernel_launches": launches,
        }
        native_observed = False
        native_provenance: dict[str, object] = {}
        if isinstance(native_measurements, Mapping):
            native_evidence = native_measurements.get(operator)
            if not isinstance(native_evidence, Mapping):
                raise ValueError(f"native measurement is absent for {operator}")
            raw_path = Path(str(native_evidence.get("path", "")))
            if not raw_path.is_absolute():
                raw_path = (ROOT / raw_path).resolve()
            expected_native_sha256 = str(native_evidence.get("sha256", ""))
            if not raw_path.is_file() or _sha256(raw_path) != expected_native_sha256:
                raise ValueError(f"native measurement hash mismatch for {operator}")
            native_payload = _json(raw_path)
            native_device = native_payload.get("device")
            native_protocol = native_payload.get("protocol")
            native_launch = native_payload.get("launch")
            native_schema = native_payload.get("schema_version")
            if (
                native_schema
                not in {
                    "hetero-p17-sealed-native-operator/v1",
                    "hetero-p17-sealed-native-pytorch-operator/v1",
                }
                or native_payload.get("operator_type") != operator
                or native_payload.get("checkpoint_revision")
                != capture.get("model_revision")
                or native_payload.get("batch_size") != capture["shape"]["batch_size"]
                or native_payload.get("context_length")
                != capture["shape"]["context_length"]
                or native_payload.get("dtype") != capture.get("dtype")
                or not isinstance(native_device, Mapping)
                or native_device.get("name") != "NVIDIA GeForce RTX 3070"
                or native_device.get("compute_capability") != "8.6"
                or not isinstance(native_protocol, Mapping)
                or int(native_protocol.get("warmup_iterations", -1)) < 50
                or int(native_protocol.get("measured_iterations", -1)) < 500
                or native_protocol.get("timer") != "cuda_event_per_iteration"
                or not isinstance(native_launch, Mapping)
                or native_launch.get("target_sm") != 86
                or native_payload.get("measurement_scope")
                != "native_rtx3070_local_vram"
            ):
                raise ValueError(f"native measurement contract mismatch for {operator}")
            if native_payload.get("implementation") not in {
                None,
                compilation.get("implementation"),
            }:
                raise ValueError(f"native implementation mismatch for {operator}")
            if native_schema == "hetero-p17-sealed-native-pytorch-operator/v1":
                if (
                    native_launch.get("program_kind")
                    != "python_pytorch_launch_program"
                    or native_launch.get("launch_program_sha256")
                    != binary_sha256
                    or operator_target.get("kind")
                    != "python_pytorch_launch_program"
                ):
                    raise ValueError(f"native launch-program mismatch for {operator}")
            native_observed = True
            native_provenance = {
                "native_measurement": _repository_path(raw_path),
                "native_measurement_sha256": expected_native_sha256,
                "native_warmup_iterations": native_protocol["warmup_iterations"],
                "native_measured_iterations": native_protocol[
                    "measured_iterations"
                ],
            }
        identities.append(
            {
                "operator_type": operator,
                "execution_identity": {
                    "schema_version": EXECUTION_IDENTITY_SCHEMA,
                    "executable_sha256": binary_sha256,
                    "launch_contract_sha256": canonical_sha256(launch_contract),
                    "kernel_sequence_sha256": sequence_sha256,
                    "target_sm": int(compilation["target_sm"]),
                    "kernel_launch_count": len(launches),
                    "native_measurement_observed": native_observed,
                    "trace_capture_observed": True,
                    "provenance": {
                        "recapture_record": _repository_path(record_path),
                        "recapture_record_sha256": _sha256(record_path),
                        "trace_manifest": _repository_path(manifest_path),
                        "trace_manifest_sha256": expected[
                            "trace_manifest_sha256"
                        ],
                        "kernels_list_sha256": expected["kernels_list_sha256"],
                        "trace_files": trace_evidence,
                        "target_kind": operator_target.get("kind", "elf_binary"),
                        "target_components": operator_target.get("components"),
                        **native_provenance,
                    },
                },
            }
        )
    same_binary_complete = all(
        bool(item["execution_identity"]["native_measurement_observed"])
        for item in identities
    )
    payload = {
        "schema_version": EXECUTION_IDENTITY_CATALOG_SCHEMA,
        "catalog_id": "p17.sm86.sealed.same_binary.trace_execution_identity",
        "operators": identities,
        "operator_count": len(identities),
        "claim_boundary": {
            "same_binary_native_pairing_complete": same_binary_complete,
            "reason": (
                "sealed Trace and native RTX 3070 measurements use the same "
                "executable"
                if same_binary_complete
                else "sealed Trace identity exists, but the native RTX 3070 "
                "measurement has not yet been produced by this exact executable"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"P17 execution identity catalog written: {args.output} "
        f"({len(identities)} trace-observed operators)"
    )


if __name__ == "__main__":
    main()
