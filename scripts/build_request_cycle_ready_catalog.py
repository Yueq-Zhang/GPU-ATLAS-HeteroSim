#!/usr/bin/env python3
"""Build a strict catalog and qualification map from ready operator Artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontend.hetero.operator_artifact import OperatorArtifactManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--qualification-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--existing-artifact", action="append", default=[])
    parser.add_argument("--existing-qualification", action="append", default=[])
    parser.add_argument("operators", nargs="*")
    args = parser.parse_args()

    artifacts: list[tuple[str, Path]] = []
    qualification_records: dict[str, str] = {}
    existing_artifacts: dict[str, Path] = {}
    existing_qualifications: dict[str, Path] = {}
    for values, destination in (
        (args.existing_artifact, existing_artifacts),
        (args.existing_qualification, existing_qualifications),
    ):
        for item in values:
            operator, separator, path_value = item.partition("=")
            if not separator or not operator or not path_value:
                raise ValueError("existing records must use OPERATOR=PATH")
            destination[operator] = Path(path_value).resolve()
    if set(existing_artifacts) != set(existing_qualifications):
        raise ValueError("existing Artifact and qualification operators must match")

    candidates: list[tuple[str, Path, Path]] = [
        (operator, path, existing_qualifications[operator])
        for operator, path in existing_artifacts.items()
    ]
    for operator in args.operators:
        path = (
            args.artifact_dir
            / (
                f"tinyllama_prefill_bs1_ctx16_{operator}_sm86_"
                "shared_hbdram_range_rebase.json"
            )
        ).resolve()
        record = (
            args.qualification_root
            / f"{operator.replace('_', '-')}-range-rebase"
            / "qualification_record.json"
        ).resolve()
        candidates.append((operator, path, record))

    if not candidates:
        raise ValueError("at least one Artifact is required")

    for operator, path, record in candidates:
        artifact = OperatorArtifactManifest.load(path)
        if (
            artifact.compatibility_key.operator != operator
            or not artifact.request_cycle_ready
        ):
            raise ValueError(f"Artifact is not request-cycle ready: {operator}")
        payload = json.loads(record.read_text(encoding="utf-8"))
        if payload.get("status") != "passed" or payload.get("trace_id") not in (
            artifact.artifact_id,
            artifact.artifact_id.removesuffix(".shared_hbdram_range_rebase_v1"),
        ):
            raise ValueError(f"qualification does not match {operator}")
        artifacts.append((operator, path))
        qualification_records[operator] = str(record)

    catalog_path = args.output_prefix.with_name(
        args.output_prefix.name + "_catalog.json"
    )
    qualification_path = args.output_prefix.with_name(
        args.output_prefix.name + "_qualification_map.json"
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = {
        "schema_version": "hetero-operator-artifact-catalog/v1",
        "required_operators": [operator for operator, _ in artifacts],
        "zero_fallback_required": True,
        "artifacts": [str(path) for _, path in artifacts],
    }
    qualification_map = {
        "schema_version": "hetero-coupled-qualification-map/v1",
        "records": qualification_records,
    }
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    qualification_path.write_text(
        json.dumps(qualification_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(catalog_path)
    print(qualification_path)


if __name__ == "__main__":
    main()
