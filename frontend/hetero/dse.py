"""Deterministic outer-loop design-space exploration."""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .runner import execute_run, simulation_input_key
from .schema import validate_config


class DseError(ValueError):
    pass


def _set_path(config: dict[str, Any], dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise DseError(f"invalid DSE path: {dotted_path}")
    cursor: Any = config
    for part in parts[:-1]:
        if isinstance(cursor, dict):
            if part not in cursor:
                raise DseError(f"DSE path does not exist: {dotted_path}")
            cursor = cursor[part]
        elif isinstance(cursor, list) and part.isdigit():
            index = int(part)
            if index >= len(cursor):
                raise DseError(f"DSE list index is out of range: {dotted_path}")
            cursor = cursor[index]
        else:
            raise DseError(f"DSE path does not resolve: {dotted_path}")
    final = parts[-1]
    if isinstance(cursor, dict):
        if final not in cursor:
            raise DseError(f"DSE path does not exist: {dotted_path}")
        cursor[final] = value
    elif isinstance(cursor, list) and final.isdigit():
        index = int(final)
        if index >= len(cursor):
            raise DseError(f"DSE list index is out of range: {dotted_path}")
        cursor[index] = value
    else:
        raise DseError(f"DSE path does not resolve: {dotted_path}")


def enumerate_candidates(
    base_config: Mapping[str, object], search: Mapping[str, object]
) -> list[dict[str, object]]:
    axes = search.get("axes")
    if not isinstance(axes, Mapping) or not axes:
        raise DseError("DSE axes must be a non-empty object")
    names = sorted(str(name) for name in axes)
    values: list[list[object]] = []
    for name in names:
        candidates = axes[name]
        if not isinstance(candidates, list) or not candidates:
            raise DseError(f"DSE axis {name} must be a non-empty array")
        values.append(candidates)
    combinations = 1
    for items in values:
        combinations *= len(items)
    limit = int(search.get("max_candidates", 256))
    if limit <= 0 or combinations > limit:
        raise DseError(
            f"DSE candidate count {combinations} exceeds max_candidates={limit}"
        )
    result: list[dict[str, object]] = []
    for index, combination in enumerate(itertools.product(*values)):
        candidate = deepcopy(dict(base_config))
        for name, value in zip(names, combination):
            _set_path(candidate, name, value)
        experiment = dict(candidate["experiment"])  # type: ignore[arg-type]
        experiment["name"] = f"{experiment['name']}.dse{index:04d}"
        candidate["experiment"] = experiment
        validate_config(candidate)
        result.append(candidate)
    return result


def run_dse(
    base_config: Mapping[str, object],
    search: Mapping[str, object],
    project_root: Path,
    output_root: Path,
) -> Path:
    candidates = enumerate_candidates(base_config, search)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        run_dir = execute_run(dict(candidate), project_root, output_root / "runs")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        records.append(
            {
                "candidate_id": index,
                "simulation_input_key": simulation_input_key(candidate),
                "run_dir": str(run_dir),
                "profile": candidate["system"]["profile"],  # type: ignore[index]
                "makespan_fs": metrics.get("makespan_fs"),
                "requests": metrics.get("requests", []),
                "fidelity": metrics.get("fidelity", {}),
                "performance_claim_allowed": metrics.get(
                    "performance_claim_allowed", False
                ),
            }
        )
    objective = str(search.get("objective", "makespan_fs"))
    ranked = sorted(
        records,
        key=lambda item: (
            item.get(objective) is None,
            item.get(objective) if item.get(objective) is not None else 0,
            int(item["candidate_id"]),
        ),
    )
    report = {
        "schema_version": "hetero-dse-report/v1",
        "objective": objective,
        "candidate_count": len(records),
        "ranking": ranked,
        "qualification_status": "unqualified_until_target_backend_validation",
    }
    report_path = output_root / "dse_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path
