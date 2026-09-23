#!/usr/bin/env python3
"""Compare Accel-Sim native GDDR6, Ramulator2 GDDR6, and RTX 3070 timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from frontend.hetero.backends.accel_sim import (
    AccelSimBackend,
    AccelSimBackendConfig,
)
from frontend.hetero.trace_manifest import TraceManifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = PROJECT_ROOT / (
    "configs/hetero/backends/"
    "gpu_accelsim_rtx3070_ramulator2_gddr6_16ch_range_rebase.json"
)
DEFAULT_CATALOG = (
    PROJECT_ROOT / "validation/p17/gpu_operator_pairing/simulator_native_vram.json"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT / "validation/p17/gpu_operator_pairing/native_vram_pairing_audit.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "validation/gddr6_model_comparison"
PROJECTION_OPERATOR_TYPES = frozenset(
    {
        "down_projection",
        "gate_up_projection",
        "lm_head",
        "output_projection",
        "qkv_projection",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _manifest_path(operator: dict[str, Any]) -> Path:
    provenance = operator["execution_identity"]["provenance"]
    relative = provenance.get("trace_manifest")
    if isinstance(relative, str):
        candidate = PROJECT_ROOT / relative
        if candidate.is_file():
            return candidate
    candidate = Path(operator["trace_manifest"])
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_or_run(
    backend: AccelSimBackend,
    manifest: TraceManifest,
    output: Path,
) -> dict[str, Any]:
    expected_key = backend.simulation_key(manifest)
    stats_path = output / "stats.json"
    if stats_path.is_file():
        previous = _load_json(stats_path)
        if previous.get("simulation_key") == expected_key:
            return previous
    result = backend.run(manifest, output)
    return _load_json(result.output_directory / "stats.json")


def _validate_external(stats: dict[str, Any]) -> None:
    external = stats.get("external_memory_stats")
    if not isinstance(external, dict):
        raise TypeError("Ramulator2 run is missing external-memory statistics")
    required_equal = (
        ("gpu_parents", "gpu_completed"),
        ("children_sent", "children_completed"),
    )
    for issued, completed in required_equal:
        if external.get(issued) != external.get(completed):
            raise ValueError(f"request conservation failed: {issued} != {completed}")
    required_zero = (
        "address_unmapped",
        "atlas_parents",
        "atlas_children",
        "atlas_completed",
        "outstanding",
    )
    for key in required_zero:
        if external.get(key) != 0:
            raise ValueError(f"expected {key}=0, observed {external.get(key)!r}")
    if external.get("instances") != 1:
        raise ValueError("Ramulator2 must be the unique DRAM timing owner")


def _relative(candidate: int, reference: int) -> float:
    return (candidate - reference) / reference


def _mean_absolute(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return fmean(abs(float(item[key])) for item in rows)


def _render_markdown(payload: dict[str, Any]) -> str:
    configuration = payload["configuration"]
    internal_transaction_bytes = configuration[
        "accel_sim_internal_transaction_bytes"
    ]
    ramulator2_transaction_bytes = configuration["ramulator2_transaction_bytes"]
    if internal_transaction_bytes == ramulator2_transaction_bytes:
        transaction_parity = (
            f"- Transaction granularity: {internal_transaction_bytes} B in both "
            "modeled memories."
        )
    else:
        transaction_parity = (
            f"- Known non-parity: the internal model transfers "
            f"{internal_transaction_bytes} B per GDDR burst, whereas Ramulator2 "
            f"reports a {ramulator2_transaction_bytes} B transaction. The bridge "
            "therefore splits each internal-model-sized request."
        )
    lines = [
        "# GDDR6 model comparison",
        "",
        (
            "| Operator | Native RTX 3070 (us) | Accel-Sim internal GDDR6 (us) | "
            "Ramulator2 GDDR6 (us) | Ramulator2 vs internal | Internal vs native | "
            "Ramulator2 vs native |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["operators"]:
        lines.append(
            "| {operator_type} | {native:.3f} | {internal:.3f} | {ramulator:.3f} | "
            "{ram_delta:+.2%} | {internal_delta:+.2%} | {ram_native_delta:+.2%} |".format(
                operator_type=item["operator_type"],
                native=item["native_latency_fs"] / 1e9,
                internal=item["accel_sim_internal_latency_fs"] / 1e9,
                ramulator=item["ramulator2_latency_fs"] / 1e9,
                ram_delta=item["ramulator2_vs_internal_relative_delta"],
                internal_delta=item["accel_sim_internal_vs_native_relative_delta"],
                ram_native_delta=item["ramulator2_vs_native_relative_delta"],
            )
        )
    summary = payload["summary"]
    classes = summary["class_breakdown"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Operators: {summary['operator_count']}",
            (
                "- Ramulator2 vs Accel-Sim internal GDDR6 mean absolute relative "
                f"difference: {summary['ramulator2_vs_internal_mard']:.2%}"
            ),
            (
                "- Accel-Sim internal GDDR6 vs native RTX 3070 MARE: "
                f"{summary['accel_sim_internal_vs_native_mare']:.2%}"
            ),
            (
            "- Ramulator2 GDDR6 vs native RTX 3070 MARE: "
                f"{summary['ramulator2_vs_native_mare']:.2%}"
            ),
            (
                "- Ramulator2 vs internal median absolute relative difference: "
                f"{summary['ramulator2_vs_internal_median_ard']:.2%}"
            ),
            (
                "- Within 15% (Ramulator2 vs internal / internal vs native / "
                "Ramulator2 vs native): "
                f"{summary['ramulator2_vs_internal_within_15pct']}/"
                f"{summary['accel_sim_internal_vs_native_within_15pct']}/"
                f"{summary['ramulator2_vs_native_within_15pct']}"
            ),
            (
                "- Largest Ramulator2-vs-internal difference: "
                f"{summary['ramulator2_vs_internal_max_operator']} "
                f"({summary['ramulator2_vs_internal_max_abs_delta']:.2%})"
            ),
            (
                "- Projection operators "
                f"({classes['projection']['operator_count']}): Ramulator2 vs internal "
                f"{classes['projection']['ramulator2_vs_internal_mard']:.2%}; "
                "internal vs native "
                f"{classes['projection']['accel_sim_internal_vs_native_mare']:.2%}."
            ),
            (
                "- Other operators "
                f"({classes['other']['operator_count']}): Ramulator2 vs internal "
                f"{classes['other']['ramulator2_vs_internal_mard']:.2%}; "
                "internal vs native "
                f"{classes['other']['accel_sim_internal_vs_native_mare']:.2%}."
            ),
            (
                "- Every Ramulator2 point is a deterministic double run with one "
                "timing owner, conserved parent/child requests, zero unmapped "
                "addresses, and zero outstanding requests."
            ),
            "",
            "## Configuration parity",
            "",
            (
                "- Unique address capacity: "
                f"{configuration['accel_sim_internal_capacity_bytes'] / 1024**3:.0f} "
                "GiB (internal) versus "
                f"{configuration['ramulator2_capacity_bytes'] / 1024**3:.0f} GiB "
                "(Ramulator2)."
            ),
            (
                f"- Ramulator2 organization: "
                f"{configuration['ramulator2_channels']} channels, "
                f"{configuration['ramulator2_dq_bits_per_channel']} data pins per "
                f"channel, {configuration['ramulator2_rate_MTps']} MT/s."
            ),
            (
                "- Peak bandwidth: "
                f"{configuration['accel_sim_internal_gddr6_peak_Bps'] / 1e9:.3f} "
                "GB/s (internal) versus "
                f"{configuration['ramulator2_gddr6_peak_Bps'] / 1e9:.3f} GB/s "
                "(Ramulator2), a "
                f"{configuration['peak_bandwidth_relative_delta']:+.3%} difference."
            ),
            (
                "- GPU-to-memory adapter: near-zero latency and effectively "
                "unlimited bandwidth; this is not a PCIe/CXL experiment."
            ),
            transaction_parity,
            "",
            "## Interpretation boundary",
            "",
            (
                "The GPU core, L1/L2, and GPU NoC are identical across the two "
                "simulated legs. The Ramulator2 leg replaces the DRAM controller, "
                "scheduler, timing state machine, transaction granularity, and "
                "address mapper. Its external link is configured as a near-zero-cost "
                "adapter, not as PCIe/CXL. Because Ramulator2 uses "
                "OneLevelInterleave while the RTX 3070 Accel-Sim profile uses its "
                "native partition-indexing policy, the measured delta is not a pure "
                "timing-only delta."
            ),
            "",
            (
                "The native column reuses the P17 CUDA-event median measurements. "
                "Those measurements synchronize each Python/PyTorch operator "
                "iteration and were not collected with locked clocks or a "
                "CUPTI-isolated kernel window. These results are diagnostic; they "
                "do not establish a performance-qualified model."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=DEFAULT_BACKEND)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--operators",
        help="Comma-separated subset; default runs every operator in the P17 catalog",
    )
    args = parser.parse_args()

    catalog = _load_json(args.catalog)
    audit = _load_json(args.audit)
    native_by_type = {item["operator_type"]: item for item in audit["operators"]}
    selected = None
    if args.operators:
        selected = {value.strip() for value in args.operators.split(",") if value.strip()}

    config = AccelSimBackendConfig.load(args.backend)
    backend = AccelSimBackend(config)
    external = config.external_memory
    if external is None or external.address_translation is None:
        raise ValueError("comparison backend requires translated external memory")
    dram = external.bandwidth_contract.internal_dram
    rows: list[dict[str, Any]] = []
    for operator in catalog["operators"]:
        operator_type = operator["operator_type"]
        if selected is not None and operator_type not in selected:
            continue
        manifest_path = _manifest_path(operator)
        manifest = TraceManifest.load(manifest_path)
        run1 = _load_or_run(
            backend, manifest, args.output / "runs" / operator_type / "run1"
        )
        run2 = _load_or_run(
            backend, manifest, args.output / "runs" / operator_type / "run2"
        )
        _validate_external(run1)
        _validate_external(run2)
        if (
            run1["cycles"] != run2["cycles"]
            or run1["instructions"] != run2["instructions"]
            or run1["external_memory_stats"] != run2["external_memory_stats"]
        ):
            raise ValueError(f"double-run signature mismatch: {operator_type}")

        native = native_by_type[operator_type]
        native_fs = int(native["measured_latency_fs"])
        internal_fs = int(operator["operator_latency_fs"])
        ramulator_fs = int(run1["duration_fs"])
        rows.append(
            {
                "operator_type": operator_type,
                "shape_key": operator["shape_key"],
                "trace_manifest": str(manifest_path),
                "trace_manifest_sha256": operator["trace_manifest_sha256"],
                "instructions": int(run1["instructions"]),
                "native_latency_fs": native_fs,
                "accel_sim_internal_cycles": int(operator["cycles"]),
                "accel_sim_internal_latency_fs": internal_fs,
                "ramulator2_cycles": int(run1["cycles"]),
                "ramulator2_latency_fs": ramulator_fs,
                "ramulator2_vs_internal_relative_delta": _relative(
                    ramulator_fs, internal_fs
                ),
                "accel_sim_internal_vs_native_relative_delta": _relative(
                    internal_fs, native_fs
                ),
                "ramulator2_vs_native_relative_delta": _relative(
                    ramulator_fs, native_fs
                ),
                "ramulator2_external_memory_stats": run1["external_memory_stats"],
                "double_run_signature_equal": True,
            }
        )

    if selected is not None:
        missing = selected - {item["operator_type"] for item in rows}
        if missing:
            raise ValueError(f"operators absent from catalog: {sorted(missing)}")
    if not rows:
        raise ValueError("no operators selected")

    max_ramulator2_delta = max(
        rows, key=lambda item: abs(item["ramulator2_vs_internal_relative_delta"])
    )
    projection_rows = [
        item for item in rows if item["operator_type"] in PROJECTION_OPERATOR_TYPES
    ]
    other_rows = [
        item for item in rows if item["operator_type"] not in PROJECTION_OPERATOR_TYPES
    ]
    payload = {
        "schema_version": "hetero-gddr6-model-comparison/v1",
        "comparison_scope": {
            "gpu": "RTX 3070 SM86",
            "workload": "TinyLlama-1.1B layer-0 prefill, BS=1, context=16, FP16",
            "core_frequency_hz": config.core_frequency_hz,
            "same_trace_and_gpu_core_cache_noc": True,
            "ramulator2_unique_timing_owner": True,
            "gddr6_model_parity_qualified": False,
            "ramulator2_address_mapper": "OneLevelInterleave",
            "accel_sim_internal_partition_indexing": 2,
            "pure_dram_timing_only_comparison": False,
            "performance_claim_allowed": False,
        },
        "configuration": {
            "backend": str(args.backend),
            "catalog": str(args.catalog),
            "native_audit": str(args.audit),
            "accel_sim_internal_gddr6_peak_Bps": 448_064_000_000,
            "ramulator2_gddr6_peak_Bps": dram.peak_payload_bandwidth_Bps,
            "peak_bandwidth_relative_delta": _relative(
                dram.peak_payload_bandwidth_Bps, 448_064_000_000
            ),
            "accel_sim_internal_capacity_bytes": 4 * 1024**3,
            "ramulator2_capacity_bytes": external.address_translation.capacity_bytes,
            "ramulator2_channels": dram.channel_count,
            "ramulator2_dq_bits_per_channel": dram.dq_bits_per_channel,
            "accel_sim_internal_transaction_bytes": 32,
            "ramulator2_transaction_bytes": dram.transaction_bytes,
            "ramulator2_rate_MTps": dram.rate_MTps,
        },
        "operators": rows,
        "summary": {
            "operator_count": len(rows),
            "ramulator2_vs_internal_mard": fmean(
                abs(item["ramulator2_vs_internal_relative_delta"]) for item in rows
            ),
            "ramulator2_vs_internal_median_ard": median(
                abs(item["ramulator2_vs_internal_relative_delta"]) for item in rows
            ),
            "ramulator2_vs_internal_within_15pct": sum(
                abs(item["ramulator2_vs_internal_relative_delta"]) <= 0.15
                for item in rows
            ),
            "ramulator2_vs_internal_max_operator": max_ramulator2_delta[
                "operator_type"
            ],
            "ramulator2_vs_internal_max_abs_delta": abs(
                max_ramulator2_delta["ramulator2_vs_internal_relative_delta"]
            ),
            "accel_sim_internal_vs_native_mare": fmean(
                abs(item["accel_sim_internal_vs_native_relative_delta"])
                for item in rows
            ),
            "accel_sim_internal_vs_native_within_15pct": sum(
                abs(item["accel_sim_internal_vs_native_relative_delta"]) <= 0.15
                for item in rows
            ),
            "ramulator2_vs_native_mare": fmean(
                abs(item["ramulator2_vs_native_relative_delta"]) for item in rows
            ),
            "ramulator2_vs_native_within_15pct": sum(
                abs(item["ramulator2_vs_native_relative_delta"]) <= 0.15
                for item in rows
            ),
            "class_breakdown": {
                "projection": {
                    "operator_count": len(projection_rows),
                    "ramulator2_vs_internal_mard": _mean_absolute(
                        projection_rows, "ramulator2_vs_internal_relative_delta"
                    ),
                    "accel_sim_internal_vs_native_mare": _mean_absolute(
                        projection_rows,
                        "accel_sim_internal_vs_native_relative_delta",
                    ),
                    "ramulator2_vs_native_mare": _mean_absolute(
                        projection_rows, "ramulator2_vs_native_relative_delta"
                    ),
                },
                "other": {
                    "operator_count": len(other_rows),
                    "ramulator2_vs_internal_mard": _mean_absolute(
                        other_rows, "ramulator2_vs_internal_relative_delta"
                    ),
                    "accel_sim_internal_vs_native_mare": _mean_absolute(
                        other_rows, "accel_sim_internal_vs_native_relative_delta"
                    ),
                    "ramulator2_vs_native_mare": _mean_absolute(
                        other_rows, "ramulator2_vs_native_relative_delta"
                    ),
                },
            },
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        _render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
