#!/usr/bin/env python3
"""Validate deterministic P29 framework replay evidence on both GPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HOSTS = ("local-sm86", "remote-sm89")
PROFILES = ("huggingface", "vllm", "tensorrt_llm")
EXPECTED_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def at(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        require(isinstance(value, dict) and key in value, f"missing field: {dotted_path}")
        value = value[key]
    return value


def compare_fields(
    first: dict[str, Any], replay: dict[str, Any], fields: tuple[str, ...], label: str
) -> None:
    for field in fields:
        require(
            at(first, field) == at(replay, field),
            f"replay mismatch for {label}: {field}",
        )


def validate_profile(
    evidence_root: Path, host: str, profile: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    first_path = evidence_root / host / f"{profile}_live_smoke.json"
    replay_path = evidence_root / "replay" / host / f"{profile}_live_smoke.json"
    require(first_path.is_file(), f"missing first-run evidence: {first_path}")
    require(replay_path.is_file(), f"missing replay evidence: {replay_path}")
    first = read_json(first_path)
    replay = read_json(replay_path)
    label = f"{host}/{profile}"

    common_fields = (
        "schema_version",
        "framework",
        "framework_version",
        "model",
        "requested_revision",
        "python",
        "torch_version",
        "compiled_cuda",
        "device",
        "workload",
        "qualification",
    )
    compare_fields(first, replay, common_fields, label)
    require(
        at(first, "requested_revision") == EXPECTED_REVISION,
        f"unexpected model revision: {label}",
    )
    require(
        at(first, "qualification.performance_claim_allowed") is False,
        f"replay must not enable a performance claim: {label}",
    )

    if profile == "huggingface":
        stable_fields = (
            "resolved_revision",
            "dtype",
            "execution.cuda_peak_allocated_bytes",
            "execution.input_ids_sha256",
            "execution.prefill_logits_shape",
            "execution.prefill_last_logits_sha256",
            "execution.next_token_id",
            "execution.next_token_text",
            "execution.decode_logits_shape",
            "execution.decode_last_logits_sha256",
        )
        timing_fields = (
            "execution.model_load_seconds",
            "execution.prefill_cuda_ms",
            "execution.decode_cuda_ms",
        )
        semantic_output = {
            "next_token_id": at(first, "execution.next_token_id"),
            "next_token_text": at(first, "execution.next_token_text"),
            "prefill_last_logits_sha256": at(
                first, "execution.prefill_last_logits_sha256"
            ),
            "decode_last_logits_sha256": at(first, "execution.decode_last_logits_sha256"),
        }
    else:
        stable_fields = (
            "execution.token_ids",
            "execution.token_ids_sha256",
            "execution.text",
        )
        if profile == "vllm":
            stable_fields += ("model_runner",)
        else:
            stable_fields += ("execution.finished",)
        timing_fields = (
            "execution.engine_load_seconds",
            "execution.generation_seconds",
        )
        semantic_output = {
            "token_ids": at(first, "execution.token_ids"),
            "token_ids_sha256": at(first, "execution.token_ids_sha256"),
            "text": at(first, "execution.text"),
        }

    compare_fields(first, replay, stable_fields, label)
    first_timings = {field: at(first, field) for field in timing_fields}
    replay_timings = {field: at(replay, field) for field in timing_fields}
    result = {
        "deterministic_replay_passed": True,
        "semantic_output": semantic_output,
        "timing_fields_excluded_from_pass_criterion": True,
        "first_run_timings": first_timings,
        "replay_timings": replay_timings,
        "evidence_sha256": {
            "first_run": sha256(first_path),
            "replay": sha256(replay_path),
        },
    }
    return first, replay, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("validation/p29/framework_runtimes"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    first_runs: dict[str, dict[str, dict[str, Any]]] = {}
    replays: dict[str, dict[str, dict[str, Any]]] = {}
    host_results: dict[str, Any] = {}
    for host in HOSTS:
        first_runs[host] = {}
        replays[host] = {}
        profile_results: dict[str, Any] = {}
        for profile in PROFILES:
            first, replay, result = validate_profile(args.evidence_root, host, profile)
            first_runs[host][profile] = first
            replays[host][profile] = replay
            profile_results[profile] = result

        for leg_name, leg in (
            ("first_run", first_runs[host]),
            ("replay", replays[host]),
        ):
            require(
                at(leg["vllm"], "execution.token_ids_sha256")
                == at(leg["tensorrt_llm"], "execution.token_ids_sha256"),
                f"vLLM/TRT-LLM output mismatch: {host}/{leg_name}",
            )
        host_results[host] = {
            "profiles": profile_results,
            "vllm_trtllm_token_parity_passed": True,
        }

    for leg_name, source in (("first_run", first_runs), ("replay", replays)):
        require(
            at(source["local-sm86"]["huggingface"], "execution.input_ids_sha256")
            == at(source["remote-sm89"]["huggingface"], "execution.input_ids_sha256"),
            f"cross-host input mismatch: {leg_name}",
        )
        require(
            at(source["local-sm86"]["huggingface"], "execution.next_token_id")
            == at(source["remote-sm89"]["huggingface"], "execution.next_token_id"),
            f"cross-host HF semantic output mismatch: {leg_name}",
        )
        for profile in ("vllm", "tensorrt_llm"):
            require(
                at(source["local-sm86"][profile], "execution.token_ids_sha256")
                == at(source["remote-sm89"][profile], "execution.token_ids_sha256"),
                f"cross-host token output mismatch: {profile}/{leg_name}",
            )

    payload = {
        "schema_version": "hetero-framework-live-replay-qualification/v1",
        "model_revision": EXPECTED_REVISION,
        "hosts": host_results,
        "replay_passed": True,
        "same_host_stable_result_hashes_passed": True,
        "cross_host_semantic_output_passed": True,
        "cross_gpu_logits_bitwise_equality_required": False,
        "timing_fields_excluded_from_pass_criterion": True,
        "runtime_to_p28_online_adapter_connected": False,
        "automatic_framework_trace_capture_qualified": False,
        "end_to_end_simulator_integration_qualified": False,
        "performance_claim_allowed": False,
        "claim_boundary": (
            "This record qualifies a second real framework request and deterministic "
            "same-host functional replay. Timings are observational only. Cross-SM "
            "logit hashes are not required to match, and no online adapter, automatic "
            "trace capture, simulator coupling, or performance claim is qualified."
        ),
    }
    output = args.output or args.evidence_root / "replay_qualification_record.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
