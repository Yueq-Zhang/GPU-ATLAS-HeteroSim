#!/usr/bin/env python3
"""Validate the two-host P29 framework installation and live smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "local-sm86": {
        "name": "NVIDIA GeForce RTX 3070",
        "compute_capability": [8, 6],
    },
    "remote-sm89": {
        "name": "NVIDIA GeForce RTX 4090",
        "compute_capability": [8, 9],
    },
}
EXPECTED_VERSIONS = {
    "huggingface": "5.16.1",
    "vllm": "0.29.0",
    "tensorrt_llm": "1.2.1",
}
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


def validate_host(root: Path, host: str) -> dict[str, Any]:
    expected = EXPECTED[host]
    host_root = root / host
    probes: dict[str, Any] = {}
    smokes: dict[str, Any] = {}
    evidence: dict[str, str] = {}

    for profile in EXPECTED_VERSIONS:
        probe_path = host_root / f"{profile}_probe.json"
        smoke_path = host_root / f"{profile}_live_smoke.json"
        lock_path = host_root / f"{profile}_requirements.lock"
        for path in (probe_path, smoke_path, lock_path):
            require(path.is_file(), f"missing P29 evidence: {path}")
            evidence[str(path.relative_to(root))] = sha256(path)

        probe = read_json(probe_path)
        smoke = read_json(smoke_path)
        probes[profile] = probe
        smokes[profile] = smoke
        require(probe["torch"]["cuda_available"], f"CUDA unavailable: {host}/{profile}")
        require(
            probe["torch"]["device_name"] == expected["name"],
            f"device identity mismatch: {host}/{profile}",
        )
        require(
            probe["torch"]["compute_capability"]
            == expected["compute_capability"],
            f"compute capability mismatch: {host}/{profile}",
        )
        require(
            smoke["device"]["name"] == expected["name"],
            f"smoke device mismatch: {host}/{profile}",
        )
        require(
            smoke["requested_revision"] == EXPECTED_REVISION,
            f"model revision mismatch: {host}/{profile}",
        )
        require(
            smoke["qualification"]["performance_claim_allowed"] is False,
            f"P29 must not enable a performance claim: {host}/{profile}",
        )

    require(
        probes["huggingface"]["packages"]["transformers"]
        == EXPECTED_VERSIONS["huggingface"],
        f"Transformers version mismatch on {host}",
    )
    require(
        probes["vllm"]["packages"]["vllm"] == EXPECTED_VERSIONS["vllm"],
        f"vLLM version mismatch on {host}",
    )
    require(
        probes["tensorrt_llm"]["packages"]["tensorrt_llm"]
        == EXPECTED_VERSIONS["tensorrt_llm"],
        f"TensorRT-LLM version mismatch on {host}",
    )
    hf_qualification = smokes["huggingface"]["qualification"]
    require(hf_qualification["real_model_loaded"], f"HF load failed on {host}")
    require(hf_qualification["real_prefill_executed"], f"HF prefill failed on {host}")
    require(hf_qualification["real_decode_executed"], f"HF decode failed on {host}")
    require(
        hf_qualification["runtime_to_p28_event_adapter_connected"] is False,
        f"unexpected HF adapter claim on {host}",
    )
    vllm_qualification = smokes["vllm"]["qualification"]
    require(vllm_qualification["real_engine_loaded"], f"vLLM load failed on {host}")
    require(vllm_qualification["real_request_executed"], f"vLLM request failed on {host}")
    require(
        vllm_qualification["scheduler_event_adapter_connected"] is False,
        f"unexpected vLLM scheduler-adapter claim on {host}",
    )
    trt_qualification = smokes["tensorrt_llm"]["qualification"]
    require(trt_qualification["real_engine_loaded"], f"TRT-LLM load failed on {host}")
    require(trt_qualification["real_request_executed"], f"TRT-LLM request failed on {host}")
    require(
        trt_qualification["runtime_event_adapter_connected"] is False,
        f"unexpected TRT-LLM adapter claim on {host}",
    )
    require(
        smokes["vllm"]["execution"]["token_ids_sha256"]
        == smokes["tensorrt_llm"]["execution"]["token_ids_sha256"],
        f"vLLM/TRT-LLM output token mismatch on {host}",
    )
    return {
        "device": expected,
        "profiles": {
            "huggingface": {
                "version": EXPECTED_VERSIONS["huggingface"],
                "cuda_probe_passed": True,
                "real_prefill_decode_passed": True,
            },
            "vllm": {
                "version": EXPECTED_VERSIONS["vllm"],
                "cuda_probe_passed": True,
                "real_request_passed": True,
                "model_runner": smokes["vllm"]["model_runner"],
            },
            "tensorrt_llm": {
                "version": EXPECTED_VERSIONS["tensorrt_llm"],
                "cuda_probe_passed": True,
                "real_request_passed": True,
            },
        },
        "evidence_sha256": dict(sorted(evidence.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("validation/p29/framework_runtimes"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hosts = {
        host: validate_host(args.evidence_root, host) for host in EXPECTED
    }
    payload = {
        "schema_version": "hetero-framework-runtime-installation/v1",
        "model_revision": EXPECTED_REVISION,
        "hosts": hosts,
        "installation_passed": True,
        "cuda_execution_passed": True,
        "live_framework_smoke_passed": True,
        "runtime_to_p28_online_adapter_connected": False,
        "automatic_framework_trace_capture_qualified": False,
        "end_to_end_simulator_integration_qualified": False,
        "performance_claim_allowed": False,
        "claim_boundary": (
            "P29 qualifies isolated installation and one real framework request per "
            "profile on each GPU. It does not qualify online P28 callbacks, automatic "
            "SASS capture, simulator coupling, or performance."
        ),
    }
    output = args.output or args.evidence_root / "installation_record.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
