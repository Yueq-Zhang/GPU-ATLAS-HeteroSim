#!/usr/bin/env python3
"""Prove the P21 q_len=1 elementwise RoPE matches Transformers exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from workloads.python import tinyllama_decode_rope_sm86_operator as adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kv-lengths", nargs="+", type=int, default=[17, 18, 19, 20])
    args = parser.parse_args()

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for kv_length in args.kv_lengths:
            reference = adapter._ORIGINAL_DECODE_TARGET(
                "rope", model, kv_length, 1
            ).run()
            candidate = adapter._decode_target("rope", model, kv_length, 1).run()
            torch.cuda.synchronize()
            max_abs_diff = {
                name: float(
                    (reference[name].float() - candidate[name].float())
                    .abs()
                    .max()
                    .item()
                )
                for name in reference
            }
            records.append(
                {
                    "kv_length": kv_length,
                    "all_equal": all(
                        torch.equal(reference[name], candidate[name])
                        for name in reference
                    ),
                    "max_abs_diff": max_abs_diff,
                }
            )

    passed = all(bool(record["all_equal"]) for record in records)
    payload = {
        "schema_version": "hetero-p21-rope-equivalence/v1",
        "status": "passed" if passed else "failed",
        "model": "TinyLlama-1.1B",
        "phase": "decode_step",
        "batch_size": 1,
        "context_length": 16,
        "q_len": 1,
        "kv_lengths": args.kv_lengths,
        "reference": "transformers.LlamaRotaryEmbedding_plus_apply_rotary_pos_emb",
        "candidate": "tinyllama.rope_decode_q1_elementwise_outer_product_sm86_replay",
        "records": records,
        "performance_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
