#!/usr/bin/env python3
"""Capture the exact q_len=1 TinyLlama RoPE without an SM89-only BMM helper.

Transformers expresses the outer product between ``inv_freq`` and the single
Decode position as a matrix multiplication.  On the RTX 4090 software stack
used for P21 that one helper is JIT selected as SM89 while the remaining
PyTorch kernels execute their packaged SM86 cubins.  The elementwise expression
below is mathematically identical for q_len=1 and keeps the complete captured
kernel sequence on one replayable SM86 ISA target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from workloads.python import tinyllama_decode_operator as base


_ORIGINAL_DECODE_TARGET = base._decode_target
_ORIGINAL_PROGRAM_IDENTITY = base._launch_program_identity


def _decode_target(
    name: str, model: Any, kv_length: int, batch_size: int
) -> base.Target:
    if name != "rope":
        return _ORIGINAL_DECODE_TARGET(name, model, kv_length, batch_size)

    head_dim = int(model.config.head_dim)
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    query = base._host_random((batch_size, num_heads, 1, head_dim), 21)
    key = base._host_random((batch_size, num_kv_heads, 1, head_dim), 22)
    position_ids = torch.full(
        (batch_size, 1), kv_length - 1, dtype=torch.long, device="cuda"
    )
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq
    attention_scaling = rotary.attention_scaling

    def run_rope() -> dict[str, torch.Tensor]:
        position = position_ids[:, :, None].float()
        frequency = inv_freq[None, None, :].float()
        frequencies = position * frequency
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = (embedding.cos() * attention_scaling).to(dtype=query.dtype)
        sine = (embedding.sin() * attention_scaling).to(dtype=query.dtype)
        query_out, key_out = base.apply_rotary_pos_emb(query, key, cosine, sine)
        return {
            "cos": cosine,
            "sin": sine,
            "query_output": query_out,
            "key_output": key_out,
        }

    return base.Target(
        run_rope,
        {"query": query, "key": key, "position_ids": position_ids},
        {"inv_freq": inv_freq},
        "tinyllama.rope_decode_q1_elementwise_outer_product_sm86_replay",
    )


def _launch_program_identity() -> tuple[str, dict[str, object]]:
    _, components = _ORIGINAL_PROGRAM_IDENTITY()
    adapter = Path(__file__).resolve()
    components = dict(components)
    components.update(
        {
            "adapter_source": str(adapter),
            "adapter_source_sha256": base._file_sha256(adapter),
            "entrypoint": str(Path(sys.argv[0]).resolve()),
        }
    )
    return base._canonical_sha256(components), components


base._decode_target = _decode_target
base._launch_program_identity = _launch_program_identity


if __name__ == "__main__":
    base.main()
