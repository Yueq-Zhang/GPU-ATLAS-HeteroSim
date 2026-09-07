from __future__ import annotations

import pytest

from scripts.seal_gpu_trace_binary_identity import _resolve_binary_contract


def test_mixed_binary_contract_remains_fail_closed_by_default() -> None:
    with pytest.raises(ValueError, match="mixed SASS"):
        _resolve_binary_contract(
            [80, 86],
            required_binary_sm=86,
            allowed_binary_sms=[],
            replay_target_sm=None,
        )


def test_explicit_ampere_mixed_binary_contract_records_replay_target() -> None:
    assert _resolve_binary_contract(
        [80, 86],
        required_binary_sm=None,
        allowed_binary_sms=[80, 86],
        replay_target_sm=86,
    ) == (86, "ampere_sm80_sm86_opcode_compatible")


def test_mixed_binary_contract_rejects_unqualified_architecture() -> None:
    with pytest.raises(ValueError, match="limited to SM80/SM86"):
        _resolve_binary_contract(
            [86, 89],
            required_binary_sm=None,
            allowed_binary_sms=[80, 86, 89],
            replay_target_sm=86,
        )
