from copy import deepcopy

import pytest

from frontend.hetero.bandwidth import BandwidthContract, BandwidthContractError


def _valid_contract() -> dict[str, object]:
    return {
        "schema_version": "hetero-bandwidth-contract/v1",
        "external_link": {
            "protocol": "direct_memory_phy",
            "request_payload_bandwidth_Bps": 12_800_000_000,
            "response_payload_bandwidth_Bps": 12_800_000_000,
            "request_header_bytes": 16,
            "response_header_bytes": 16,
            "flit_bytes": 32,
            "propagation_latency_fs": 10_000_000,
            "queue_depth_transactions": 64,
            "credits": 64,
            "duplex_mode": "full_duplex",
            "clock_hz": 400_000_000,
        },
        "logic_die_gateway": {
            "clock_hz": 400_000_000,
            "ingress_queue_depth": 128,
            "parent_table_entries": 256,
            "split_width_per_cycle": 4,
            "issue_width_per_cycle": 4,
            "completion_width_per_cycle": 4,
            "ordering_policy": "ordering_domain_fifo",
            "write_ack_policy": "durable",
        },
        "internal_dram": {
            "implementation": "HBDRAM",
            "channel_count": 16,
            "pseudochannels_per_channel": 1,
            "dq_bits_per_channel": 512,
            "channel_width_bits": 512,
            "transfers_per_clock": 1,
            "rate_MTps": 400,
            "nBL_cycles": 1,
            "tCK_ps": 2500,
            "internal_prefetch_size": 1,
            "transaction_bytes": 64,
            "peak_payload_bandwidth_Bps": 409_600_000_000,
        },
    }


def test_edge_hbdram_contract_closes_exactly() -> None:
    contract = BandwidthContract.load(_valid_contract())
    dram = contract.internal_dram
    assert int(dram.derived_transaction_bytes) == 64
    assert int(dram.phy_peak_bandwidth_Bps) == 409_600_000_000
    assert int(dram.command_peak_bandwidth_Bps) == 409_600_000_000
    assert contract.external_link.request_payload_bandwidth_Bps == 12_800_000_000


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("transaction_bytes", 32, "transaction_bytes"),
        ("tCK_ps", 2000, "tCK_ps"),
        ("peak_payload_bandwidth_Bps", 204_800_000_000, "declared peak"),
        ("nBL_cycles", 4, "command peak"),
    ],
)
def test_internal_dram_contradictions_are_rejected(
    field: str, value: int, message: str
) -> None:
    payload = deepcopy(_valid_contract())
    payload["internal_dram"][field] = value  # type: ignore[index]
    with pytest.raises(BandwidthContractError, match=message):
        BandwidthContract.load(payload)


def test_link_credit_overcommit_is_rejected() -> None:
    payload = deepcopy(_valid_contract())
    payload["external_link"]["credits"] = 65  # type: ignore[index]
    with pytest.raises(BandwidthContractError, match="credits"):
        BandwidthContract.load(payload)
