"""Exact external-link and internal-DRAM bandwidth contracts.

The contract deliberately keeps the GPU-facing link independent from the
Ramulator2 organization.  All arithmetic is integer/rational so a candidate
configuration either closes exactly or is rejected before simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping


class BandwidthContractError(ValueError):
    """Raised when a bandwidth contract is incomplete or contradictory."""


def _mapping(payload: object, path: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise BandwidthContractError(f"{path} must be an object")
    return payload


def _positive_int(payload: Mapping[str, object], key: str, path: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BandwidthContractError(f"{path}.{key} must be a positive integer")
    return value


def _unsigned_int(payload: Mapping[str, object], key: str, path: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BandwidthContractError(f"{path}.{key} must be an unsigned integer")
    return value


def _require_exact_keys(
    payload: Mapping[str, object], required: set[str], path: str
) -> None:
    missing = required - payload.keys()
    extra = payload.keys() - required
    if missing or extra:
        raise BandwidthContractError(
            f"{path} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


@dataclass(frozen=True, slots=True)
class ExternalLinkContract:
    protocol: str
    request_payload_bandwidth_Bps: int
    response_payload_bandwidth_Bps: int
    request_header_bytes: int
    response_header_bytes: int
    flit_bytes: int
    propagation_latency_fs: int
    queue_depth_transactions: int
    credits: int
    duplex_mode: str
    clock_hz: int

    @classmethod
    def load(cls, payload: object) -> "ExternalLinkContract":
        path = "bandwidth_contract.external_link"
        source = _mapping(payload, path)
        required = {
            "protocol",
            "request_payload_bandwidth_Bps",
            "response_payload_bandwidth_Bps",
            "request_header_bytes",
            "response_header_bytes",
            "flit_bytes",
            "propagation_latency_fs",
            "queue_depth_transactions",
            "credits",
            "duplex_mode",
            "clock_hz",
        }
        _require_exact_keys(source, required, path)
        protocol = source.get("protocol")
        duplex_mode = source.get("duplex_mode")
        if protocol not in {"direct_memory_phy", "pcie_dma", "cxl_mem"}:
            raise BandwidthContractError(f"{path}.protocol is invalid")
        if duplex_mode not in {"full_duplex", "half_duplex"}:
            raise BandwidthContractError(f"{path}.duplex_mode is invalid")
        result = cls(
            protocol=str(protocol),
            request_payload_bandwidth_Bps=_positive_int(
                source, "request_payload_bandwidth_Bps", path
            ),
            response_payload_bandwidth_Bps=_positive_int(
                source, "response_payload_bandwidth_Bps", path
            ),
            request_header_bytes=_unsigned_int(source, "request_header_bytes", path),
            response_header_bytes=_unsigned_int(
                source, "response_header_bytes", path
            ),
            flit_bytes=_positive_int(source, "flit_bytes", path),
            propagation_latency_fs=_unsigned_int(
                source, "propagation_latency_fs", path
            ),
            queue_depth_transactions=_positive_int(
                source, "queue_depth_transactions", path
            ),
            credits=_positive_int(source, "credits", path),
            duplex_mode=str(duplex_mode),
            clock_hz=_positive_int(source, "clock_hz", path),
        )
        if result.credits > result.queue_depth_transactions:
            raise BandwidthContractError(
                f"{path}.credits must not exceed queue_depth_transactions"
            )
        return result


@dataclass(frozen=True, slots=True)
class InternalDramContract:
    implementation: str
    channel_count: int
    pseudochannels_per_channel: int
    dq_bits_per_channel: int
    channel_width_bits: int
    transfers_per_clock: int
    rate_MTps: int
    nBL_cycles: int
    tCK_ps: int
    internal_prefetch_size: int
    transaction_bytes: int
    peak_payload_bandwidth_Bps: int

    @classmethod
    def load(cls, payload: object) -> "InternalDramContract":
        path = "bandwidth_contract.internal_dram"
        source = _mapping(payload, path)
        required = {
            "implementation",
            "channel_count",
            "pseudochannels_per_channel",
            "dq_bits_per_channel",
            "channel_width_bits",
            "transfers_per_clock",
            "rate_MTps",
            "nBL_cycles",
            "tCK_ps",
            "internal_prefetch_size",
            "transaction_bytes",
            "peak_payload_bandwidth_Bps",
        }
        _require_exact_keys(source, required, path)
        implementation = source.get("implementation")
        if not isinstance(implementation, str) or not implementation:
            raise BandwidthContractError(f"{path}.implementation is required")
        result = cls(
            implementation=implementation,
            channel_count=_positive_int(source, "channel_count", path),
            pseudochannels_per_channel=_positive_int(
                source, "pseudochannels_per_channel", path
            ),
            dq_bits_per_channel=_positive_int(
                source, "dq_bits_per_channel", path
            ),
            channel_width_bits=_positive_int(source, "channel_width_bits", path),
            transfers_per_clock=_positive_int(
                source, "transfers_per_clock", path
            ),
            rate_MTps=_positive_int(source, "rate_MTps", path),
            nBL_cycles=_positive_int(source, "nBL_cycles", path),
            tCK_ps=_positive_int(source, "tCK_ps", path),
            internal_prefetch_size=_positive_int(
                source, "internal_prefetch_size", path
            ),
            transaction_bytes=_positive_int(source, "transaction_bytes", path),
            peak_payload_bandwidth_Bps=_positive_int(
                source, "peak_payload_bandwidth_Bps", path
            ),
        )
        result.validate()
        return result

    @property
    def derived_transaction_bytes(self) -> Fraction:
        return Fraction(self.internal_prefetch_size * self.channel_width_bits, 8)

    @property
    def derived_tCK_ps(self) -> Fraction:
        return Fraction(1_000_000 * self.transfers_per_clock, self.rate_MTps)

    @property
    def phy_peak_bandwidth_Bps(self) -> Fraction:
        return Fraction(
            self.channel_count * self.dq_bits_per_channel * self.rate_MTps * 1_000_000,
            8,
        )

    @property
    def command_peak_bandwidth_Bps(self) -> Fraction:
        return Fraction(
            self.channel_count
            * self.pseudochannels_per_channel
            * self.transaction_bytes
            * 1_000_000_000_000,
            self.nBL_cycles * self.tCK_ps,
        )

    @property
    def clock_hz(self) -> Fraction:
        return Fraction(1_000_000_000_000, self.tCK_ps)

    def validate(self) -> None:
        if self.dq_bits_per_channel % 8 or self.channel_width_bits % 8:
            raise BandwidthContractError(
                "DRAM DQ and channel_width must be whole-byte widths"
            )
        if self.derived_transaction_bytes.denominator != 1 or int(
            self.derived_transaction_bytes
        ) != self.transaction_bytes:
            raise BandwidthContractError(
                "DRAM transaction_bytes disagrees with "
                "internal_prefetch_size * channel_width_bits / 8"
            )
        if self.derived_tCK_ps.denominator != 1 or int(self.derived_tCK_ps) != self.tCK_ps:
            raise BandwidthContractError(
                "DRAM tCK_ps disagrees with rate_MTps and transfers_per_clock"
            )
        if self.clock_hz.denominator != 1:
            raise BandwidthContractError(
                "DRAM tCK_ps must derive an integer clock frequency"
            )
        declared = Fraction(self.peak_payload_bandwidth_Bps, 1)
        if self.phy_peak_bandwidth_Bps != declared:
            raise BandwidthContractError(
                "DRAM declared peak bandwidth disagrees with DQ * rate * channels"
            )
        if self.command_peak_bandwidth_Bps != declared:
            raise BandwidthContractError(
                "DRAM command peak disagrees with transaction_bytes, nBL and tCK"
            )


@dataclass(frozen=True, slots=True)
class LogicDieGatewayContract:
    clock_hz: int
    ingress_queue_depth: int
    parent_table_entries: int
    split_width_per_cycle: int
    issue_width_per_cycle: int
    completion_width_per_cycle: int
    ordering_policy: str
    write_ack_policy: str

    @classmethod
    def load(cls, payload: object) -> "LogicDieGatewayContract":
        path = "bandwidth_contract.logic_die_gateway"
        source = _mapping(payload, path)
        required = {
            "clock_hz",
            "ingress_queue_depth",
            "parent_table_entries",
            "split_width_per_cycle",
            "issue_width_per_cycle",
            "completion_width_per_cycle",
            "ordering_policy",
            "write_ack_policy",
        }
        _require_exact_keys(source, required, path)
        ordering = source.get("ordering_policy")
        ack = source.get("write_ack_policy")
        if ordering not in {"fifo", "ordering_domain_fifo"}:
            raise BandwidthContractError(f"{path}.ordering_policy is invalid")
        if ack not in {"durable", "posted"}:
            raise BandwidthContractError(f"{path}.write_ack_policy is invalid")
        result = cls(
            clock_hz=_positive_int(source, "clock_hz", path),
            ingress_queue_depth=_positive_int(
                source, "ingress_queue_depth", path
            ),
            parent_table_entries=_positive_int(
                source, "parent_table_entries", path
            ),
            split_width_per_cycle=_positive_int(
                source, "split_width_per_cycle", path
            ),
            issue_width_per_cycle=_positive_int(
                source, "issue_width_per_cycle", path
            ),
            completion_width_per_cycle=_positive_int(
                source, "completion_width_per_cycle", path
            ),
            ordering_policy=str(ordering),
            write_ack_policy=str(ack),
        )
        if result.ingress_queue_depth > result.parent_table_entries:
            raise BandwidthContractError(
                f"{path}.ingress_queue_depth must not exceed parent_table_entries"
            )
        return result


@dataclass(frozen=True, slots=True)
class BandwidthContract:
    schema_version: str
    external_link: ExternalLinkContract
    logic_die_gateway: LogicDieGatewayContract
    internal_dram: InternalDramContract

    @classmethod
    def load(cls, payload: object) -> "BandwidthContract":
        source = _mapping(payload, "bandwidth_contract")
        _require_exact_keys(
            source,
            {
                "schema_version",
                "external_link",
                "logic_die_gateway",
                "internal_dram",
            },
            "bandwidth_contract",
        )
        if source.get("schema_version") != "hetero-bandwidth-contract/v1":
            raise BandwidthContractError(
                "bandwidth_contract.schema_version must be hetero-bandwidth-contract/v1"
            )
        return cls(
            schema_version="hetero-bandwidth-contract/v1",
            external_link=ExternalLinkContract.load(source["external_link"]),
            logic_die_gateway=LogicDieGatewayContract.load(
                source["logic_die_gateway"]
            ),
            internal_dram=InternalDramContract.load(source["internal_dram"]),
        )
