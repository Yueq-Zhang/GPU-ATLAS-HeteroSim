from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontend.hetero.booksim2_memory_runtime import (
    BookSim2MemoryRuntimeError,
    run_booksim2_ramulator2_qos,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/hetero/p25/p27_booksim2_ramulator2_qos.json"


class _FakeNoC:
    def __init__(self, _library: Path, _network: Path, _nodes: int) -> None:
        self.cycle = 0
        self.pending: list[dict[str, int]] = []
        self.ready: list[dict[str, int]] = []
        self.accepted_packets = 0
        self.accepted_flits = 0
        self.delivered_packets = 0
        self.delivered_flits = 0

    def submit(self, flow_id: int, source_node: int, destination_node: int, flit_count: int) -> None:
        self.pending.append(
            {
                "flow_id": flow_id,
                "source_node": source_node,
                "destination_node": destination_node,
                "flit_count": flit_count,
                "accepted_cycle": self.cycle,
            }
        )
        self.accepted_packets += 1
        self.accepted_flits += flit_count

    def step(self) -> None:
        self.cycle += 1
        for packet in self.pending:
            packet["completion_cycle"] = self.cycle
            self.ready.append(packet)
        self.pending.clear()

    def pop_completions(self) -> list[dict[str, int]]:
        result, self.ready = self.ready, []
        self.delivered_packets += len(result)
        self.delivered_flits += sum(item["flit_count"] for item in result)
        return result

    def stats(self) -> dict[str, object]:
        drained = not self.pending and not self.ready
        return {
            "cycles": self.cycle,
            "accepted_packets": self.accepted_packets,
            "injected_packets": self.accepted_packets,
            "delivered_packets": self.delivered_packets,
            "accepted_flits": self.accepted_flits,
            "delivered_flits": self.delivered_flits,
            "outstanding_credits": 0,
            "queued_packets": len(self.pending),
            "completed_unpolled_packets": len(self.ready),
            "network_drained": drained,
            "zero_in_flight": drained,
        }

    def close(self) -> dict[str, object]:
        return self.stats()


class _FakeDRAM:
    GPU_INITIATOR = 0
    ATLAS_INITIATOR = 1
    SEND_RETRY = 0
    SEND_ACCEPTED = 1

    def __init__(self, _root: Path, _config: object) -> None:
        self.pending: list[tuple[int, int]] = []
        self.ready: list[tuple[int, int]] = []
        self.accepted = 0
        self.completed = 0

    def send_from_noc(
        self,
        parent_id: int,
        _global_address: int,
        _size_bytes: int,
        _operation: str,
        initiator: int,
        _ordering_domain: int,
        _sequence_number: int,
    ) -> int:
        self.pending.append((parent_id, initiator))
        self.accepted += 1
        return self.SEND_ACCEPTED

    def tick(self) -> None:
        self.ready.extend(self.pending)
        self.pending.clear()

    def pop_completions(self) -> list[dict[str, int]]:
        result = [
            {"parent_id": parent_id, "initiator": initiator}
            for parent_id, initiator in self.ready
        ]
        self.completed += len(result)
        self.ready.clear()
        return result

    def stats(self) -> dict[str, object]:
        return {
            "accepted_parent_ids": self.accepted,
            "observed_completion_ids": self.completed,
            "outstanding": len(self.pending) + len(self.ready),
        }

    def close(self) -> dict[str, object]:
        return self.stats()


def _load() -> tuple[list[dict[str, object]], dict[str, object]]:
    config = json.loads(PROFILE.read_text(encoding="utf-8"))
    return config.pop("requests"), config


def test_four_realistic_noc_segments_surround_each_durable_parent() -> None:
    requests, config = _load()
    result = run_booksim2_ramulator2_qos(
        ROOT,
        requests,
        config,
        noc_factory=_FakeNoC,
        dram_factory=_FakeDRAM,
    )
    assert result["conservation"]["parent_requests"] == 4
    assert result["conservation"]["packets"] == 16
    assert result["conservation"]["zero_in_flight"] is True
    assert result["qos"]["max_observed_packet_wait_cycles"] > 0
    assert {item["initiator"] for item in result["requests"]} == {
        "gpu0",
        "atlas0.compute",
    }
    for parent_id in range(1, 5):
        events = [
            item["event"]
            for item in result["request_events"]
            if item["parent_id"] == parent_id
        ]
        assert events.count("NOC_SEGMENT_COMPLETE") == 4
        assert events.index("RAMULATOR_ACCEPT") < events.index("RAMULATOR_DURABLE")


def test_unknown_initiator_fails_closed() -> None:
    requests, config = _load()
    requests[0]["initiator"] = "cpu0"
    with pytest.raises(BookSim2MemoryRuntimeError, match="unknown memory initiator"):
        run_booksim2_ramulator2_qos(
            ROOT,
            requests,
            config,
            noc_factory=_FakeNoC,
            dram_factory=_FakeDRAM,
        )
