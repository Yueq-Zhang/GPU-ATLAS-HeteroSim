import json
from pathlib import Path

import pytest

from frontend.hetero.qos_watchdog import (
    InterconnectActivationError,
    ProgressWatchdogError,
    QoSConfigurationError,
    activate_interconnect_adapter,
    simulate_qos_micro_stress,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/hetero/p25/p25_qos_watchdog_micro.json"


def _profile() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_qos_micro_stress_is_deterministic_and_conserves_work() -> None:
    profile = _profile()
    first = simulate_qos_micro_stress(
        profile["requests"], profile["policy"]  # type: ignore[arg-type]
    )
    second = simulate_qos_micro_stress(
        profile["requests"], profile["policy"]  # type: ignore[arg-type]
    )
    assert first == second
    assert first["conservation"]["all_requests_finished"]
    assert first["conservation"]["zero_work_in_flight"]
    assert first["performance_claim_allowed"] is False


def test_qos_micro_stress_covers_both_devices_and_bounds_starvation() -> None:
    profile = _profile()
    result = simulate_qos_micro_stress(
        profile["requests"], profile["policy"]  # type: ignore[arg-type]
    )
    metrics = result["metrics"]
    assert set(metrics["service_units_by_resource"]) == {
        "gpu0",
        "atlas0.compute",
    }
    assert metrics["max_observed_wait_cycles"] <= 5
    prefix = metrics["service_units_by_class_in_fairness_window"]
    assert prefix["latency"] > prefix["throughput"]
    assert metrics["same_group_prefix_jain_fairness"]["normal-gpu-peers"] >= 0.95
    assert result["watchdog"]["forward_progress_proven_for_run"]


@pytest.mark.parametrize("fault", ["deadlock", "livelock"])
def test_progress_watchdog_classifies_fault(fault: str) -> None:
    profile = _profile()
    with pytest.raises(ProgressWatchdogError) as caught:
        simulate_qos_micro_stress(
            profile["requests"],  # type: ignore[arg-type]
            profile["policy"],  # type: ignore[arg-type]
            fault_injection=fault,
        )
    assert caught.value.kind == fault


def test_unknown_qos_class_is_rejected() -> None:
    profile = _profile()
    requests = [dict(profile["requests"][0])]  # type: ignore[index]
    requests[0]["qos_class"] = "missing"
    with pytest.raises(QoSConfigurationError, match="unknown QoS class"):
        simulate_qos_micro_stress(
            requests, profile["policy"]  # type: ignore[arg-type]
        )


def test_analytical_interconnect_is_explicit_and_not_booksim2() -> None:
    record = activate_interconnect_adapter({"backend": "analytical"}, ROOT)
    assert record["active_backend"] == "analytical"
    assert record["booksim2_active"] is False
    assert record["performance_claim_allowed"] is False


def test_booksim2_request_fails_closed_without_runtime_files() -> None:
    config = json.loads(
        (ROOT / "configs/hetero/p25/p25_booksim2_fail_closed.json").read_text(
            encoding="utf-8"
        )
    )
    with pytest.raises(InterconnectActivationError, match="does not exist"):
        activate_interconnect_adapter(config, ROOT)


def test_booksim2_activation_requires_and_checks_runtime_probe(
    tmp_path: Path,
) -> None:
    library = tmp_path / "libbooksim2_adapter.so"
    network = tmp_path / "mesh.cfg"
    library.write_bytes(b"qualification fixture")
    network.write_text("k = 1; n = 1;\n", encoding="utf-8")
    config = {
        "backend": "booksim2",
        "adapter_abi": "gpu-atlas-booksim2-adapter/v1",
        "adapter_library": str(library),
        "network_config": str(network),
    }
    with pytest.raises(InterconnectActivationError, match="probe is unavailable"):
        activate_interconnect_adapter(config, ROOT)

    def probe(_library: Path, _network: Path) -> dict[str, object]:
        return {
            "adapter_abi": "gpu-atlas-booksim2-adapter/v1",
            "cycle_step_api": True,
            "credit_accounting": True,
            "deterministic_probe_passed": True,
        }

    record = activate_interconnect_adapter(config, ROOT, booksim2_probe=probe)
    assert record["booksim2_active"] is True
    assert record["activation_proven"] is True
    assert record["performance_claim_allowed"] is False
