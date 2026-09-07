"""P25 deterministic QoS, fairness and forward-progress micro-model.

This module deliberately models *scheduler micro-cycles*, not GPU, ATLAS or
interconnect clock cycles.  It is suitable for deterministic control-plane and
liveness qualification only; it never opens a performance claim.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


class QoSConfigurationError(ValueError):
    """Raised when a P25 workload or policy is malformed."""


class ProgressWatchdogError(RuntimeError):
    """Raised when unfinished ready work makes no useful progress."""

    def __init__(self, kind: str, cycle: int, window_cycles: int) -> None:
        self.kind = kind
        self.cycle = cycle
        self.window_cycles = window_cycles
        super().__init__(
            f"{kind} detected at micro-cycle {cycle} after "
            f"{window_cycles} cycles without useful progress"
        )


class InterconnectActivationError(RuntimeError):
    """Raised when a requested interconnect backend cannot be proven active."""


BookSimProbe = Callable[[Path, Path], Mapping[str, object]]


def _jain(values: Sequence[float]) -> float:
    if not values or not any(values):
        return 1.0
    total = sum(values)
    return total * total / (len(values) * sum(value * value for value in values))


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def activate_interconnect_adapter(
    config: Mapping[str, object],
    project_root: Path,
    *,
    booksim2_probe: BookSimProbe | None = None,
) -> dict[str, object]:
    """Validate an interconnect activation request and fail closed.

    A BookSim2 file path alone is not activation evidence.  Activation also
    requires a runtime probe that proves the expected step/credit API and a
    deterministic smoke test.  The production repository does not yet provide
    that binding, so normal BookSim2 requests stop here rather than silently
    falling back to the analytical model.
    """

    backend = str(config.get("backend", "analytical"))
    if backend == "analytical":
        return {
            "requested_backend": backend,
            "active_backend": "analytical",
            "booksim2_active": False,
            "activation_proven": True,
            "performance_claim_allowed": False,
        }
    if backend != "booksim2":
        raise InterconnectActivationError(
            f"unsupported interconnect backend: {backend}"
        )

    expected_abi = "gpu-atlas-booksim2-adapter/v1"
    if str(config.get("adapter_abi", "")) != expected_abi:
        raise InterconnectActivationError(
            f"BookSim2 adapter ABI must be {expected_abi}"
        )
    required_paths: list[Path] = []
    for key in ("adapter_library", "network_config"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise InterconnectActivationError(f"BookSim2 {key} is required")
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise InterconnectActivationError(
                f"BookSim2 {key} does not exist: {path}"
            )
        required_paths.append(path)
    if booksim2_probe is None:
        raise InterconnectActivationError(
            "BookSim2 runtime probe is unavailable; analytical fallback is forbidden"
        )
    probe = dict(booksim2_probe(required_paths[0], required_paths[1]))
    required_evidence = {
        "adapter_abi": expected_abi,
        "cycle_step_api": True,
        "credit_accounting": True,
        "deterministic_probe_passed": True,
    }
    if any(probe.get(key) != value for key, value in required_evidence.items()):
        raise InterconnectActivationError(
            "BookSim2 runtime probe did not satisfy the activation contract"
        )
    return {
        "requested_backend": backend,
        "active_backend": backend,
        "booksim2_active": True,
        "activation_proven": True,
        "adapter_library": str(required_paths[0]),
        "network_config": str(required_paths[1]),
        "probe": probe,
        "performance_claim_allowed": False,
    }


def _validate_inputs(
    requests: Sequence[Mapping[str, object]], policy: Mapping[str, object]
) -> tuple[dict[str, int], int, int, int]:
    if not requests:
        raise QoSConfigurationError("at least one request is required")
    weights_raw = policy.get("class_weights")
    if not isinstance(weights_raw, Mapping) or not weights_raw:
        raise QoSConfigurationError("class_weights must be a non-empty mapping")
    weights = {str(key): int(value) for key, value in weights_raw.items()}
    if any(value <= 0 for value in weights.values()):
        raise QoSConfigurationError("all QoS weights must be positive")
    max_starvation = int(policy.get("max_starvation_cycles", 0))
    watchdog = int(policy.get("watchdog_window_cycles", 0))
    max_cycles = int(policy.get("max_micro_cycles", 0))
    if min(max_starvation, watchdog, max_cycles) <= 0:
        raise QoSConfigurationError(
            "starvation, watchdog and maximum cycle limits must be positive"
        )
    identifiers: set[str] = set()
    for request in requests:
        request_id = str(request.get("request_id", ""))
        qos_class = str(request.get("qos_class", ""))
        stages = request.get("stages")
        if not request_id or request_id in identifiers:
            raise QoSConfigurationError("request IDs must be non-empty and unique")
        identifiers.add(request_id)
        if qos_class not in weights:
            raise QoSConfigurationError(f"unknown QoS class: {qos_class}")
        if not isinstance(stages, Sequence) or not stages:
            raise QoSConfigurationError(f"{request_id} must contain stages")
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise QoSConfigurationError("each stage must be a mapping")
            if not str(stage.get("resource_id", "")):
                raise QoSConfigurationError("stage resource_id is required")
            if int(stage.get("work_units", 0)) <= 0:
                raise QoSConfigurationError("stage work_units must be positive")
    return weights, max_starvation, watchdog, max_cycles


def simulate_qos_micro_stress(
    requests: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    *,
    fault_injection: str | None = None,
) -> dict[str, object]:
    """Run deterministic weighted arbitration with bounded starvation.

    ``deadlock`` and ``livelock`` are qualification-only fault injections used
    to prove that both watchdog branches actually terminate a broken run.
    """

    weights, max_starvation, watchdog_window, max_cycles = _validate_inputs(
        requests, policy
    )
    if fault_injection not in {None, "deadlock", "livelock"}:
        raise QoSConfigurationError("unknown fault injection")

    states: dict[str, dict[str, object]] = {}
    total_work_by_class: dict[str, int] = defaultdict(int)
    for raw in requests:
        request = dict(raw)
        request_id = str(request["request_id"])
        stages = [dict(stage) for stage in request["stages"]]  # type: ignore[arg-type]
        qos_class = str(request["qos_class"])
        total_work = sum(int(stage["work_units"]) for stage in stages)
        total_work_by_class[qos_class] += total_work
        states[request_id] = {
            "request_id": request_id,
            "qos_class": qos_class,
            "fairness_group": str(request.get("fairness_group", request_id)),
            "weight": weights[qos_class],
            "arrival_cycle": int(request.get("arrival_cycle", 0)),
            "stages": stages,
            "stage_index": 0,
            "remaining_units": int(stages[0]["work_units"]),
            "ready_cycle": int(request.get("arrival_cycle", 0)),
            "last_service_cycle": None,
            "service_units": 0,
            "prefix_service_units": 0,
            "virtual_runtime": 0.0,
            "max_wait_cycles": 0,
            "completion_cycle": None,
            "resource_service": defaultdict(int),
        }

    service_events: list[dict[str, object]] = []
    progress_samples: list[dict[str, object]] = []
    class_prefix_service: dict[str, int] = defaultdict(int)
    prefix_window = int(policy.get("fairness_window_cycles", watchdog_window * 2))
    last_progress_cycle = -1
    progress_anchor_digest: str | None = None
    livelock_nonce = 0

    for cycle in range(max_cycles):
        unfinished = [
            state for state in states.values() if state["completion_cycle"] is None
        ]
        if not unfinished:
            break
        ready = [
            state
            for state in unfinished
            if int(state["arrival_cycle"]) <= cycle
            and int(state["ready_cycle"]) <= cycle
        ]
        by_resource: dict[str, list[dict[str, object]]] = defaultdict(list)
        for state in ready:
            stages = state["stages"]
            stage = stages[int(state["stage_index"])]  # type: ignore[index]
            by_resource[str(stage["resource_id"])].append(state)

        useful_progress = 0
        for resource_id in sorted(by_resource):
            candidates = by_resource[resource_id]
            for candidate in candidates:
                last = candidate["last_service_cycle"]
                wait = (
                    cycle - int(candidate["ready_cycle"])
                    if last is None
                    else cycle - int(last) - 1
                )
                candidate["max_wait_cycles"] = max(
                    int(candidate["max_wait_cycles"]), wait
                )
            starving = [
                item
                for item in candidates
                if (
                    cycle - int(item["ready_cycle"])
                    if item["last_service_cycle"] is None
                    else cycle - int(item["last_service_cycle"]) - 1
                )
                >= max_starvation
            ]
            if starving:
                selected = min(
                    starving,
                    key=lambda item: (
                        -(
                            cycle - int(item["ready_cycle"])
                            if item["last_service_cycle"] is None
                            else cycle - int(item["last_service_cycle"]) - 1
                        ),
                        -int(item["weight"]),
                        str(item["request_id"]),
                    ),
                )
                arbitration_reason = "starvation_override"
            else:
                selected = min(
                    candidates,
                    key=lambda item: (
                        float(item["virtual_runtime"]),
                        -int(item["weight"]),
                        int(item["ready_cycle"]),
                        str(item["request_id"]),
                    ),
                )
                arbitration_reason = "weighted_fair"

            if fault_injection == "deadlock":
                continue
            if fault_injection == "livelock":
                livelock_nonce += 1
                service_events.append(
                    {
                        "cycle": cycle,
                        "resource_id": resource_id,
                        "request_id": selected["request_id"],
                        "event": "RETRY",
                        "useful_progress": False,
                    }
                )
                continue

            selected["remaining_units"] = int(selected["remaining_units"]) - 1
            selected["service_units"] = int(selected["service_units"]) + 1
            selected["virtual_runtime"] = float(selected["virtual_runtime"]) + (
                1.0 / int(selected["weight"])
            )
            selected["last_service_cycle"] = cycle
            selected["resource_service"][resource_id] += 1  # type: ignore[index]
            qos_class = str(selected["qos_class"])
            if cycle < prefix_window:
                class_prefix_service[qos_class] += 1
                selected["prefix_service_units"] = (
                    int(selected["prefix_service_units"]) + 1
                )
            useful_progress += 1
            service_events.append(
                {
                    "cycle": cycle,
                    "resource_id": resource_id,
                    "request_id": selected["request_id"],
                    "qos_class": qos_class,
                    "event": "SERVICE",
                    "arbitration_reason": arbitration_reason,
                    "useful_progress": True,
                }
            )
            if int(selected["remaining_units"]) == 0:
                next_stage = int(selected["stage_index"]) + 1
                stages = selected["stages"]
                if next_stage == len(stages):  # type: ignore[arg-type]
                    selected["completion_cycle"] = cycle + 1
                else:
                    selected["stage_index"] = next_stage
                    selected["remaining_units"] = int(
                        stages[next_stage]["work_units"]  # type: ignore[index]
                    )
                    selected["ready_cycle"] = cycle + 1
                    selected["last_service_cycle"] = None

        state_view = [
            (
                request_id,
                int(state["stage_index"]),
                int(state["remaining_units"]),
                state["completion_cycle"],
                livelock_nonce,
            )
            for request_id, state in sorted(states.items())
        ]
        state_digest = _canonical_digest(state_view)
        ready_count = len(ready)
        if useful_progress:
            last_progress_cycle = cycle
            progress_anchor_digest = state_digest
        elif ready_count:
            if progress_anchor_digest is None:
                progress_anchor_digest = state_digest
            stalled_cycles = cycle - last_progress_cycle
            if stalled_cycles >= watchdog_window:
                kind = (
                    "deadlock"
                    if state_digest == progress_anchor_digest
                    else "livelock"
                )
                raise ProgressWatchdogError(kind, cycle, watchdog_window)
        progress_samples.append(
            {
                "cycle": cycle,
                "ready_requests": ready_count,
                "useful_progress_units": useful_progress,
                "state_digest": state_digest,
            }
        )
    else:
        raise ProgressWatchdogError("timeout", max_cycles, watchdog_window)

    request_records = []
    service_by_resource: dict[str, int] = defaultdict(int)
    normalized_service = []
    fairness_group_values: dict[str, list[float]] = defaultdict(list)
    for request_id, state in sorted(states.items()):
        resource_service = dict(sorted(state["resource_service"].items()))
        for resource_id, units in resource_service.items():
            service_by_resource[resource_id] += int(units)
        normalized_service.append(
            int(state["service_units"]) / int(state["weight"])
        )
        fairness_group_values[str(state["fairness_group"])].append(
            float(state["prefix_service_units"])
        )
        request_records.append(
            {
                "request_id": request_id,
                "qos_class": state["qos_class"],
                "weight": state["weight"],
                "fairness_group": state["fairness_group"],
                "arrival_cycle": state["arrival_cycle"],
                "completion_cycle": state["completion_cycle"],
                "service_units": state["service_units"],
                "service_units_in_fairness_window": state[
                    "prefix_service_units"
                ],
                "max_wait_cycles": state["max_wait_cycles"],
                "resource_service_units": resource_service,
            }
        )
    total_input_work = sum(total_work_by_class.values())
    total_service = sum(int(item["service_units"]) for item in request_records)
    return {
        "schema_version": "hetero-p25-qos-runtime/v1",
        "timing_semantics": "deterministic_scheduler_micro_cycles",
        "performance_claim_allowed": False,
        "requests": request_records,
        "service_events": service_events,
        "progress_samples": progress_samples,
        "metrics": {
            "makespan_micro_cycles": max(
                int(item["completion_cycle"]) for item in request_records
            ),
            "service_units_by_resource": dict(sorted(service_by_resource.items())),
            "service_units_by_class_in_fairness_window": dict(
                sorted(class_prefix_service.items())
            ),
            "weighted_jain_fairness": _jain(normalized_service),
            "same_group_prefix_jain_fairness": {
                group: _jain(values)
                for group, values in sorted(fairness_group_values.items())
                if len(values) > 1
            },
            "max_observed_wait_cycles": max(
                int(item["max_wait_cycles"]) for item in request_records
            ),
            "starvation_override_count": sum(
                item.get("arbitration_reason") == "starvation_override"
                for item in service_events
            ),
        },
        "conservation": {
            "input_requests": len(requests),
            "finished_requests": sum(
                item["completion_cycle"] is not None for item in request_records
            ),
            "input_work_units": total_input_work,
            "service_work_units": total_service,
            "all_requests_finished": all(
                item["completion_cycle"] is not None for item in request_records
            ),
            "zero_work_in_flight": total_input_work == total_service,
        },
        "watchdog": {
            "window_cycles": watchdog_window,
            "deadlock_detected": False,
            "livelock_detected": False,
            "forward_progress_proven_for_run": True,
        },
        "qualification_boundary": (
            "P25 functional QoS/liveness micro-model only; micro-cycles are not "
            "GPU, ATLAS, BookSim2 or DRAM performance cycles"
        ),
    }
