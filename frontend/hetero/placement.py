"""Deterministic manual and first-match rule placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .ir import ModelNode


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    node_id: str
    target_device: str
    matched_rule: int | None
    reason: str


def _matches(node: ModelNode, match: Mapping[str, Any], active_batch: int) -> bool:
    checks = {
        "phase": node.phase.value,
        "layer_id": node.layer_id,
        "operator_group": node.attributes.get("operator_group"),
    }
    for key, expected in checks.items():
        if key in match and match[key] != expected:
            return False
    kv_len = int(node.attributes.get("attention_kv_len", 0))
    if "kv_len_min" in match and kv_len < int(match["kv_len_min"]):
        return False
    if "kv_len_max" in match and kv_len > int(match["kv_len_max"]):
        return False
    if "active_batch_min" in match and active_batch < int(match["active_batch_min"]):
        return False
    return True


def place_nodes(
    nodes: Sequence[ModelNode],
    placement: Mapping[str, Any],
    active_batch: int = 1,
) -> list[PlacementDecision]:
    mode = placement.get("mode", "manual")
    if mode not in {"manual", "rule_based"}:
        raise ValueError(f"placement mode {mode!r} is not executable in M1-M3")
    default = str(placement.get("default_target", "gpu0"))
    rules = placement.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("placement.rules must be an array")
    decisions: list[PlacementDecision] = []
    for node in nodes:
        target = default
        matched: int | None = None
        if mode == "rule_based":
            for index, rule in enumerate(rules):
                if not isinstance(rule, Mapping):
                    raise ValueError("each placement rule must be an object")
                match = rule.get("match", {})
                if not isinstance(match, Mapping):
                    raise ValueError("placement rule match must be an object")
                if _matches(node, match, active_batch):
                    target = str(rule["target"])
                    matched = index
                    break
        decisions.append(
            PlacementDecision(
                node_id=node.node_id,
                target_device=target,
                matched_rule=matched,
                reason="first_match" if matched is not None else "default_target",
            )
        )
    return decisions
