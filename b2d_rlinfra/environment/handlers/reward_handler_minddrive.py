"""MindDrive-compatible sparse reward handler."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import numpy as np

__layer__ = (2, "Environment")

logger = logging.getLogger("Interface Wrapper")


HARD_PENALTY_EVENTS = {
    "COLLISION_PEDESTRIAN",
    "COLLISION_VEHICLE",
    "COLLISION_STATIC",
    "TRAFFIC_LIGHT_INFRACTION",
    "STOP_INFRACTION",
    "ROUTE_DEVIATION",
    "OUTSIDE_ROUTE_LANES_INFRACTION",
}

EVENT_ALIASES = {
    "NOT_IN_CARLANE": "OUTSIDE_ROUTE_LANES_INFRACTION",
}

def _event_type(event: Any) -> str:
    if isinstance(event, Mapping):
        value = event.get("event_type", event.get("type", ""))
    else:
        get_type = getattr(event, "get_type", None)
        value = get_type() if callable(get_type) else event
    return getattr(value, "name", str(value))


def _canonical_event_type(event: Any) -> str:
    event_type = _event_type(event)
    return EVENT_ALIASES.get(event_type, event_type)


def _event_details(event: Any) -> Mapping[str, Any]:
    if isinstance(event, Mapping):
        details = event.get("details", {}) or {}
        return details if isinstance(details, Mapping) else {}
    get_dict = getattr(event, "get_dict", None)
    details = get_dict() if callable(get_dict) else {}
    return details if isinstance(details, Mapping) else {}


def _as_route_ratio(value: Any, *, ratio_hint: bool = False) -> Optional[float]:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return None
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    if not ratio_hint:
        value = value / 100.0
    elif value > 1.0:
        value = value / 100.0
    return float(min(max(value, 0.0), 1.0))


def _route_ratio_from_event(event: Any) -> Optional[float]:
    if isinstance(event, Mapping):
        ratio = _as_route_ratio(event.get("route_completed"), ratio_hint=False)
        if ratio is not None:
            return ratio
    details = _event_details(event)
    ratio = _as_route_ratio(details.get("route_completed"), ratio_hint=False)
    if ratio is not None:
        return ratio
    nested = details.get("dict")
    if isinstance(nested, Mapping):
        return _as_route_ratio(nested.get("route_completed"), ratio_hint=False)
    return None


def _route_ratio_from_info(info: Mapping[str, Any]) -> float:
    for key in ("route_completed_ratio", "truncation_route_completed_ratio"):
        ratio = _as_route_ratio(info.get(key), ratio_hint=True)
        if ratio is not None:
            return ratio
    best_ratio = 0.0
    for event_key in ("terminate_events", "all_events"):
        events = info.get(event_key, []) or []
        for event in events:
            ratio = _route_ratio_from_event(event)
            if ratio is not None:
                best_ratio = max(best_ratio, ratio)
    return float(best_ratio)


class RewardHandler:
    """Sparse terminal reward matching MindDrive rollout behavior."""

    def __init__(self, config: Mapping[str, Any], scenario_name: str = "leaderboardv2"):
        self.config = config
        self.scenario_name = scenario_name

    def reset(self) -> None:
        return None

    def destroy(self) -> None:
        return None

    def _classify(self, info: Mapping[str, Any], *, terminated: bool) -> tuple[float, str, str]:
        success_event = ""
        route_ratio = _route_ratio_from_info(info)

        # ``all_events`` is a cumulative history from the leaderboard criteria.
        # Only ``terminate_events`` represents events that fired for this step.
        for event in info.get("terminate_events", []) or []:
            event_type = _canonical_event_type(event)
            if event_type in HARD_PENALTY_EVENTS:
                return -1.0, "hard_penalty", event_type
            if event_type in ("SUCCESS", "ROUTE_COMPLETION"):
                success_event = event_type

        if terminated and not success_event:
            for event in info.get("all_events", []) or []:
                event_type = _canonical_event_type(event)
                if event_type != "ROUTE_COMPLETION":
                    continue
                ratio = _route_ratio_from_event(event)
                if ratio is None:
                    ratio = route_ratio
                if ratio >= 0.999:
                    success_event = event_type
                    break
            if not success_event and route_ratio >= 0.999:
                success_event = "ROUTE_COMPLETION"

        if success_event:
            return 1.0, "success", success_event
        return 0.0, "none", ""

    def generate_reward(
        self,
        observation: Any,
        terminated: bool,
        info: Mapping[str, Any],
        action: Any,
        name: str = "leaderboardv2",
        crash_message: str = "",
    ):
        if not isinstance(info, dict):
            info = dict(info or {})

        if crash_message == "Simulation crashed":
            reward = 0.0
            reason = "simulation_crashed"
            event = ""
            logger.warning("Simulation Crashed, Reward Is Set As 0.0")
        else:
            reward, reason, event = self._classify(info, terminated=bool(terminated))

        info["minddrive_reward"] = float(reward)
        info["minddrive_reward_reason"] = reason
        info["minddrive_reward_event"] = event
        info["minddrive_reward_route_completed_ratio"] = _route_ratio_from_info(info)
        return np.array(reward, dtype=np.float32), info, False
