"""Classify training episodes with the B2D success and completion rules."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

_HARD_INFRACTION_EVENT_TYPES = frozenset({
    "COLLISION_PEDESTRIAN",
    "COLLISION_VEHICLE",
    "COLLISION_STATIC",
    "COLLISION_UNKNOWN",
    "TRAFFIC_LIGHT_INFRACTION",
    "STOP_INFRACTION",
    "OUTSIDE_ROUTE_LANES_INFRACTION",
    "ROUTE_DEVIATION",
    "YIELD_TO_EMERGENCY_VEHICLE",
    "VEHICLE_BLOCKED",
    "SCENARIO_TIMEOUT",
    "ROUTE_TIMEOUT",
    "ON_SIDEWALK_INFRACTION",
    "OUTSIDE_LANE_INFRACTION",
    "WRONG_WAY_INFRACTION",
})

_SOFT_INFRACTION_EVENT_TYPES = frozenset({
    "MIN_SPEED_INFRACTION",
})

_SUCCESS_EVENT_TYPES = frozenset({
    "SUCCESS",
})


def _extract_event_type(event: Any) -> str:
    if isinstance(event, Mapping):
        etype = (
            event.get("type")
            or event.get("event_type")
            or event.get("event")
            or event.get("name")
            or ""
        )
        return str(etype).strip()
    etype = getattr(event, "event_type", None) or getattr(event, "type", None)
    return str(etype).strip() if etype is not None else ""


def _iter_events(events: Optional[Iterable[Any]]) -> List[Any]:
    if events is None:
        return []
    try:
        return list(events)
    except TypeError:
        return []


def collect_episode_signals(
    episode_infos: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate useful flags from every step ``info`` of one episode.

    Parameters
    ----------
    episode_infos:
        Ordered list of ``info`` dicts produced by ``env.step`` for a single
        episode (most recent last).

    Returns
    -------
    dict with keys:
        ``route_completed`` : float in [0, 100], best value observed.
        ``has_hard_infraction`` : bool.
        ``has_success_event`` : bool (custom SUCCESS termination event).
        ``last_info`` : last info dict (for caller convenience).
    """
    best_completed = 0.0
    has_hard = False
    has_success = False

    for info in episode_infos:
        if not isinstance(info, Mapping):
            continue

        # ``route_completed_ratio`` is sometimes filled directly by the reward
        # handler (scale 0..1).  Fall back to ROUTE_COMPLETION events (percent).
        rcr = info.get("route_completed_ratio")
        if isinstance(rcr, (int, float)):
            best_completed = max(best_completed, float(rcr) * 100.0)

        for event in _iter_events(info.get("all_events")):
            etype = _extract_event_type(event)
            if not etype:
                continue
            if etype == "ROUTE_COMPLETION":
                details = event.get("details", {}) if isinstance(event, Mapping) else {}
                rc = details.get("route_completed") if isinstance(details, Mapping) else None
                if isinstance(rc, (int, float)):
                    best_completed = max(best_completed, float(rc))
            elif etype in _HARD_INFRACTION_EVENT_TYPES:
                has_hard = True
            # Soft infractions are intentionally ignored (B2D exception list).

        for event in _iter_events(info.get("terminate_events")):
            etype = _extract_event_type(event)
            details = event.get("details", {}) if isinstance(event, Mapping) else {}
            rc = details.get("route_completed") if isinstance(details, Mapping) else None
            if isinstance(rc, (int, float)):
                best_completed = max(best_completed, float(rc))
            if etype in _HARD_INFRACTION_EVENT_TYPES:
                has_hard = True
            if etype in _SUCCESS_EVENT_TYPES:
                has_success = True

    return {
        "route_completed": float(min(best_completed, 100.0)),
        "has_hard_infraction": bool(has_hard),
        "has_success_event": bool(has_success),
        "last_info": dict(episode_infos[-1]) if episode_infos else {},
    }


def classify_episode(
    episode_infos: Sequence[Mapping[str, Any]],
    *,
    completion_threshold: float = 90.0,
    strict_success_threshold: float = 99.9,
    require_clean_for_high_completion: bool = True,
) -> Dict[str, Any]:
    """Classify an episode according to the B2D-style success rule.

    Parameters
    ----------
    episode_infos:
        Ordered list of ``info`` dicts for one episode.
    completion_threshold:
        Route completion percentage above which the trajectory is eligible for
        the static buffer as a **high-completion** (non-SUCCESS) sample.
    strict_success_threshold:
        Route completion percentage above which the trajectory qualifies as a
        **SUCCESS** (combined with no hard infractions).
    require_clean_for_high_completion:
        If True (default), high-completion tier also requires no hard
        infractions – matching B2D spirit and the user's requirement that
        high-completion samples be useful demos.

    Returns
    -------
    dict with:
        ``is_success``        : bool, strict B2D success.
        ``is_high_completion``: bool, completion ≥ ``completion_threshold``
                                (and clean if configured so).
        ``route_completed``   : float (0..100).
        ``has_hard_infraction``: bool.
        ``has_success_event`` : bool.
    """
    sig = collect_episode_signals(episode_infos)

    completed = sig["route_completed"]
    has_hard = sig["has_hard_infraction"]
    has_success_event = sig["has_success_event"]

    # ``is_success`` is the training-time B2D analogue:
    #   * route_completed ~ 100%
    #   * AND no hard infraction
    # ``SUCCESS`` events are useful diagnostics, but they do not override the
    # strict completion threshold.
    is_success = (
        completed >= strict_success_threshold
        and not has_hard
    )

    is_high_completion = completed >= completion_threshold
    if require_clean_for_high_completion:
        is_high_completion = is_high_completion and not has_hard

    # SUCCESS implies high_completion regardless of the threshold.
    if is_success:
        is_high_completion = True

    return {
        "is_success": bool(is_success),
        "is_high_completion": bool(is_high_completion),
        "route_completed": float(completed),
        "has_hard_infraction": bool(has_hard),
        "has_success_event": bool(has_success_event),
    }
