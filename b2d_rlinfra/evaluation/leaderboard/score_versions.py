#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROUND_DIGITS = 3
ROUND_DIGITS_SCORE = 6

_FIXED_PENALTIES_V20 = {
    "collisions_pedestrian": 0.5,
    "collisions_vehicle": 0.6,
    "collisions_layout": 0.65,
    "red_light": 0.7,
    "stop_infraction": 0.8,
    "scenario_timeouts": 0.7,
    "yield_emergency_vehicle_infractions": 0.7,
}

_INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
]

_LABELS_V20 = [
    "Avg. driving score",
    "Avg. route completion",
    "Avg. infraction penalty",
    "Collisions with pedestrians",
    "Collisions with vehicles",
    "Collisions with layout",
    "Red lights infractions",
    "Stop sign infractions",
    "Off-road infractions",
    "Route deviations",
    "Route timeouts",
    "Agent blocked",
    "Yield emergency vehicles infractions",
    "Scenario timeouts",
    "Min speed infractions",
]

_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)%")
_OUTSIDE_ROUTE_LANES_RE = re.compile(
    r"for about\s+(-?\d+(?:\.\d+)?)\s+meters\s+\((-?\d+(?:\.\d+)?)%\s+of the completed route\)",
    re.IGNORECASE,
)


class ScoreVersionError(ValueError):
    pass


def _clip_percentage(value: float) -> float:
    return max(0.0, min(100.0, value))


def _extract_metric(
    labels: Sequence[str],
    values: Sequence[str],
    label: str,
) -> Optional[float]:
    try:
        index = list(labels).index(label)
    except ValueError:
        return None

    try:
        return float(values[index])
    except (TypeError, ValueError, IndexError):
        return None


def parse_outside_route_lanes_message(message: str) -> Tuple[float, float]:
    match = _OUTSIDE_ROUTE_LANES_RE.search(str(message))
    if not match:
        raise ScoreVersionError(f"Unable to parse outside-route-lanes message: {message}")

    meters = float(match.group(1))
    percentage = _clip_percentage(float(match.group(2)))
    return meters, percentage


def parse_outside_route_lanes_percentage(message: str) -> float:
    _, percentage = parse_outside_route_lanes_message(message)
    return percentage


def parse_min_speed_percentage(message: str) -> float:
    match = _PERCENT_RE.search(str(message))
    if not match:
        raise ScoreVersionError(f"Unable to parse min-speed message: {message}")
    return _clip_percentage(float(match.group(1)))


def _is_missing_route_record(record: Mapping[str, Any]) -> bool:
    return str(record.get("status", "")).strip() in (
        "Failed - Missing shard result",
        "Failed - Missing route result after retries",
    )


def _get_messages(record: Mapping[str, Any], key: str) -> List[str]:
    infractions = record.get("infractions", {}) or {}
    messages = infractions.get(key, []) or []
    return [str(message) for message in messages]


def _build_v20_route_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    updated_record = copy.deepcopy(dict(record))
    updated_record.setdefault("scores", {})
    updated_record.setdefault("infractions", {})
    updated_record.setdefault("meta", {})

    score_route = float(updated_record["scores"].get("score_route", 0.0) or 0.0)
    if _is_missing_route_record(updated_record):
        score_penalty = 0.0
    else:
        score_penalty = 1.0

        for key, penalty_value in _FIXED_PENALTIES_V20.items():
            for _ in _get_messages(updated_record, key):
                score_penalty *= penalty_value

        for message in _get_messages(updated_record, "outside_route_lanes"):
            percentage = parse_outside_route_lanes_percentage(message)
            score_penalty *= (1.0 - percentage / 100.0)

        for message in _get_messages(updated_record, "min_speed_infractions"):
            percentage = parse_min_speed_percentage(message)
            score_penalty *= (1.0 - (1.0 - 0.7) * (1.0 - percentage / 100.0))

    infractions = updated_record.get("infractions", {}) or {}
    updated_record["num_infractions"] = sum(len(values or []) for values in infractions.values())
    updated_record["scores"]["score_route"] = round(score_route, ROUND_DIGITS_SCORE)
    updated_record["scores"]["score_penalty"] = round(score_penalty, ROUND_DIGITS_SCORE)
    updated_record["scores"]["score_composed"] = round(max(score_route * score_penalty, 0.0), ROUND_DIGITS_SCORE)
    return updated_record


def _compute_global_status(records: Sequence[Mapping[str, Any]]) -> Tuple[str, List[Tuple[Any, Any, Any]]]:
    global_status = "Perfect"
    exceptions: List[Tuple[Any, Any, Any]] = []

    for record in records:
        route_status = str(record.get("status", ""))
        route_result = "Failed" if "Failed" in route_status else route_status
        if route_result == "Failed":
            exceptions.append((record.get("route_id"), record.get("index"), route_status))
            global_status = "Failed"
        elif global_status == "Perfect" and route_result != "Perfect":
            global_status = route_result

    return global_status, exceptions


def _get_global_infraction_value(record: Mapping[str, Any], key: str) -> float:
    messages = _get_messages(record, key)
    if key == "outside_route_lanes":
        if not messages:
            return 0.0
        meters, _ = parse_outside_route_lanes_message(messages[0])
        return meters / 1000.0
    return float(len(messages))


def _compute_scores_std_dev(
    records: Sequence[Mapping[str, Any]],
    rounded_means: Mapping[str, float],
) -> Dict[str, float]:
    std_dev = {
        "score_composed": 0.0,
        "score_route": 0.0,
        "score_penalty": 0.0,
    }
    total_routes = len(records)
    if total_routes <= 1:
        return std_dev

    for record in records:
        scores = record.get("scores", {}) or {}
        for key in std_dev:
            diff = float(scores.get(key, 0.0) or 0.0) - float(rounded_means.get(key, 0.0) or 0.0)
            std_dev[key] += math.pow(diff, 2)

    for key in std_dev:
        std_dev[key] = round(math.sqrt(std_dev[key] / float(total_routes - 1)), ROUND_DIGITS)
    return std_dev


def _build_v20_global_record(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_routes = len(records)
    scores_mean = {
        "score_composed": 0.0,
        "score_route": 0.0,
        "score_penalty": 0.0,
    }
    infractions = {key: 0.0 for key in _INFRACTION_KEYS}
    meta = {
        "total_length": 0.0,
        "duration_game": 0.0,
        "duration_system": 0.0,
        "exceptions": [],
    }

    global_status, exceptions = _compute_global_status(records)
    meta["exceptions"] = exceptions

    for record in records:
        scores = record.get("scores", {}) or {}
        if total_routes:
            scores_mean["score_route"] += float(scores.get("score_route", 0.0) or 0.0) / total_routes
            scores_mean["score_penalty"] += float(scores.get("score_penalty", 0.0) or 0.0) / total_routes
            scores_mean["score_composed"] += float(scores.get("score_composed", 0.0) or 0.0) / total_routes

        record_meta = record.get("meta", {}) or {}
        meta["total_length"] += float(record_meta.get("route_length", 0.0) or 0.0)
        meta["duration_game"] += float(record_meta.get("duration_game", 0.0) or 0.0)
        meta["duration_system"] += float(record_meta.get("duration_system", 0.0) or 0.0)

    rounded_means = {
        "score_composed": round(scores_mean["score_composed"], ROUND_DIGITS_SCORE),
        "score_route": round(scores_mean["score_route"], ROUND_DIGITS_SCORE),
        "score_penalty": round(scores_mean["score_penalty"], ROUND_DIGITS_SCORE),
    }
    std_dev = _compute_scores_std_dev(records, rounded_means)

    km_driven = 0.0
    for record in records:
        record_meta = record.get("meta", {}) or {}
        scores = record.get("scores", {}) or {}
        km_driven += (
            float(record_meta.get("route_length", 0.0) or 0.0) / 1000.0
            * float(scores.get("score_route", 0.0) or 0.0) / 100.0
        )
        for key in infractions:
            infractions[key] += _get_global_infraction_value(record, key)

    km_driven = max(km_driven, 0.001)
    for key in infractions:
        if key != "outside_route_lanes":
            infractions[key] /= km_driven
        infractions[key] = round(infractions[key], ROUND_DIGITS)

    return {
        "index": -1,
        "route_id": -1,
        "status": global_status,
        "infractions": infractions,
        "scores_mean": rounded_means,
        "scores_std_dev": std_dev,
        "meta": {
            "total_length": meta["total_length"],
            "duration_game": meta["duration_game"],
            "duration_system": meta["duration_system"],
            "exceptions": meta["exceptions"],
        },
    }


def _build_values(global_record: Mapping[str, Any]) -> List[str]:
    infractions = global_record.get("infractions", {}) or {}
    scores_mean = global_record.get("scores_mean", {}) or {}
    return [
        str(scores_mean.get("score_composed", 0)),
        str(scores_mean.get("score_route", 0)),
        str(scores_mean.get("score_penalty", 0)),
        str(infractions.get("collisions_pedestrian", 0)),
        str(infractions.get("collisions_vehicle", 0)),
        str(infractions.get("collisions_layout", 0)),
        str(infractions.get("red_light", 0)),
        str(infractions.get("stop_infraction", 0)),
        str(infractions.get("outside_route_lanes", 0)),
        str(infractions.get("route_dev", 0)),
        str(infractions.get("route_timeout", 0)),
        str(infractions.get("vehicle_blocked", 0)),
        str(infractions.get("yield_emergency_vehicle_infractions", 0)),
        str(infractions.get("scenario_timeouts", 0)),
        str(infractions.get("min_speed_infractions", 0)),
    ]


def build_v20_statistics_from_payload(v21_payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(dict(v21_payload))
    checkpoint = payload.setdefault("_checkpoint", {})
    records = checkpoint.get("records", []) or []
    updated_records = [_build_v20_route_record(record) for record in records]
    global_record = _build_v20_global_record(updated_records)

    checkpoint["records"] = updated_records
    checkpoint["global_record"] = global_record
    if not checkpoint.get("progress"):
        checkpoint["progress"] = [len(updated_records), len(updated_records)]

    payload["labels"] = list(_LABELS_V20)
    payload["values"] = _build_values(global_record)
    return payload


def extract_summary_metrics_from_statistics_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    labels = payload.get("labels", []) or []
    values = payload.get("values", []) or []
    checkpoint = payload.get("_checkpoint", {}) or {}

    return {
        "entry_status": payload.get("entry_status"),
        "eligible": payload.get("eligible"),
        "score_mean": _extract_metric(labels, values, "Avg. driving score"),
        "route_completion_mean": _extract_metric(labels, values, "Avg. route completion"),
        "score_penalty_mean": _extract_metric(labels, values, "Avg. infraction penalty"),
        "collisions_pedestrian": _extract_metric(labels, values, "Collisions with pedestrians"),
        "collisions_vehicle": _extract_metric(labels, values, "Collisions with vehicles"),
        "min_speed_infractions": _extract_metric(labels, values, "Min speed infractions"),
        "global_record": checkpoint.get("global_record", {}) or {},
        "labels": list(labels),
        "values": list(values),
    }
