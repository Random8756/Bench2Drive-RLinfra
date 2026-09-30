"""Pure helpers for adaptive route sampling.

Maintains a sliding window of per-scenario success rates and converts them
into sampling weights (low success rate -> larger weight) so harder
scenarios are visited more often.

This module intentionally stays CARLA-free so the same logic can be reused
by runtime code and lightweight standalone tests.
"""

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import random

__layer__ = (1, "Scenario")

ADAPTIVE_META_KEY = "__adaptive_meta__"

DEFAULT_ADAPTIVE_CONFIG: Dict[str, Any] = {
    "window_size": 8,
    "success_threshold": 100.0,
    "low_success_rate": 0.5,
    "high_success_rate": 0.8,
    "low_success_weight": 2.0,
    "high_success_weight": 0.5,
    "warmup_min_samples": 4,
    "log_interval_updates": 100,
    "hard_scenarios": [],
    "hard_scenario_max_gap_samples": 0,
}


def _normalize_name_list(raw_names: Any) -> List[str]:
    if raw_names is None:
        return []
    values = [raw_names] if isinstance(raw_names, str) else list(raw_names)
    names: List[str] = []
    seen = set()
    for value in values:
        name = str(value).strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def normalize_adaptive_config(raw_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = dict(DEFAULT_ADAPTIVE_CONFIG)
    raw = dict(raw_config or {})
    config["window_size"] = max(1, int(raw.get("window_size", config["window_size"])))
    config["success_threshold"] = float(raw.get("success_threshold", config["success_threshold"]))
    config["low_success_rate"] = float(raw.get("low_success_rate", config["low_success_rate"]))
    config["high_success_rate"] = float(raw.get("high_success_rate", config["high_success_rate"]))
    config["low_success_weight"] = max(1e-6, float(raw.get("low_success_weight", config["low_success_weight"])))
    config["high_success_weight"] = max(1e-6, float(raw.get("high_success_weight", config["high_success_weight"])))
    config["warmup_min_samples"] = max(0, int(raw.get("warmup_min_samples", config["warmup_min_samples"])))
    config["log_interval_updates"] = max(0, int(raw.get("log_interval_updates", config["log_interval_updates"])))
    config["hard_scenarios"] = _normalize_name_list(
        raw.get("hard_scenarios", raw.get("hard_scenario", config["hard_scenarios"]))
    )
    config["hard_scenario_max_gap_samples"] = max(
        0,
        int(
            raw.get(
                "hard_scenario_max_gap_samples",
                raw.get("hard_scenario_max_gap", config["hard_scenario_max_gap_samples"]),
            )
        ),
    )
    return config


def default_sampling_state() -> Dict[str, Any]:
    return {
        "recent_results": [],
        "sampling_weight": 1.0,
    }


def default_meta_state() -> Dict[str, Any]:
    return {
        "update_count": 0,
        "sample_count": 0,
        "hard_next_due": {},
    }


def compute_sampling_weight(recent_results: Sequence[int], adaptive_config: Mapping[str, Any]) -> float:
    warmup_min_samples = int(adaptive_config.get("warmup_min_samples", DEFAULT_ADAPTIVE_CONFIG["warmup_min_samples"]))
    if len(recent_results) < warmup_min_samples:
        return 1.0

    success_rate = sum(recent_results) / len(recent_results)
    if success_rate > float(adaptive_config.get("high_success_rate", DEFAULT_ADAPTIVE_CONFIG["high_success_rate"])):
        return float(adaptive_config.get("high_success_weight", DEFAULT_ADAPTIVE_CONFIG["high_success_weight"]))
    if success_rate < float(adaptive_config.get("low_success_rate", DEFAULT_ADAPTIVE_CONFIG["low_success_rate"])):
        return float(adaptive_config.get("low_success_weight", DEFAULT_ADAPTIVE_CONFIG["low_success_weight"]))
    return 1.0


def ensure_shared_scenarios(
    shared_stats: Optional[MutableMapping[str, Dict[str, Any]]],
    shared_lock: Any,
    scenario_names: Iterable[str],
) -> None:
    if shared_stats is None or shared_lock is None:
        return
    with shared_lock:
        if ADAPTIVE_META_KEY not in shared_stats:
            shared_stats[ADAPTIVE_META_KEY] = default_meta_state()
        for scenario_name in scenario_names:
            if scenario_name not in shared_stats:
                shared_stats[scenario_name] = default_sampling_state()


def _record_field(record: Any, field_name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def is_record_crashed(record: Any) -> bool:
    status = str(_record_field(record, "status", "") or "").strip().lower()
    return (
        status == "crashed"
        or bool(_record_field(record, "crash_reason", ""))
        or bool(_record_field(record, "crash_type", ""))
        or bool(_record_field(record, "crash_detail", ""))
    )


def is_success_record(record: Any, adaptive_config: Mapping[str, Any]) -> bool:
    score_route = float(_record_field(record, "score_route", 0.0) or 0.0)
    success_threshold = float(
        adaptive_config.get("success_threshold", DEFAULT_ADAPTIVE_CONFIG["success_threshold"])
    )
    return (not is_record_crashed(record)) and score_route >= success_threshold


def build_updated_sampling_state(
    previous_state: Optional[Mapping[str, Any]],
    success: bool,
    adaptive_config: Mapping[str, Any],
) -> Dict[str, Any]:
    state = dict(previous_state or {})
    recent_results = list(state.get("recent_results", []) or [])
    recent_results.append(1 if success else 0)
    window_size = int(adaptive_config.get("window_size", DEFAULT_ADAPTIVE_CONFIG["window_size"]))
    recent_results = recent_results[-window_size:]
    return {
        "recent_results": recent_results,
        "sampling_weight": compute_sampling_weight(recent_results, adaptive_config),
    }


def _snapshot_shared_stats_unlocked(
    shared_stats: MutableMapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for scenario_name, state in shared_stats.items():
        if scenario_name == ADAPTIVE_META_KEY:
            continue
        state_dict = dict(state or {})
        recent_results = list(state_dict.get("recent_results", []) or [])
        success_rate = (sum(recent_results) / len(recent_results)) if recent_results else None
        snapshot[scenario_name] = {
            "recent_results": recent_results,
            "sampling_weight": float(state_dict.get("sampling_weight", 1.0)),
            "success_rate": success_rate,
        }
    return snapshot


def update_shared_sampling_state(
    shared_stats: Optional[MutableMapping[str, Dict[str, Any]]],
    shared_lock: Any,
    scenario_name: str,
    record: Any,
    adaptive_config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if shared_stats is None or shared_lock is None or record is None:
        return None

    success = is_success_record(record, adaptive_config)
    with shared_lock:
        if ADAPTIVE_META_KEY not in shared_stats:
            shared_stats[ADAPTIVE_META_KEY] = default_meta_state()
        previous_state = dict(shared_stats.get(scenario_name, {}) or {})
        next_state = build_updated_sampling_state(previous_state, success, adaptive_config)
        shared_stats[scenario_name] = next_state
        meta_state = dict(shared_stats.get(ADAPTIVE_META_KEY, {}) or {})
        meta_state["update_count"] = int(meta_state.get("update_count", 0)) + 1
        shared_stats[ADAPTIVE_META_KEY] = meta_state
        return next_state


def update_shared_sampling_state_with_periodic_snapshot(
    shared_stats: Optional[MutableMapping[str, Dict[str, Any]]],
    shared_lock: Any,
    scenario_name: str,
    record: Any,
    adaptive_config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if shared_stats is None or shared_lock is None or record is None:
        return None

    success = is_success_record(record, adaptive_config)
    log_interval = int(adaptive_config.get("log_interval_updates", DEFAULT_ADAPTIVE_CONFIG["log_interval_updates"]))
    with shared_lock:
        if ADAPTIVE_META_KEY not in shared_stats:
            shared_stats[ADAPTIVE_META_KEY] = default_meta_state()
        previous_state = dict(shared_stats.get(scenario_name, {}) or {})
        next_state = build_updated_sampling_state(previous_state, success, adaptive_config)
        shared_stats[scenario_name] = next_state

        meta_state = dict(shared_stats.get(ADAPTIVE_META_KEY, {}) or {})
        update_count = int(meta_state.get("update_count", 0)) + 1
        meta_state["update_count"] = update_count
        shared_stats[ADAPTIVE_META_KEY] = meta_state

        snapshot = None
        if log_interval > 0 and update_count % log_interval == 0:
            snapshot = _snapshot_shared_stats_unlocked(shared_stats)

        return {
            "state": next_state,
            "update_count": update_count,
            "snapshot": snapshot,
        }


def snapshot_sampling_weights(
    scenario_names: Sequence[str],
    shared_stats: Optional[MutableMapping[str, Dict[str, Any]]],
    shared_lock: Any,
) -> List[float]:
    if shared_stats is None or shared_lock is None:
        return [1.0] * len(scenario_names)

    with shared_lock:
        return [
            max(1e-6, float(dict(shared_stats.get(name, {}) or {}).get("sampling_weight", 1.0)))
            for name in scenario_names
        ]


def _choice_by_weight(
    scenario_names: Sequence[str],
    weights: Sequence[float],
    rng: Optional[random.Random] = None,
) -> str:
    chooser = rng if rng is not None else random
    return chooser.choices(list(scenario_names), weights=list(weights), k=1)[0]


def select_scenario_name(
    scenario_names: Sequence[str],
    shared_stats: Optional[MutableMapping[str, Dict[str, Any]]],
    shared_lock: Any,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    if not scenario_names:
        return None
    weights = snapshot_sampling_weights(scenario_names, shared_stats, shared_lock)
    return _choice_by_weight(scenario_names, weights, rng)


def select_scenario_name_for_sample(
    scenario_names: Sequence[str],
    shared_stats: Optional[MutableMapping[str, Dict[str, Any]]],
    shared_lock: Any,
    adaptive_config: Mapping[str, Any],
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    if not scenario_names:
        return None
    if shared_stats is None or shared_lock is None:
        return select_scenario_name(scenario_names, shared_stats, shared_lock, rng)

    ordered_names = list(scenario_names)
    scenario_set = set(ordered_names)
    hard_names = [
        name
        for name in _normalize_name_list(adaptive_config.get("hard_scenarios"))
        if name in scenario_set
    ]
    hard_order = {name: idx for idx, name in enumerate(hard_names)}
    max_gap = int(
        adaptive_config.get(
            "hard_scenario_max_gap_samples",
            DEFAULT_ADAPTIVE_CONFIG["hard_scenario_max_gap_samples"],
        )
    )

    with shared_lock:
        meta_state = dict(shared_stats.get(ADAPTIVE_META_KEY, {}) or default_meta_state())
        sample_count = int(meta_state.get("sample_count", 0))
        hard_next_due = dict(meta_state.get("hard_next_due", {}) or {})

        if hard_names and max_gap > 0:
            for offset, name in enumerate(hard_names):
                if name not in hard_next_due:
                    first_due = max(0, round((offset + 1) * max_gap / len(hard_names)) - 1)
                    hard_next_due[name] = sample_count + first_due
            hard_next_due = {name: int(hard_next_due[name]) for name in hard_names}
            overdue = [name for name in hard_names if sample_count >= hard_next_due[name]]
            selected = min(overdue, key=lambda name: (hard_next_due[name], hard_order[name])) if overdue else None
        else:
            hard_next_due = {}
            selected = None

        if selected is None:
            weights = [
                max(1e-6, float(dict(shared_stats.get(name, {}) or {}).get("sampling_weight", 1.0)))
                for name in ordered_names
            ]
            selected = _choice_by_weight(ordered_names, weights, rng)

        sample_count += 1
        meta_state["sample_count"] = sample_count
        if selected in hard_next_due and max_gap > 0:
            hard_next_due[selected] = sample_count + max(0, max_gap - 1)
        meta_state["hard_next_due"] = hard_next_due
        shared_stats[ADAPTIVE_META_KEY] = meta_state
        return selected
