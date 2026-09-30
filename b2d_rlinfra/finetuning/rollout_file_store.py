"""Rollout file writing, schema validation, GAE, and manifest state."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from b2d_rlinfra.finetuning.coordination import atomic_write_json
from b2d_rlinfra.finetuning.messages import RolloutMeta
from b2d_rlinfra.finetuning.rollout_pack import RolloutPackReader, RolloutPackWriter

REQUIRED_ROLLOUT_FIELDS = (
    "collector_id",
    "episode_id",
    "policy_version",
    "actions",
    "rewards",
    "episode_starts",
    "values",
    "old_action_log_probs",
    "advantages",
    "returns",
    "policy_input_state",
)


def compute_returns_and_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float,
    bootstrap_non_terminal: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE for one complete, non-crash episode fragment."""
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    if rewards_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("rewards and values must be 1-D arrays")
    if rewards_arr.shape[0] != values_arr.shape[0]:
        raise ValueError("rewards and values length mismatch")
    advantages = np.zeros_like(rewards_arr, dtype=np.float32)
    last_gae_lam = 0.0
    for step in reversed(range(len(rewards_arr))):
        if step == len(rewards_arr) - 1:
            next_non_terminal = 1.0 if bootstrap_non_terminal else 0.0
            next_value = float(bootstrap_value)
        else:
            next_non_terminal = 1.0
            next_value = float(values_arr[step + 1])
        delta = rewards_arr[step] + float(gamma) * next_value * next_non_terminal - values_arr[step]
        last_gae_lam = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    returns = advantages + values_arr
    return returns.astype(np.float32), advantages.astype(np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sequence_length(value: Any) -> int:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("policy_input_state dict is empty")
        return _sequence_length(next(iter(value.values())))
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError("rollout field must have a sample dimension")
    return int(arr.shape[0])


def _is_json_scalar(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return False
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, (str, int, float, bool)) or value is None


def _route_completion_ratio_from_info(info: Mapping[str, Any]) -> Optional[float]:
    if not isinstance(info, Mapping):
        return None

    def _as_ratio(value: Any, *, ratio_hint: bool) -> Optional[float]:
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
        if ratio_hint:
            if value > 1.0:
                value = value / 100.0
        else:
            value = value / 100.0
        return float(min(max(value, 0.0), 1.0))

    for key in ("route_completed_ratio", "truncation_route_completed_ratio"):
        ratio = _as_ratio(info.get(key), ratio_hint=True)
        if ratio is not None:
            return ratio

    for key in ("simple_reward_RC", "score_route"):
        ratio = _as_ratio(info.get(key), ratio_hint=False)
        if ratio is not None:
            return ratio

    best_ratio = None
    for event_key in ("terminate_events", "all_events"):
        events = info.get(event_key)
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            details = event.get("details")
            detail_maps = [details] if isinstance(details, Mapping) else []
            nested = details.get("dict") if isinstance(details, Mapping) else None
            if isinstance(nested, Mapping):
                detail_maps.append(nested)
            for detail in detail_maps:
                ratio = _as_ratio(detail.get("route_completed"), ratio_hint=False)
                if ratio is not None:
                    best_ratio = ratio if best_ratio is None else max(best_ratio, ratio)

    return best_ratio


def validate_rollout_payload(payload: Mapping[str, Any]) -> int:
    missing = [key for key in REQUIRED_ROLLOUT_FIELDS if key not in payload]
    if missing:
        raise ValueError(f"rollout payload missing required fields: {missing}")
    num_steps = _sequence_length(payload["actions"])
    if num_steps <= 0:
        raise ValueError("rollout payload must contain at least one step")
    for key in (
        "rewards",
        "episode_starts",
        "values",
        "old_action_log_probs",
        "advantages",
        "returns",
        "policy_input_state",
    ):
        length = _sequence_length(payload[key])
        if length != num_steps:
            raise ValueError(f"rollout field {key!r} length {length} != actions length {num_steps}")
    return num_steps


class RolloutFileStore:
    def __init__(self, rollout_dir: str):
        self.rollout_dir = Path(rollout_dir)
        self.rollout_dir.mkdir(parents=True, exist_ok=True)

    def episode_path(self, collector_id: int, episode_id: int) -> Path:
        collector_dir = self.rollout_dir / f"collector_{int(collector_id):03d}"
        collector_dir.mkdir(parents=True, exist_ok=True)
        return collector_dir / f"episode_{int(episode_id):06d}.rollout"

    def write_episode(
        self,
        *,
        collector_id: int,
        episode_id: int,
        policy_version: int,
        actions: Any,
        rewards: Any,
        episode_starts: Any,
        values: Any,
        old_action_log_probs: Any,
        advantages: Any,
        returns: Any,
        policy_input_state: Any,
        terminated: bool,
        truncated: bool,
        info: Optional[Mapping[str, Any]] = None,
        optional_fields: Optional[Mapping[str, Any]] = None,
    ) -> RolloutMeta:
        info = dict(info or {})
        sample_fields: Dict[str, Any] = {
            "actions": actions if isinstance(actions, Mapping) else np.asarray(actions),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "episode_starts": np.asarray(episode_starts, dtype=np.bool_),
            "values": np.asarray(values, dtype=np.float32),
            "old_action_log_probs": np.asarray(old_action_log_probs, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
            "returns": np.asarray(returns, dtype=np.float32),
            "policy_input_state": policy_input_state,
        }
        optional_metadata: Dict[str, Any] = {}
        for key, value in dict(optional_fields or {}).items():
            if key in sample_fields or key in {"collector_id", "episode_id", "policy_version"}:
                raise ValueError(f"optional rollout field {key!r} conflicts with a required field")
            if _is_json_scalar(value):
                optional_metadata[key] = _json_safe(value)
            else:
                # Non-scalars are sample data. RolloutPackWriter performs the
                # strict fixed-shape numeric validation and rejects ragged data.
                sample_fields[key] = value
        validation_payload = dict(sample_fields)
        validation_payload.update(
            collector_id=collector_id,
            episode_id=episode_id,
            policy_version=policy_version,
        )
        num_steps = validate_rollout_payload(validation_payload)
        meta = RolloutMeta(
            collector_id=int(collector_id),
            episode_id=int(episode_id),
            file_path=str(self.episode_path(collector_id, episode_id)),
            num_steps=num_steps,
            policy_version=int(policy_version),
            reward_sum=float(np.asarray(rewards, dtype=np.float32).sum()),
            terminated=bool(terminated),
            truncated=bool(truncated),
            route_completion_ratio=_route_completion_ratio_from_info(info),
            crashed=False,
            route_id=str(info.get("route_id", "")),
            scenario_name=str(info.get("scenario_name", "")),
            town=str(info.get("town", "")),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        route_completion = meta.route_completion_ratio
        outcome = (
            "truncated"
            if bool(truncated)
            else (
                "success"
                if info.get("minddrive_reward_reason") == "success"
                or (route_completion is not None and float(route_completion) >= 0.999)
                else "failure"
            )
        )
        metadata = {
            "collector_id": int(collector_id),
            "episode_id": int(episode_id),
            "policy_version": int(policy_version),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": False,
            "outcome": outcome,
            "rollout_meta": meta.to_dict(),
            "terminal_info": _json_safe(info),
            "optional": optional_metadata,
        }
        RolloutPackWriter.write(
            meta.file_path,
            metadata=metadata,
            sample_fields=sample_fields,
            num_steps=num_steps,
        )
        return meta

class RolloutManifest:
    """Append-only manifest with replayed ready/selected/consumed state."""

    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records_cache: Dict[str, Dict[str, Any]] = {}
        self._offset = 0

    def append_event(self, event: Mapping[str, Any]) -> None:
        record = dict(event)
        record.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def add_ready(self, meta: Mapping[str, Any]) -> None:
        if meta.get("crashed"):
            return
        self.append_event({"event": "ready", "file_path": str(meta["file_path"]), "meta": dict(meta)})

    def add_crash(self, meta: Mapping[str, Any]) -> None:
        self.append_event({"event": "crash", "meta": dict(meta)})

    def mark_selected(self, file_paths: Iterable[str], update_id: int) -> None:
        for path in file_paths:
            self.append_event({"event": "selected", "file_path": str(path), "update_id": int(update_id)})

    def mark_consumed(self, file_paths: Iterable[str], update_id: int) -> None:
        for path in file_paths:
            self.append_event({"event": "consumed", "file_path": str(path), "update_id": int(update_id)})

    def mark_dropped_stale(
        self,
        file_path: str,
        *,
        update_id: int,
        current_policy_version: int,
        rollout_policy_version: int,
        lag: int,
    ) -> None:
        self.append_event(
            {
                "event": "dropped_stale",
                "file_path": str(file_path),
                "update_id": int(update_id),
                "current_policy_version": int(current_policy_version),
                "rollout_policy_version": int(rollout_policy_version),
                "lag": int(lag),
                "reason": "policy_lag",
            }
        )

    def rollback_selected(self, file_paths: Iterable[str], update_id: int) -> None:
        for path in file_paths:
            self.append_event({"event": "ready", "file_path": str(path), "update_id": int(update_id)})

    def records(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return dict(self._records_cache)
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self._offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                file_path = event.get("file_path")
                if not file_path:
                    continue
                record = self._records_cache.setdefault(file_path, {})
                if event.get("event") == "ready":
                    record.update(event.get("meta") or {})
                    record["file_path"] = file_path
                    record["state"] = "ready"
                    if "update_id" in event:
                        record["update_id"] = event["update_id"]
                elif event.get("event") == "selected":
                    record["state"] = "selected"
                    record["update_id"] = event.get("update_id")
                elif event.get("event") == "consumed":
                    record["state"] = "consumed"
                    record["update_id"] = event.get("update_id")
                elif event.get("event") == "dropped_stale":
                    record["state"] = "dropped_stale"
                    record["update_id"] = event.get("update_id")
                    record["drop_reason"] = event.get("reason", "policy_lag")
                    record["policy_lag"] = event.get("lag")
            self._offset = f.tell()
        return dict(self._records_cache)

    def cleanup_consumed_files(self, *, keep_last_n_update_rollouts: int = 0) -> List[Path]:
        """Delete consumed RolloutPack files outside the recent-update retention window."""
        records = self.records()
        consumed_update_ids = sorted(
            {
                int(record["update_id"])
                for record in records.values()
                if record.get("state") == "consumed" and record.get("update_id") is not None
            }
        )
        keep_count = max(0, int(keep_last_n_update_rollouts))
        keep_update_ids = set(consumed_update_ids[-keep_count:]) if keep_count else set()

        deleted: List[Path] = []
        for record in records.values():
            if record.get("state") != "consumed":
                continue
            update_id = record.get("update_id")
            if update_id is None or int(update_id) in keep_update_ids:
                continue
            path = Path(str(record.get("file_path", "")))
            if path.suffix != ".rollout" or not path.exists():
                continue
            path.unlink()
            deleted.append(path)
        return deleted


def select_ready_with_policy_lag(
    manifest: RolloutManifest,
    *,
    max_files: int,
    policy_version: int,
    update_id: int,
    max_policy_lag: Optional[int],
    stale_action: str,
    cleanup_stale: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    selected: List[Dict[str, Any]] = []
    stale_count = 0
    fresh_count = 0
    action = str(stale_action).lower()
    for record in manifest.records().values():
        if record.get("state") != "ready":
            continue
        if record.get("crashed"):
            continue
        path = Path(str(record.get("file_path", "")))
        if not path.exists():
            continue
        lag = int(policy_version) - int(record.get("policy_version", policy_version))
        if max_policy_lag is not None and lag > int(max_policy_lag):
            stale_count += 1
            if action == "drop":
                manifest.mark_dropped_stale(
                    str(path),
                    update_id=update_id,
                    current_policy_version=policy_version,
                    rollout_policy_version=int(record.get("policy_version", -1)),
                    lag=lag,
                )
                if cleanup_stale:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                continue
        else:
            fresh_count += 1
        selected.append(record)
        if len(selected) >= int(max_files):
            break
    if len(selected) < int(max_files):
        return [], stale_count, fresh_count
    return selected, stale_count, fresh_count


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def write_update_index(
    index_path: str,
    update_id: int,
    rollout_files: Sequence[str],
    *,
    sample_filter: Optional[Mapping[str, Any]] = None,
) -> Path:
    filter_cfg = dict(sample_filter or {})
    entries = []
    global_start = 0
    samples_before_filter = 0
    counts_by_outcome = {"success": 0, "failure": 0, "truncated": 0}
    expected_schema = None
    for source_path in rollout_files:
        header, header_crc = RolloutPackReader.read_header(source_path)
        metadata = dict(header.get("metadata") or {})
        outcome = str(metadata.get("outcome", "failure"))
        if outcome not in counts_by_outcome:
            raise ValueError(f"invalid RolloutPack outcome={outcome!r}: {source_path}")
        num_steps = int(header["num_steps"])
        samples_before_filter += num_steps
        window_key = "success_terminal_window_steps" if outcome == "success" else "failure_terminal_window_steps"
        window = _positive_int(filter_cfg.get(window_key, 0))
        retained_count = num_steps if window <= 0 or window >= num_steps else window
        retained_start = num_steps - retained_count
        fingerprint = str(header["schema_fingerprint"])
        if expected_schema is None:
            expected_schema = fingerprint
        elif fingerprint != expected_schema:
            raise ValueError(
                f"RolloutPack sample schemas differ within update: expected={expected_schema} "
                f"got={fingerprint} path={source_path}"
            )
        counts_by_outcome[outcome] += retained_count
        global_stop = global_start + retained_count
        entries.append(
            {
                "source_path": str(source_path),
                "file_size": int(Path(source_path).stat().st_size),
                "header_crc32": int(header_crc),
                "schema_fingerprint": fingerprint,
                "num_steps": num_steps,
                "outcome": outcome,
                "retained_start": retained_start,
                "retained_count": retained_count,
                "global_start": global_start,
                "global_stop": global_stop,
            }
        )
        global_start = global_stop
    if global_start <= 0:
        raise ValueError("update sample plan contains no retained samples")
    payload = {
        "format": "b2d-update-sample-plan",
        "version": 2,
        "update_id": int(update_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_fingerprint": expected_schema,
        "samples_before_filter": int(samples_before_filter),
        "samples_after_filter": int(global_start),
        "sample_counts_by_outcome": counts_by_outcome,
        "entries": entries,
    }
    return atomic_write_json(index_path, payload)
