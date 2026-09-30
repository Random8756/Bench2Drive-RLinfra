"""Save successful on-policy trajectories for offline BC/value warmup."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .episode_success import classify_episode

logger = logging.getLogger("Training Loop")


_CONTEXT_KEYS = ("scenario_type", "scenario_name", "scenario_instance_name", "route_id", "town")


@dataclass
class _TrajectoryTransition:
    obs: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


def _safe_tag(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip() or default
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._")
    return text or default


def _compact_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep success/context signals while dropping large visualization payloads."""
    if not isinstance(info, Mapping):
        return {}
    compact = dict(info)
    compact.pop("bev_image", None)
    return compact


def _last_context(infos: List[Dict[str, Any]], terminal_info: Mapping[str, Any]) -> Dict[str, str]:
    context: Dict[str, str] = {}
    sources: List[Mapping[str, Any]] = [terminal_info] + list(reversed(infos))
    for key in _CONTEXT_KEYS:
        value = None
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            candidate = source.get(key)
            if candidate is not None and str(candidate).strip():
                value = candidate
                break
        context[key] = str(value).strip() if value is not None else ""
    return context


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + float(gamma) * running
        returns[idx] = running
    return returns


class SuccessTrajectoryRecorder:
    """Per-worker episode recorder that saves strict-success trajectories."""

    def __init__(
        self,
        *,
        output_dir: Path,
        gamma: float,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        compress: bool = True,
        strict_success_threshold: float = 99.9,
        max_success_trajectories: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.gamma = float(gamma)
        self.config_path = str(config_path or "")
        self.checkpoint_path = str(checkpoint_path or "")
        self.compress = bool(compress)
        self.strict_success_threshold = float(strict_success_threshold)
        self.max_success_trajectories = max(0, int(max_success_trajectories))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "index.jsonl"
        self._pending: Dict[int, List[_TrajectoryTransition]] = {}
        self._saved_counts_by_scenario: Dict[str, int] = {}
        self._discarded_count = 0

        logger.info(
            "[SuccessTrajectory] output_dir=%s max_success_trajectories_per_scenario=%d",
            self.output_dir,
            self.max_success_trajectories,
        )

    @property
    def saved_count(self) -> int:
        return int(sum(self._saved_counts_by_scenario.values()))

    def _scenario_saved_count(self, scenario_tag: str) -> int:
        return int(self._saved_counts_by_scenario.get(str(scenario_tag), 0))

    def _scenario_quota_reached(self, scenario_tag: str) -> bool:
        return (
            self.max_success_trajectories > 0
            and self._scenario_saved_count(scenario_tag) >= self.max_success_trajectories
        )

    def record(
        self,
        *,
        worker_id: int,
        obs: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        transition = _TrajectoryTransition(
            obs=obs,
            action=action,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=_compact_info(info),
        )
        self._pending.setdefault(int(worker_id), []).append(transition)

    def discard(self, worker_id: int) -> None:
        if self._pending.pop(int(worker_id), None):
            self._discarded_count += 1

    def reset(self) -> None:
        self._pending.clear()

    def flush_if_success(
        self,
        *,
        worker_id: int,
        terminal_info: Mapping[str, Any],
        episode_num: int,
    ) -> Optional[Path]:
        transitions = self._pending.pop(int(worker_id), None)
        if not transitions:
            return None

        info_sequence = [tr.info for tr in transitions]
        context = _last_context(info_sequence, terminal_info)
        scenario_tag = _safe_tag(context.get("scenario_name"), "unknown_scenario")
        if self._scenario_quota_reached(scenario_tag):
            return None

        classification = classify_episode(
            info_sequence,
            strict_success_threshold=self.strict_success_threshold,
        )
        if not classification["is_success"]:
            return None

        rewards = np.asarray([tr.reward for tr in transitions], dtype=np.float32)
        returns = _discounted_returns(rewards, self.gamma)
        actions_raw = np.asarray([tr.action for tr in transitions])
        if np.issubdtype(actions_raw.dtype, np.integer):
            actions = actions_raw.astype(np.int64, copy=False)
        else:
            actions = actions_raw.astype(np.float32, copy=False)
        terminateds = np.asarray([tr.terminated for tr in transitions], dtype=np.bool_)
        truncateds = np.asarray([tr.truncated for tr in transitions], dtype=np.bool_)
        episode_starts = np.zeros(len(transitions), dtype=np.bool_)
        if len(episode_starts) > 0:
            episode_starts[0] = True

        payload: Dict[str, np.ndarray] = {
            "actions": actions,
            "rewards": rewards,
            "returns": returns,
            "terminateds": terminateds,
            "truncateds": truncateds,
            "episode_starts": episode_starts,
        }
        payload.update(self._stack_observations([tr.obs for tr in transitions]))

        route_tag = _safe_tag(context.get("route_id"), "unknown_route")
        scenario_dir = self.output_dir / scenario_tag
        scenario_dir.mkdir(parents=True, exist_ok=True)

        episode_return = float(rewards.sum())
        steps = int(len(transitions))
        filename = (
            f"{route_tag}_ep_{int(episode_num):06d}_"
            f"steps_{steps}_return_{episode_return:.1f}.npz"
        )
        target_path = scenario_dir / filename
        tmp_path = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "wb") as f:
            if self.compress:
                np.savez_compressed(f, **payload)
            else:
                np.savez(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)

        record = {
            "file": str(target_path.relative_to(self.output_dir)),
            "scenario_type": context.get("scenario_type", ""),
            "scenario_name": context.get("scenario_name", ""),
            "scenario_instance_name": context.get("scenario_instance_name", ""),
            "route_id": context.get("route_id", ""),
            "town": context.get("town", ""),
            "steps": steps,
            "return": episode_return,
            "route_completed": float(classification["route_completed"]),
            "config": self.config_path,
            "checkpoint": self.checkpoint_path,
        }
        self._append_index(record)
        self._saved_counts_by_scenario[scenario_tag] = self._scenario_saved_count(scenario_tag) + 1
        logger.info(
            "[SuccessTrajectory] saved %s (scenario=%s saved_for_scenario=%d route=%s steps=%d return=%.1f rc=%.1f)",
            target_path,
            record["scenario_name"],
            self._scenario_saved_count(scenario_tag),
            record["route_id"],
            steps,
            episode_return,
            record["route_completed"],
        )
        return target_path

    def _append_index(self, record: Dict[str, Any]) -> None:
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _stack_observations(self, observations: List[Any]) -> Dict[str, np.ndarray]:
        if not observations:
            return {}
        first = observations[0]
        if isinstance(first, Mapping):
            payload: Dict[str, np.ndarray] = {}
            for key in first.keys():
                if key == "bev_image":
                    continue
                values = [
                    np.asarray(obs[key])
                    for obs in observations
                    if isinstance(obs, Mapping) and key in obs
                ]
                if len(values) != len(observations):
                    continue
                payload[f"obs_{key}"] = np.stack(values, axis=0)
            return payload
        return {"obs_vector": np.stack([np.asarray(obs) for obs in observations], axis=0)}
