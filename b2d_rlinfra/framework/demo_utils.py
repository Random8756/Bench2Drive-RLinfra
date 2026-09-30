"""Utilities for the CARLA env rollout demo.

This module contains the support pieces used by env_rollout_demo.py: runtime
config parsing, action selection, compact console summaries, and a small toy
rollout buffer.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# =============================================================================
# 1. Runtime configuration
# =============================================================================
@dataclass
class DemoRuntimeConfig:
    config_path: Path
    env_config: Dict[str, Any]
    demo_config: Dict[str, Any]
    num_envs: int
    min_ready: int
    steps: int
    step_timeout: float
    reset_timeout: float
    manage_servers: bool
    max_episode_steps: int
    agent: "DemoAgent"


# =============================================================================
# 2. Demo agent
# =============================================================================
class DemoAgent:
    """Minimal policy-like object for choosing demo actions."""

    def __init__(self, env_config: Dict[str, Any], action_cfg: Dict[str, Any]) -> None:
        self.mode = str(action_cfg.get("mode", "fixed")).lower()
        self.index = int(action_cfg.get("index", 8))
        self.discrete_actions = self._load_discrete_actions(env_config)
        if self.mode not in {"fixed", "random"}:
            raise ValueError(f"Unsupported demo.action.mode={self.mode!r}")
        if not 0 <= self.index < len(self.discrete_actions):
            raise ValueError(f"demo.action.index={self.index} outside [0, {len(self.discrete_actions)})")

    def act(self, worker_id: int, observation: Any) -> int:
        del worker_id, observation
        if self.mode == "fixed":
            return self.index
        return random.randrange(len(self.discrete_actions))

    @staticmethod
    def _load_discrete_actions(env_config: Dict[str, Any]) -> List[Any]:
        action_cfg = env_config.get("action_space", {})
        if action_cfg.get("type") != "discrete":
            raise ValueError("env_rollout_demo currently expects env.action_space.type=discrete")
        actions = action_cfg.get("discrete_actions_list")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Discrete action_space requires a non-empty discrete_actions_list")
        return actions


# =============================================================================
# 3. Config loading
# =============================================================================
def load_demo_runtime_config(config_path: Path) -> DemoRuntimeConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    expected = {"env", "demo"}
    actual = set(cfg)
    if actual != expected:
        raise ValueError(f"Config top-level keys must be exactly {sorted(expected)}, got {sorted(actual)}")
    if not isinstance(cfg["env"], dict):
        raise ValueError("Config key 'env' must be a mapping")
    if not isinstance(cfg["demo"], dict):
        raise ValueError("Config key 'demo' must be a mapping")

    env_config = copy.deepcopy(cfg["env"])
    demo_config = copy.deepcopy(cfg["demo"])
    carla_cfg = env_config.get("carla", {})
    env_section = env_config.get("environment", {})

    return DemoRuntimeConfig(
        config_path=config_path,
        env_config=env_config,
        demo_config=demo_config,
        num_envs=int(carla_cfg.get("num_envs", 1)),
        min_ready=int(demo_config.get("min_ready", 1)),
        steps=int(demo_config.get("steps", 200)),
        step_timeout=float(demo_config.get("timeout", 30.0)),
        reset_timeout=float(demo_config.get("reset_timeout", 120.0)),
        manage_servers=bool(demo_config.get("manage_servers", True)),
        max_episode_steps=int(env_section.get("max_episode_steps", 10000)),
        agent=DemoAgent(env_config, demo_config.get("action", {})),
    )


# =============================================================================
# 4. Demo rollout buffer
# =============================================================================
class DemoRolloutBuffer:
    """Small worker-indexed transition buffer for episode lifecycle examples."""

    def __init__(self) -> None:
        self._episodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._episode_ids: Dict[int, int] = defaultdict(int)
        self._next_episode_start: Dict[int, bool] = {}
        self.committed_episodes = 0
        self.discarded_episodes = 0

    def start_episode(self, worker_id: int, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        self._episode_ids[worker_id] += 1
        self._episodes[worker_id] = []
        self._next_episode_start[worker_id] = True
        return {
            "episode_id": self._episode_ids[worker_id],
            "route": (info or {}).get("route_id") or "?",
            "town": (info or {}).get("town") or "?",
        }

    def append_step(
        self,
        worker_id: int,
        obs: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        transition = {
            "obs": obs,
            "action": action,
            "reward": float(reward),
            "episode_start": bool(self._next_episode_start.pop(worker_id, False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info or {}),
            "route_completed_ratio": (info or {}).get("route_completed_ratio"),
        }
        self._episodes[worker_id].append(transition)
        return {
            "buffered_steps": len(self._episodes[worker_id]),
            "last_reward": f"{float(reward):.3f}",
        }

    def finish_episode(self, worker_id: int) -> Dict[str, Any]:
        transitions = self._episodes.pop(worker_id, [])
        self._next_episode_start.pop(worker_id, None)
        total_reward = sum(float(item.get("reward", 0.0)) for item in transitions)
        self.committed_episodes += 1
        return {
            "committed_steps": len(transitions),
            "episode_reward": f"{total_reward:.3f}",
            "committed_episodes": self.committed_episodes,
        }

    def discard_episode(self, worker_id: int) -> Dict[str, Any]:
        transitions = self._episodes.pop(worker_id, [])
        self._next_episode_start.pop(worker_id, None)
        self.discarded_episodes += 1
        return {
            "discarded_steps": len(transitions),
            "discarded_episodes": self.discarded_episodes,
        }

    def buffered_steps(self, worker_id: int) -> int:
        return len(self._episodes.get(worker_id, []))

    def clear(self) -> None:
        self._episodes.clear()
        self._next_episode_start.clear()


# =============================================================================
# 5. Console output helpers
# =============================================================================
def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _shape(value: Any) -> str:
    if value is None:
        return "None"
    try:
        return str(tuple(np.asarray(value).shape))
    except Exception:
        return type(value).__name__


def observation_shape_summary(observation: Any) -> str:
    if not isinstance(observation, dict):
        return f"obs={type(observation).__name__}"
    keys = ("vector", "scalars", "bev_mask", "bev_image", "rgb")
    parts = [f"{key}={_shape(observation.get(key))}" for key in keys if key in observation]
    return " ".join(parts) if parts else f"obs_keys={sorted(observation.keys())}"


def route_context_text(info: Optional[Dict[str, Any]]) -> str:
    if not isinstance(info, dict):
        return "route=? scenario=? town=?"
    route = info.get("route_id") or info.get("route") or "?"
    scenario = info.get("scenario_name") or info.get("scenario_instance_name") or "?"
    town = info.get("town") or "?"
    return f"route={route} scenario={scenario} town={town}"


def print_reset_ready(worker_id: int, observation: Any, info: Optional[Dict[str, Any]]) -> None:
    reason = (info or {}).get("reset_reason") or "unknown"
    print(
        f"reset-ready worker={worker_id} reason={reason} "
        f"{route_context_text(info)} {observation_shape_summary(observation)}",
        flush=True,
    )


def print_worker_step(
    worker_id: int,
    observation: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Optional[Dict[str, Any]],
) -> None:
    rc = _to_float((info or {}).get("route_completed_ratio")) or 0.0
    print(
        f"step-result worker={worker_id} reward={float(reward):.3f} "
        f"term={terminated} trunc={truncated} rc={rc:.3f} "
        f"{observation_shape_summary(observation)}",
        flush=True,
    )


def print_crash_discard(worker_id: int, info: Optional[Dict[str, Any]], buffered_steps: int) -> None:
    crash_type = (info or {}).get("crash_type") or (info or {}).get("crash_reason") or "unknown"
    print(
        f"crash-discard worker={worker_id} type={crash_type} "
        f"discard_episode buffered_steps={buffered_steps}",
        flush=True,
    )


def print_terminal_observation(
    worker_id: int,
    observation: Any,
    terminated: bool,
    truncated: bool,
) -> None:
    print(
        f"terminal-observation worker={worker_id} "
        f"term={terminated} trunc={truncated} {observation_shape_summary(observation)}",
        flush=True,
    )
