"""Normalize Bench2Drive observations for DrivePi0 rl_finetune."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def unwrap_observation(obs: Any) -> Dict[str, Any]:
    if not isinstance(obs, Mapping):
        raise TypeError("DrivePi0 expects dict observations")
    if "observation" in obs and isinstance(obs["observation"], Mapping):
        return dict(obs["observation"])
    return dict(obs)


def require_drivepi0_obs(
    obs: Any,
    *,
    rgb_key: str,
    state_key: str,
) -> Dict[str, Any]:
    mapping = unwrap_observation(obs)
    if rgb_key not in mapping:
        raise KeyError(f"DrivePi0 observation requires RGB key {rgb_key!r}")
    if state_key not in mapping:
        raise KeyError(f"DrivePi0 observation requires state key {state_key!r}")
    return mapping
