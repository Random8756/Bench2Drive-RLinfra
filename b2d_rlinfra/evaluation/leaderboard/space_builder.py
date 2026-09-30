from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

from gymnasium import spaces

from .runtime_config import PROJECT_ROOT, resolve_project_path

from b2d_rlinfra.environment.spaces import (
    build_action_space as _build_action_space_from_yaml,
    build_observation_space_dict as _build_observation_space_dict_from_yaml,
)


def normalize_env_config_paths(env_config: Dict, base_dir: Path) -> Dict:
    config = deepcopy(env_config)
    obs_cfg = config.get("observation_space", {})

    for key in ("vector", "vis_bev"):
        section = obs_cfg.get(key)
        if not isinstance(section, dict):
            continue
        map_dir = section.get("map_dir")
        if map_dir:
            section["map_dir"] = str(
                resolve_project_path(
                    map_dir,
                    base_dir=base_dir,
                    project_root=PROJECT_ROOT,
                    allow_missing=True,
                )
            )

    return config


def build_action_space(env_config: Dict) -> spaces.Space:
    action_cfg = env_config.get("action_space", {})
    if action_cfg.get("type") == "trajectory":
        raise ValueError("trajectory action space is not supported by leaderboard eval")
    return _build_action_space_from_yaml(env_config)


def build_observation_space(env_config: Dict) -> spaces.Dict:
    obs_cfg = env_config.get("observation_space", {})
    vector_cfg = obs_cfg.get("vector", {})
    if not vector_cfg or not bool(vector_cfg.get("enable", False)):
        raise ValueError("observation_space.vector must be enabled")
    return _build_observation_space_dict_from_yaml(env_config)
