from typing import Dict, List, Optional, Union

import numpy as np
import torch
from gymnasium import spaces


class ObsUtils:
    @staticmethod
    def get_obs_keys(
        observation_space: Optional[spaces.Space],
        obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[List[str]]:
        if isinstance(observation_space, spaces.Dict):
            return list(observation_space.spaces.keys())
        if obs is not None:
            return list(obs.keys())
        return None

    @staticmethod
    def stack_obs(
        obs_list: List[Union[np.ndarray, Dict[str, np.ndarray]]],
        keys: Optional[List[str]] = None,
        observation_space: Optional[spaces.Space] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if not obs_list:
            return np.array([])
        first_obs = obs_list[0]
        if isinstance(first_obs, dict):
            if keys is None:
                keys = ObsUtils.get_obs_keys(observation_space, first_obs)
            return {
                key: np.stack([obs[key] for obs in obs_list], axis=0)
                for key in (keys or [])
                if key in first_obs
            }
        return np.stack(obs_list, axis=0)

    @staticmethod
    def concat_obs(
        obs_list: List[Union[np.ndarray, Dict[str, np.ndarray]]],
        keys: Optional[List[str]] = None,
        observation_space: Optional[spaces.Space] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if not obs_list:
            return np.array([])
        first_obs = obs_list[0]
        if isinstance(first_obs, dict):
            if keys is None:
                keys = ObsUtils.get_obs_keys(observation_space, first_obs)
            return {
                key: np.concatenate([obs[key] for obs in obs_list], axis=0)
                for key in (keys or [])
                if key in first_obs
            }
        return np.concatenate(obs_list, axis=0)

    @staticmethod
    def empty_obs(
        obs_shapes: Union[Dict[str, tuple], tuple],
        keys: Optional[List[str]] = None,
        dtype: np.dtype = np.float32,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(obs_shapes, dict):
            if keys is None:
                keys = list(obs_shapes.keys())
            return {
                key: np.zeros((0,) + tuple(obs_shapes[key]), dtype=dtype)
                for key in (keys or [])
            }
        return np.zeros((0,) + tuple(obs_shapes), dtype=dtype)

    @staticmethod
    def get_obs_length(
        obs: Union[np.ndarray, Dict[str, np.ndarray], None],
        keys: Optional[List[str]] = None,
    ) -> int:
        if obs is None:
            return 0
        if isinstance(obs, dict):
            if keys is None:
                keys = list(obs.keys())
            if not keys:
                return 0
            return len(obs[keys[0]])
        return len(obs)

    @staticmethod
    def slice_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        indices: np.ndarray,
        keys: Optional[List[str]] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if isinstance(obs, dict):
            if keys is None:
                keys = list(obs.keys())
            return {
                key: obs[key][indices]
                for key in (keys or [])
                if key in obs
            }
        return obs[indices]

    @staticmethod
    def obs_to_tensor(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        device: Union[str, torch.device],
        keys: Optional[List[str]] = None,
        observation_space: Optional[spaces.Space] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(obs, dict):
            if keys is None:
                keys = ObsUtils.get_obs_keys(observation_space, obs)
            return {
                key: torch.as_tensor(obs[key], device=device).float()
                for key in (keys or [])
                if key in obs
            }
        return torch.as_tensor(obs, device=device).float()
