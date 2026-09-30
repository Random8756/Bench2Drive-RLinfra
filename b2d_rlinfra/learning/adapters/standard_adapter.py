"""Thin adapter over ``CARLAEnvPool`` for worker-indexed async batches."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from gymnasium import spaces

__layer__ = (4, "Algorithm")


class StandardEnvAdapter:
    """Expose pool reset/step results as ``{worker_id: value}`` dicts."""

    def __init__(
        self,
        pool,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        default_timeout: float = 30.0,
        obs_key: Optional[str] = 'vector',
    ):
        self.pool = pool
        self.default_timeout = default_timeout
        self.obs_key = obs_key

        self.observation_space = observation_space or getattr(pool, 'observation_space', None)
        self.action_space = action_space or getattr(pool, 'action_space', None)
        self.num_envs = pool.num_envs

        # Raw dict observations are useful for visualization/debugging.
        self._raw_observations: Dict[int, Dict] = {}

    def _process_outputs(
        self,
        obs_dict: Dict[int, Any],
        reward_dict: Dict[int, float],
        term_dict: Dict[int, bool],
        trunc_dict: Dict[int, bool],
        info_dict: Dict[int, Dict],
    ) -> Tuple[
        Dict[int, Any],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, Dict],
    ]:
        processed_obs = {}
        processed_info = {wid: dict(info) for wid, info in info_dict.items()}

        for wid, obs in obs_dict.items():
            if isinstance(obs, dict):
                self._raw_observations[wid] = obs
            processed_obs[wid] = self._extract_obs(obs)
            info = processed_info.setdefault(wid, {})
            bev_image = self._extract_bev_image(obs, info)
            if bev_image is not None:
                info["bev_image"] = bev_image

        return processed_obs, reward_dict, term_dict, trunc_dict, processed_info

    def _extract_obs(self, obs: Union[Dict, np.ndarray]) -> Any:
        """Return the configured observation branch, or the full dict."""
        if isinstance(obs, dict):
            if self.obs_key is None:
                return obs
            if self.obs_key in obs:
                return np.asarray(obs[self.obs_key])
            elif 'bev_mask' in obs:
                return np.asarray(obs['bev_mask'])
            else:
                for v in obs.values():
                    if isinstance(v, (np.ndarray, list)):
                        return np.asarray(v)
                raise ValueError(f"Cannot extract observation from dict: {obs.keys()}")
        return np.asarray(obs)

    def _extract_bev_image(
        self,
        obs: Union[Dict, np.ndarray],
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Return BEV visualization image when the env emitted one."""
        if isinstance(info, dict):
            bev_image = info.get("bev_image")
            if bev_image is not None:
                return np.asarray(bev_image)
        if isinstance(obs, dict):
            bev_image = obs.get("bev_image")
            if bev_image is not None:
                return np.asarray(bev_image)
        return None

    def reset(
        self,
        min_ready: Optional[int] = None,
        timeout: Optional[float] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[int, Any], Dict[int, Dict]]:
        """Reset workers and return the first ready observation batch."""
        if min_ready is None:
            min_ready = self.num_envs
        if timeout is None:
            timeout = self.default_timeout * 2

        obs_dict, info_dict = self.pool.reset(
            seed=seed,
            options=options,
            min_ready=min_ready,
            timeout=timeout,
        )

        processed_obs, _, _, _, processed_info = self._process_outputs(obs_dict, {}, {}, {}, info_dict)
        return processed_obs, processed_info

    def step(
        self,
        actions: Dict[int, Any],
        min_ready: int = 1,
        timeout: Optional[float] = None,
    ) -> Tuple[
        Dict[int, Any],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, Dict],
    ]:
        """Dispatch actions and collect the next ready worker batch."""
        if timeout is None:
            timeout = self.default_timeout

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.pool.step(
            actions=actions,
            min_ready=min_ready,
            timeout=timeout,
        )

        return self._process_outputs(obs_dict, reward_dict, term_dict, trunc_dict, info_dict)

    def close(self) -> None:
        """Close the environment pool."""
        if hasattr(self.pool, 'close'):
            self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
