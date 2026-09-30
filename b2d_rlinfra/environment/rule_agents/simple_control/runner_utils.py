"""Runner utilities for the LQR-based simple-control agent."""

import json
import os
import signal
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from gymnasium import spaces

PROJECT_ROOT = str(Path(__file__).resolve().parents[4])

from .lqr_controller import LQRController

logger = logging.getLogger("Training Loop")


def load_lqr_config(config_path: str) -> dict:
    """Load LQR YAML config and return a plain dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


class LQREnvWrapper:
    """Run :class:`LQRController` inside a CARLAEnvPool worker."""

    def __init__(self, env, lqr_config: dict):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._controller = None
        self._lqr_config = lqr_config

        # Recording
        rec_cfg = lqr_config.get('recording', {})
        self.recording_enabled = rec_cfg.get('enabled', True)

        base_dir = lqr_config.get('evaluation', {}).get('log_dir', './logs/simple_control_lqr')
        worker_id = getattr(env, 'env_index', getattr(env, '_env_index', 0))
        ts = lqr_config.get('_experiment_timestamp',
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
        self._transitions_dir = Path(base_dir) / ts / 'transitions' / f'worker_{worker_id}'
        if self.recording_enabled:
            self._transitions_dir.mkdir(parents=True, exist_ok=True)

        self._episode_actions: List[np.ndarray] = []
        self._episode_rewards: List[float] = []
        self._episode_terminateds: List[bool] = []
        self._episode_truncateds: List[bool] = []
        self._episode_observations: List[Any] = []
        self._episode_count = 0
        self._last_obs = None
        self._episode_route_id: Optional[str] = None

    def __getattr__(self, name):
        if name == 'env':
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if self.recording_enabled and self._episode_actions:
            self._save_episode()

        if self._controller is None:
            lqr_cfg = self._lqr_config.get('lqr', {})
            self._controller = LQRController(
                action_space=self.action_space,
                config=lqr_cfg,
                visualize=lqr_cfg.get('visualize', False),
            )
        self._controller.reset()

        self._episode_actions = []
        self._episode_rewards = []
        self._episode_terminateds = []
        self._episode_truncateds = []
        self._episode_observations = []
        self._last_obs = obs
        self._episode_route_id = info.get('route_id', None)

        return obs, info

    def step(self, dummy_action):
        """Run one step using the LQR controller (ignoring dummy_action)."""
        lqr_action = self._controller(self._last_obs)

        obs, reward, terminated, truncated, info = self.env.step(lqr_action)
        info['expert_action'] = lqr_action.tolist()

        if self.recording_enabled:
            self._episode_observations.append(self._snapshot_obs(self._last_obs))
            self._episode_actions.append(lqr_action.copy())
            self._episode_rewards.append(float(reward))
            self._episode_terminateds.append(bool(terminated))
            self._episode_truncateds.append(bool(truncated))

        self._last_obs = obs

        if (terminated or truncated) and self.recording_enabled and self._episode_actions:
            self._episode_observations.append(self._snapshot_obs(obs))
            self._save_episode()

        return obs, reward, terminated, truncated, info

    def _save_episode(self):
        if not self._episode_actions:
            return
        ep_id = self._episode_count
        self._episode_count += 1
        route_tag = f'_{self._episode_route_id}' if self._episode_route_id else ''
        filename = f'episode_{ep_id:05d}{route_tag}'

        T = len(self._episode_actions)
        data = {
            'actions': np.array(self._episode_actions, dtype=np.float32),
            'rewards': np.array(self._episode_rewards, dtype=np.float32),
            'dones': np.array(
                [t or tr for t, tr in zip(self._episode_terminateds, self._episode_truncateds)],
                dtype=bool,
            ),
            'terminateds': np.array(self._episode_terminateds, dtype=bool),
            'truncateds': np.array(self._episode_truncateds, dtype=bool),
        }

        if self._episode_observations:
            first = self._episode_observations[0]
            if isinstance(first, dict):
                for key in first.keys():
                    data[f'obs_{key}'] = np.array(
                        [o[key] for o in self._episode_observations],
                        dtype=first[key].dtype,
                    )
            else:
                data['observations'] = np.array(self._episode_observations)

        filepath = self._transitions_dir / f'{filename}.npz'
        try:
            np.savez_compressed(str(filepath), **data)
            logger.info(f'Saved transitions: {filepath}  (steps={T})')
        except Exception as e:
            logger.error(f'Failed to save transitions: {e}')

        meta = {
            'episode_id': ep_id,
            'route_id': self._episode_route_id,
            'num_steps': T,
            'total_reward': float(sum(self._episode_rewards)),
            'terminated': bool(any(self._episode_terminateds)),
            'truncated': bool(any(self._episode_truncateds)),
            'timestamp': datetime.now().isoformat(),
        }
        meta_path = self._transitions_dir / f'{filename}.json'
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_terminateds.clear()
        self._episode_truncateds.clear()
        self._episode_observations.clear()

    @staticmethod
    def _snapshot_obs(obs):
        if isinstance(obs, dict):
            return {k: np.asarray(v).copy() for k, v in obs.items()}
        return np.asarray(obs).copy()


def make_lqr_env(config: dict, worker_id: int, carla_port: int,
                 traffic_manager_port: int):
    """
    Environment factory for CARLAEnvPool — creates the full wrapper chain
    plus the LQREnvWrapper on top.
    """
    from b2d_rlinfra.environment.carla_env import CARLAEnv
    from b2d_rlinfra.environment.wrappers import (
        ActionWrapper,
        ObservationWrapper,
        EventTerminationWrapper,
        RoutePlanWrapper,
        RewardWrapper,
    )

    env = CARLAEnv(config, env_index=worker_id, port=carla_port,
                    traffic_manager_port=traffic_manager_port)
    env = RoutePlanWrapper(env)
    env = ObservationWrapper(env)
    env = EventTerminationWrapper(env)
    env = RewardWrapper(env)
    env = ActionWrapper(env)

    full_config = config.get('_lqr_full_config', config)
    env = LQREnvWrapper(env, full_config)

    return env


def build_observation_space(env_config: dict) -> spaces.Space:
    """Build observation space from env config.

    Delegated to ``b2d_rlinfra.environment.spaces`` (single yaml-driven source of truth).
    """
    from b2d_rlinfra.environment.spaces import build_observation_space as _build_observation_space
    return _build_observation_space(env_config)


def build_action_space(env_config: dict) -> spaces.Space:
    """Build action space from env config.

    Delegated to ``b2d_rlinfra.environment.spaces`` (single yaml-driven source of truth).
    """
    from b2d_rlinfra.environment.spaces import build_action_space as _build_action_space
    return _build_action_space(env_config)


def get_carla_hosts_ports(env_config: dict):
    carla_cfg = env_config.get('carla', {})
    ports = carla_cfg.get('port', [])
    hosts = carla_cfg.get('host', '127.0.0.1')
    if not isinstance(hosts, list):
        hosts = [hosts] * max(len(ports), 1)
    while len(hosts) < len(ports):
        hosts.append(hosts[-1])
    timeout = float(carla_cfg.get('server_wait_timeout', 120.0))
    return hosts, ports, timeout


def create_env_pool_and_adapter(
    env_config: dict,
    observation_space,
    action_space,
    adapter_timeout: float,
    min_ready: int,
    full_config: dict,
):
    from b2d_rlinfra.simulation.runners.carla_env_pool import CARLAEnvPool

    carla_config = env_config.get('carla', {})
    num_envs = carla_config.get('num_envs', 1)
    env_section = env_config.get('environment', {})
    max_episode_steps = env_section.get('max_episode_steps', 10000)

    env_config_copy = dict(env_config)
    env_config_copy['_lqr_full_config'] = full_config

    pool = CARLAEnvPool(
        env_fn=make_lqr_env,
        config=env_config_copy,
        num_envs=num_envs,
        auto_reset=True,
        manage_servers=True,
        max_episode_steps=max_episode_steps,
    )
    print(f'  num_envs: {num_envs}  max_episode_steps: {max_episode_steps}')

    print('  Waiting for environments ...')
    t0 = time.time()
    while time.time() - t0 < 180.0:
        stats = pool.get_stats()
        if stats.get('active_workers', 0) >= num_envs:
            break
        time.sleep(2.0)
    stats = pool.get_stats()
    print(f'  Active: {stats.get("active_workers", 0)}/{num_envs}')

    from b2d_rlinfra.learning.adapters.standard_adapter import StandardEnvAdapter
    adapter = StandardEnvAdapter(
        pool=pool,
        observation_space=observation_space,
        action_space=action_space,
        default_timeout=adapter_timeout,
        obs_key=None,
    )
    print(f'  Adapter: StandardEnvAdapter  min_ready={min_ready}  timeout={adapter_timeout}s')
    return pool, adapter


def create_visualizer(vis_cfg: dict, log_dir: Path):
    if not vis_cfg.get('enabled', True):
        return None
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer
    return TrainingVisualizer(
        output_dir=str(log_dir / 'videos'),
        fps=vis_cfg.get('fps', 10),
        save_interval_episodes=vis_cfg.get('save_interval', 5),
        max_episodes_to_keep=vis_cfg.get('max_videos', 200),
        lazy_capture=vis_cfg.get('lazy_capture', True),
        overlay_info=vis_cfg.get('overlay_info', True),
        enabled=True,
    )


def create_run_dir(base_log_dir: str, prefix: str = 'lqr') -> Path:
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f'{prefix}_{ts}' if prefix else ts
    d = Path(base_log_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_config(config: dict, log_dir: Path):
    with open(log_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_total_routes_from_config(env_config: dict) -> int:
    routes_cfg = env_config.get('routes', {})
    routes_file = routes_cfg.get('route_files', ['resources/routes/train_routes_demo.xml'])
    if isinstance(routes_file, list):
        routes_file = routes_file[0] if routes_file else 'resources/routes/train_routes_demo.xml'
    from leaderboard.utils.route_indexer import RouteIndexer
    sampler = RouteIndexer(
        routes_file=routes_file,
        repetitions=routes_cfg.get('repetitions', 1),
        routes_subset=routes_cfg.get('routes_subset', None),
        warmup=routes_cfg.get('warmup', False),
        training=routes_cfg.get('training', True),
    )
    return sampler.get_length()


class CleanupManager:
    def __init__(self):
        self.pool = None
        self.log_dir = None
        self.visualizer = None
        self.carla_hosts = None
        self._done = False

    def register(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, k, v)

    def cleanup(self, force_kill: bool = False):
        if self._done:
            return
        self._done = True
        print('\n' + '=' * 70)
        print('  Cleaning up ...')
        print('=' * 70)
        if self.visualizer:
            try:
                self.visualizer.close()
            except Exception:
                pass
            self.visualizer = None
        if self.pool:
            try:
                self.pool.close()
            except Exception:
                pass
            self.pool = None
        print('  Done.')
        print('=' * 70)


def install_signal_handlers(cleanup_mgr: CleanupManager):
    def _handler(signum, frame):
        print('\n\n  Interrupt received ...')
        cleanup_mgr.cleanup(force_kill=True)
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
