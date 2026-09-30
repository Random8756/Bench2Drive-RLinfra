"""Shared runner utilities for RL training and evaluation entry points."""

import importlib
import logging
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from gymnasium import spaces

__layer__ = (4, "Algorithm")

logger = logging.getLogger("Training Loop")


PROJECT_ROOT_DIR = str(Path(__file__).resolve().parents[3])


def kill_carla_processes(
    carla_hosts: Optional[List[str]] = None,
    verbose: bool = True,
) -> int:
    if not carla_hosts:
        return 0
    hosts = [h for h in dict.fromkeys(carla_hosts) if h]
    if not hosts:
        return 0

    script_path = Path(PROJECT_ROOT_DIR) / "tools" / "runtime" / "kill_by_host.sh"
    if not script_path.exists():
        logger.warning("kill_by_host script not found: %s", script_path)
        return 0

    killed = 0
    if verbose:
        logger.info("Killing CARLA processes by host hosts=%d", len(hosts))
    for host in hosts:
        logger.debug("kill_by_host.sh %s", host)
        try:
            result = subprocess.run(
                ["bash", str(script_path), host],
                capture_output=True, text=True,
            )
            output = (result.stdout or "") + (result.stderr or "")
            killed += sum(1 for l in output.splitlines() if l.strip().startswith("PID:"))
            if output.strip():
                logger.debug("kill_by_host output host=%s:\n%s", host, output.rstrip())
            if result.returncode != 0:
                logger.warning("kill_by_host failed host=%s rc=%s", host, result.returncode)
        except Exception as e:
            logger.warning("kill_by_host failed host=%s: %s", host, e)
    if killed > 0:
        time.sleep(2.0)
    if verbose:
        logger.info("CARLA host cleanup finished killed=%d", killed)
    return killed


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def wait_for_ports_free(
    hosts: List[str],
    ports: List[int],
    timeout: float = 60.0,
    interval: float = 2.0,
) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        busy = [(h, p) for h, p in zip(hosts, ports) if _is_port_open(h, p)]
        if not busy:
            return True
        time.sleep(interval)
    return False


def wait_for_carla_servers_ready(
    hosts: List[str],
    ports: List[int],
    timeout: float = 120.0,
    interval: float = 2.0,
) -> bool:
    try:
        import carla
    except Exception:
        return True
    start = time.time()
    pending = list(zip(hosts, ports))
    while time.time() - start < timeout:
        still = []
        for host, port in pending:
            try:
                client = carla.Client(host, port)
                client.set_timeout(2.0)
                if client.get_server_version():
                    continue
            except Exception:
                pass
            still.append((host, port))
        if not still:
            return True
        pending = still
        time.sleep(interval)
    return False


# Built-ins may be overridden with a custom ``class_path`` using the same
# ``(env, config)`` constructor contract.
_MODEL_INTEGRATION_WRAPPERS = {
    "minddrive_route": "b2d_rlinfra.environment.model_integrations.minddrive_route.MindDriveRouteContextWrapper",
    "drivepi0_route": "b2d_rlinfra.environment.model_integrations.drivepi0_route.DrivePi0RouteContextWrapper",
}


def _resolve_model_integration_wrapper(name: str, integration_cfg: Dict):
    class_path = integration_cfg.get("class_path") or _MODEL_INTEGRATION_WRAPPERS.get(name)
    if not class_path:
        known = ", ".join(sorted(_MODEL_INTEGRATION_WRAPPERS))
        raise ValueError(
            f"Unknown model integration {name!r}; built-ins are: {known}. "
            "Set class_path to a custom wrapper class for other integrations."
        )
    module_name, _, class_name = str(class_path).rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid model integration class_path: {class_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_env(
    config: Dict,
    worker_id: int,
    carla_port: int,
    traffic_manager_port: int,
    adaptive_shared_stats=None,
    adaptive_shared_lock=None,
):
    """Environment factory function for CARLAEnvPool."""
    from b2d_rlinfra.environment.carla_env import CARLAEnv
    from b2d_rlinfra.environment.wrappers import (
        ActionWrapper,
        ObservationWrapper,
        EventTerminationWrapper,
        RoutePlanWrapper,
        RewardWrapper,
    )
    env = CARLAEnv(
        config,
        env_index=worker_id,
        port=carla_port,
        traffic_manager_port=traffic_manager_port,
        adaptive_shared_stats=adaptive_shared_stats,
        adaptive_shared_lock=adaptive_shared_lock,
    )
    env = RoutePlanWrapper(env)
    for integration_name, integration_cfg in (config.get("model_integrations", {}) or {}).items():
        integration_cfg = dict(integration_cfg or {})
        if not bool(integration_cfg.get("enable", False)):
            continue
        wrapper_cls = _resolve_model_integration_wrapper(str(integration_name), integration_cfg)
        env = wrapper_cls(env, integration_cfg)
    env = ObservationWrapper(env)
    env = EventTerminationWrapper(env)
    env = RewardWrapper(env)
    env = ActionWrapper(env)

    # Optionally add LQR expert wrapper for BC with bc_source='lqr'
    algo_config = config.get('algorithm', {})
    bc_source = algo_config.get('bc_source', 'fixed')
    if bc_source == 'lqr':
        from b2d_rlinfra.environment.wrappers import LQRExpertWrapper
        lqr_config = algo_config.get('lqr_config', {})
        env = LQRExpertWrapper(env, lqr_config=lqr_config)

    return env


def build_observation_space(env_config: Dict) -> spaces.Space:
    """Build the observation space for the policy network.

    Delegated to ``b2d_rlinfra.environment.spaces`` so that the wrapper-runtime view and the
    policy-bring-up view of the obs space are guaranteed to agree (both
    derived from the same yaml).
    """
    from b2d_rlinfra.environment.spaces import build_observation_space as _build_observation_space
    return _build_observation_space(env_config)


def build_action_space(env_config: Dict) -> spaces.Space:
    """Build the action space for the policy network.

    Delegated to ``b2d_rlinfra.environment.spaces`` (single source of truth, yaml-driven).
    """
    from b2d_rlinfra.environment.spaces import build_action_space as _build_action_space
    return _build_action_space(env_config)


def _spaces_compatible(expected: spaces.Space, actual: spaces.Space, name: str = "space"):
    if isinstance(expected, spaces.Box) and isinstance(actual, spaces.Dict):
        actual_keys = list(actual.spaces.keys())
        if actual_keys == ["vector"]:
            return _spaces_compatible(expected, actual.spaces["vector"], name=f"{name}.vector")

    if isinstance(expected, spaces.Dict) and isinstance(actual, spaces.Box):
        expected_keys = list(expected.spaces.keys())
        if expected_keys == ["vector"]:
            return _spaces_compatible(expected.spaces["vector"], actual, name=f"{name}.vector")

    if type(expected) is not type(actual):
        return False, f"{name} type mismatch: expected {type(expected).__name__}, got {type(actual).__name__}"

    if isinstance(expected, spaces.Discrete):
        if expected.n != actual.n:
            return False, f"{name}.n mismatch: expected {expected.n}, got {actual.n}"
        return True, ""

    if isinstance(expected, spaces.Box):
        if tuple(expected.shape) != tuple(actual.shape):
            return False, f"{name}.shape mismatch: expected {expected.shape}, got {actual.shape}"
        if expected.dtype != actual.dtype:
            return False, f"{name}.dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        if not np.allclose(expected.low, actual.low, atol=1e-6):
            return False, f"{name}.low mismatch"
        if not np.allclose(expected.high, actual.high, atol=1e-6):
            return False, f"{name}.high mismatch"
        return True, ""

    if isinstance(expected, spaces.Dict):
        exp_keys = set(expected.spaces.keys())
        act_keys = set(actual.spaces.keys())
        if exp_keys != act_keys:
            return False, f"{name}.keys mismatch: expected {sorted(exp_keys)}, got {sorted(act_keys)}"
        for key in sorted(exp_keys):
            ok, msg = _spaces_compatible(
                expected.spaces[key], actual.spaces[key], name=f"{name}.{key}"
            )
            if not ok:
                return False, msg
        return True, ""

    # Fallback for uncommon space types
    if repr(expected) != repr(actual):
        return False, f"{name} repr mismatch: expected {expected!r}, got {actual!r}"
    return True, ""


def create_env_pool_and_adapter(
    env_config: Dict,
    observation_space,
    action_space,
    adapter_timeout: float,
    min_ready: int,
    obs_key: Optional[str],
    algorithm_config: Optional[Dict] = None,
):
    """Build the L3 ``CARLAEnvPool`` and the L4 ``StandardEnvAdapter``.

    Returns
    -------
    tuple
        ``(l3_pool, l4_adapter)`` where ``l3_pool`` is the Simulation-Layer
        async parallel CARLA worker pool and ``l4_adapter`` is the
        Algorithm-Layer adapter bridging workers and the learner.
    """
    from b2d_rlinfra.simulation.runners.carla_env_pool import CARLAEnvPool

    carla_config = env_config.get('carla', {})
    num_envs = carla_config.get('num_envs', 1)
    env_section = env_config.get('environment', {})
    max_episode_steps = env_section.get('max_episode_steps', 10000)
    worker_config = dict(env_config)
    if algorithm_config:
        worker_config['algorithm'] = dict(algorithm_config)

    # ── L3: spawn the async parallel worker pool ───────────────────────
    l3_pool = CARLAEnvPool(
        env_fn=make_env, config=worker_config, num_envs=num_envs,
        auto_reset=True, manage_servers=True,
        max_episode_steps=max_episode_steps,
    )
    print(f"  num_envs: {num_envs}  max_episode_steps: {max_episode_steps}")

    print("  Waiting for environments ...")
    t0 = time.time()
    while time.time() - t0 < 180.0:
        stats = l3_pool.get_stats()
        if stats.get('ready_workers', 0) >= num_envs:
            break
        time.sleep(2.0)
    stats = l3_pool.get_stats()
    print(
        f"  Ready: {stats.get('ready_workers', 0)}/{num_envs}  "
        f"Active: {stats.get('active_workers', 0)}/{num_envs}"
    )

    # ── Cross-layer space sanity check (L2 vs L3 view) ─────────────────
    pool_obs_space = l3_pool.observation_space
    pool_action_space = l3_pool.action_space
    if pool_obs_space is not None:
        ok, msg = _spaces_compatible(observation_space, pool_obs_space, name="observation_space")
        if not ok:
            raise ValueError(f"Runner/Env observation space mismatch: {msg}")
    if pool_action_space is not None:
        ok, msg = _spaces_compatible(action_space, pool_action_space, name="action_space")
        if not ok:
            raise ValueError(f"Runner/Env action space mismatch: {msg}")

    # ── L4: build the environment adapter over the L3 pool ─────────────
    from b2d_rlinfra.learning.adapters.standard_adapter import StandardEnvAdapter
    l4_adapter = StandardEnvAdapter(
        pool=l3_pool, observation_space=observation_space,
        action_space=action_space, default_timeout=adapter_timeout,
        obs_key=obs_key,
    )
    print(f"  Adapter: StandardEnvAdapter  min_ready={min_ready}")
    print(f"  timeout={adapter_timeout}s  num_envs={l4_adapter.num_envs}")
    return l3_pool, l4_adapter


class CleanupManager:
    def __init__(self):
        self.pool = None
        self.model = None
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
        logger.info("Cleaning up training resources")

        if self.pool:
            try:
                self.pool.close()
            except Exception:
                pass
            self.pool = None

        if self.visualizer:
            try:
                self.visualizer.close()
            except Exception:
                pass
            self.visualizer = None

        if self.model:
            shutdown_train = getattr(self.model, "_shutdown_train_process", None)
            if callable(shutdown_train):
                try:
                    shutdown_train()
                except Exception:
                    pass

            replay_buffer = getattr(self.model, "replay_buffer", None)
            if replay_buffer is not None and hasattr(replay_buffer, "cleanup"):
                try:
                    replay_buffer.cleanup()
                except Exception:
                    pass

            model_logger = getattr(self.model, "logger", None)
            if model_logger is not None and hasattr(model_logger, "close"):
                try:
                    model_logger.close()
                except Exception:
                    pass

        if force_kill:
            kill_carla_processes(carla_hosts=self.carla_hosts, verbose=True)

        logger.info("Cleanup complete")

    def reset_for_next_segment(self):
        """Reset pool/visualizer refs for next training segment (keeps model)."""
        self.pool = None
        self.visualizer = None
        self._done = False


def install_signal_handlers(cleanup_mgr: CleanupManager):
    def _handler(signum, frame):
        logger.warning("Interrupt received, cleaning up")
        cleanup_mgr.cleanup(force_kill=True)
        sys.exit(1)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def get_feature_extractor(config):
    """Return (fe_class, fe_kwargs, obs_key, policy_kwargs)."""
    from b2d_rlinfra.learning.policies.feature_extractor import CNNFeatureExtractor, CombinedExtractor

    fe_type = config.policy.features_extractor_type.lower()

    if fe_type == "combined_v2":
        from b2d_rlinfra.learning.policies.feature_extractor_v2 import CombinedExtractorV2
        fe_class = CombinedExtractorV2
        fe_kwargs = {
            'features_dim': config.policy.features_dim,
            'use_layer_norm': config.policy.use_layer_norm,
            'cnn_channels': config.policy.cnn_channels,
            'state_neurons': config.policy.state_neurons,
            'fusion_dims': config.policy.fusion_dims,
        }
        obs_key = None
    elif fe_type == "combined_v3_rgb":
        from b2d_rlinfra.learning.policies.feature_extractor_v3_rgb import CombinedExtractorV3RGB
        fe_class = CombinedExtractorV3RGB
        fe_kwargs = {
            'features_dim': config.policy.features_dim,
            'use_layer_norm': config.policy.use_layer_norm,
            'cnn_channels': config.policy.cnn_channels,
            'state_neurons': config.policy.state_neurons,
            'fusion_dims': config.policy.fusion_dims,
            'rgb_camera_feature_dim': config.policy.rgb_camera_feature_dim,
        }
        obs_key = None
    elif fe_type == "combined":
        fe_class = CombinedExtractor
        fe_kwargs = {'features_dim': config.policy.features_dim}
        obs_key = None
    else:
        fe_class = CNNFeatureExtractor
        fe_kwargs = {'features_dim': config.policy.features_dim}
        obs_key = 'vector'

    policy_kwargs = {
        'features_extractor_class': fe_class,
        'features_extractor_kwargs': fe_kwargs,
    }

    uses_actor_critic_v2 = fe_type in ("combined_v2", "combined_v3_rgb")

    if uses_actor_critic_v2:
        policy_kwargs['use_layer_norm_policy_head'] = config.policy.use_layer_norm_policy_head

    algo_name = config.algorithm.name.lower()
    if algo_name in ('ppo', 'a2c'):
        policy_kwargs['net_arch'] = {
            'pi': config.policy.net_arch_pi,
            'vf': config.policy.net_arch_vf,
        }
        if uses_actor_critic_v2:
            policy_kwargs['value_head_type'] = config.policy.value_head_type
            policy_kwargs['value_num_bins'] = config.policy.value_num_bins
            policy_kwargs['value_support_min'] = config.policy.value_support_min
            policy_kwargs['value_support_max'] = config.policy.value_support_max
            policy_kwargs['value_transform'] = config.policy.value_transform
            policy_kwargs['action_distribution'] = config.policy.action_distribution
            policy_kwargs['beta_min_a_b_value'] = config.policy.beta_min_a_b_value
            policy_kwargs['beta_epsilon'] = config.policy.beta_epsilon
            policy_kwargs['beta_deterministic_action'] = config.policy.beta_deterministic_action
    else:
        policy_kwargs['net_arch'] = {
            'pi': config.policy.net_arch_pi,
            'qf': config.policy.net_arch_qf,
        }

    # Algorithm-specific extra policy kwargs
    if algo_name == 'sac':
        policy_kwargs['log_std_init'] = getattr(config.policy, 'log_std_init', -3.0)

    if algo_name in ('td3', 'sac'):
        # Mixture-of-Experts sub-configs (actor / critic are independent).
        policy_kwargs['moe_actor'] = {
            'enabled': getattr(config.policy, 'moe_actor_enabled', False),
            'num_experts': getattr(config.policy, 'moe_actor_num_experts', 4),
            'top_k': getattr(config.policy, 'moe_actor_top_k', 2),
            'noisy_gating': getattr(config.policy, 'moe_actor_noisy_gating', True),
        }
        policy_kwargs['moe_critic'] = {
            'enabled': getattr(config.policy, 'moe_critic_enabled', False),
            'num_experts': getattr(config.policy, 'moe_critic_num_experts', 4),
            'top_k': getattr(config.policy, 'moe_critic_top_k', 2),
            'noisy_gating': getattr(config.policy, 'moe_critic_noisy_gating', True),
        }

    # Distributional Q-value (two-hot encoding) — off-policy only
    if algo_name in ('td3', 'sac'):
        use_distributional = getattr(config.policy, 'use_distributional', False)
        if use_distributional:
            policy_kwargs['use_distributional'] = True
            policy_kwargs['num_bins'] = getattr(config.policy, 'num_bins', 255)
            policy_kwargs['v_min'] = getattr(config.policy, 'v_min', -300.0)
            policy_kwargs['v_max'] = getattr(config.policy, 'v_max', 800.0)
            policy_kwargs['use_symlog'] = getattr(config.policy, 'use_symlog', True)

    return fe_class, fe_kwargs, obs_key, policy_kwargs


def create_visualizer(config, log_dir: Path,
                      initial_step: int = 0, initial_episode: int = 0):
    vis_cfg = config.visualization
    if not vis_cfg.enabled:
        return None
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer
    return TrainingVisualizer(
        output_dir=str(log_dir / "videos"),
        fps=vis_cfg.fps,
        save_interval_episodes=vis_cfg.save_interval,
        max_episodes_to_keep=vis_cfg.max_videos,
        lazy_capture=vis_cfg.lazy_capture,
        overlay_info=vis_cfg.overlay_info,
        enabled=True,
        initial_step=initial_step,
        initial_episode=initial_episode,
    )


def get_carla_hosts_ports(env_config: Dict):
    carla_cfg = env_config.get('carla', {})
    num_envs = int(carla_cfg.get('num_envs', 1) or 1)
    ports = carla_cfg.get('port', [])
    hosts = carla_cfg.get('host', '127.0.0.1')

    if isinstance(ports, list):
        ports = list(ports[:num_envs])
    elif ports:
        ports = [ports]
    else:
        ports = []

    if isinstance(hosts, list):
        hosts = list(hosts[:num_envs])
    else:
        hosts = [hosts] * max(num_envs, len(ports), 1)

    if not hosts:
        hosts = ['127.0.0.1'] * max(num_envs, len(ports), 1)

    while len(hosts) < max(len(ports), num_envs):
        hosts.append(hosts[-1])

    hosts = hosts[:max(len(ports), num_envs)]
    server_wait_timeout = float(carla_cfg.get('server_wait_timeout', 120.0))
    return hosts, ports, server_wait_timeout


def create_run_dir(base_log_dir: str, prefix: str = "") -> Path:
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f"{prefix}_{ts}" if prefix else ts
    d = Path(base_log_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    return d


def save_run_config(config, log_dir: Path):
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)


def get_algorithm_components(config, adapter, obs_key, policy_kwargs,
                             visualizer=None, log_dir=None,
                             config_path=None, checkpoint_path=None):
    """
    Build (AlgorithmClass, algo_kwargs) based on config.algorithm.name.

    The returned algo_kwargs can be directly passed to AlgorithmClass() or
    AlgorithmClass.load().
    """
    algo = config.algorithm
    name = algo.name.lower()
    training = config.training

    common = dict(
        env=adapter,
        learning_rate=algo.learning_rate,
        gamma=algo.gamma,
        min_ready=config.adapter.min_ready,
        visualizer=visualizer,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir / 'tensorboard') if log_dir else None,
        verbose=training.verbose,
        config=config,
    )

    # Optional linear decay for learning rate (applies to both on-policy and
    # off-policy: the off-policy stack evaluates the schedule each TRAIN round
    # and ships the current lr to the train sub-process).
    if getattr(algo, 'learning_rate_final', None) is not None:
        from b2d_rlinfra.learning.utils.schedules import linear_schedule
        common['learning_rate'] = linear_schedule(
            float(algo.learning_rate),
            float(algo.learning_rate_final),
        )

    if name in ('ppo', 'a2c'):
        fe_type = config.policy.features_extractor_type.lower()
        if fe_type not in ("combined_v2", "combined_v3_rgb"):
            raise ValueError(
                "PPO/A2C require policy.feature_extractor.type=combined_v2 "
                "or combined_v3_rgb"
            )
        else:
            from b2d_rlinfra.learning.policies.actor_critic_policy_v2 import ActorCriticPolicyV2
            common['policy'] = ActorCriticPolicyV2

        on_policy = dict(
            n_steps=algo.n_steps,
            gae_lambda=algo.gae_lambda,
            ent_coef=algo.ent_coef,
            ent_coef_final=getattr(algo, 'ent_coef_final', None),
            vf_coef=algo.vf_coef,
            max_grad_norm=algo.max_grad_norm,
            normalize_advantage=getattr(algo, 'normalize_advantage', True),
            truncation_steps=getattr(algo, 'truncation', None),
            success_trajectory_config=getattr(algo, 'success_trajectory', None) or {},
            success_trajectory_log_dir=log_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

        if name == 'ppo':
            from b2d_rlinfra.learning.algorithms.ppo import PPO
            AlgoClass = PPO
            extra = dict(
                batch_size=algo.batch_size,
                n_epochs=algo.n_epochs,
                clip_range=algo.clip_range,
                target_kl=algo.target_kl,
            )
        else:
            from b2d_rlinfra.learning.algorithms.a2c import A2C
            AlgoClass = A2C
            extra = {}

        return AlgoClass, {**common, **on_policy, **extra}

    # ---- off-policy common kwargs ----
    off_policy = dict(
        buffer_size=algo.buffer_size,
        learning_starts=algo.learning_starts,
        batch_size=algo.batch_size,
        tau=algo.tau,
        train_freq=algo.train_freq,
        gradient_steps=algo.gradient_steps,
        warmup_source=getattr(algo, 'warmup_source', 'random'),
        # PER is always enabled in the off-policy stack (the train sub-process
        # requires SharedPrioritizedReplayBuffer); only the hyper-parameters
        # are user-configurable.
        per_alpha=getattr(algo, 'per_alpha', 0.6),
        per_beta=getattr(algo, 'per_beta', 0.4),
        per_beta_annealing_steps=getattr(algo, 'per_beta_annealing_steps', 100_000),
        per_min_priority=getattr(algo, 'per_min_priority', 1e-6),
        # Static (scenario-keyed) replay buffer; {} / enabled:false disables.
        static_buffer_config=getattr(algo, 'static_buffer', None) or None,
    )

    # ---- exploration (TD3/SAC) ----
    exploration_mode = getattr(algo, 'exploration_mode', 'gaussian')
    epsilon_greedy = None
    if exploration_mode == 'epsilon_greedy':
        from b2d_rlinfra.learning.utils.noise import EpsilonGreedyExploration
        epsilon_greedy = EpsilonGreedyExploration(
            explore_actions=getattr(algo, 'explore_actions', [
                [0.1, 0.0], [1.0, 0.0], [-1.0, 0.0],
                [0.7, -0.5], [0.7, 0.5], [0.7, -1.0], [0.7, 1.0],
            ]),
            exploit_prob_initial=getattr(algo, 'exploit_prob_initial', 0.7),
            exploit_prob_final=getattr(algo, 'exploit_prob_final', 0.7),
            decay_steps=getattr(algo, 'exploit_prob_decay_steps', 1_000_000),
        )
    explore_params = dict(
        exploration_mode=exploration_mode,
        epsilon_greedy=epsilon_greedy,
        explore_action_repeat=int(getattr(algo, 'explore_action_repeat', 1) or 1),
    )

    bc_params = dict(
        bc_enabled=getattr(algo, 'bc_enabled', False),
        bc_lambda_initial=getattr(algo, 'bc_lambda_initial', 1.0),
        bc_lambda_decay_steps=getattr(algo, 'bc_lambda_decay_steps', 200_000),
        bc_lambda_final=getattr(algo, 'bc_lambda_final', 0.0),
        bc_prior_action=getattr(algo, 'bc_prior_action', [0.6, 0.0]),
        bc_source=getattr(algo, 'bc_source', 'fixed'),
    )

    if name == 'td3':
        from b2d_rlinfra.learning.algorithms.td3 import TD3
        from b2d_rlinfra.learning.policies.td3_policy import TD3Policy
        from b2d_rlinfra.learning.utils.noise import NormalActionNoise

        action_dim = adapter.action_space.shape[0]
        sigma = algo.exploration_noise
        if isinstance(sigma, (int, float)):
            sigma = sigma * np.ones(action_dim)
        else:
            sigma = np.array(sigma, dtype=np.float32)
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=sigma)

        common['policy'] = TD3Policy
        extra = dict(
            policy_delay=algo.policy_delay,
            target_policy_noise=algo.target_policy_noise,
            target_noise_clip=algo.target_noise_clip,
            action_noise=action_noise,
            exploration_noise_decay_steps=getattr(algo, 'exploration_noise_decay_steps', 0),
            exploration_noise_final_scale=getattr(algo, 'exploration_noise_final_scale', 1.0),
            moe_aux_loss_weight=float(getattr(config.policy, 'moe_aux_loss_weight', 0.0)),
        )
        return TD3, {**common, **off_policy, **bc_params, **explore_params, **extra}

    if name == 'sac':
        from b2d_rlinfra.learning.algorithms.sac import SAC
        from b2d_rlinfra.learning.policies.sac_policy import SACPolicy

        common['policy'] = SACPolicy
        extra = dict(
            ent_coef=algo.ent_coef,
            target_update_interval=algo.target_update_interval,
            target_entropy=algo.target_entropy,
            moe_aux_loss_weight=float(getattr(config.policy, 'moe_aux_loss_weight', 0.0)),
        )
        return SAC, {**common, **off_policy, **bc_params, **explore_params, **extra}

    raise ValueError(f"Unknown algorithm: {name}")
