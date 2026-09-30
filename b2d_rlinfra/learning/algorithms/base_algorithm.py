"""Base algorithm class.

Abstract base for on-policy (PPO / A2C) and off-policy (SAC / TD3)
algorithms: shared plumbing for device handling, save / load, callbacks,
episode counters, and logger.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path
from collections import deque
import json
import logging
import sys
import time
import random

import numpy as np

logger = logging.getLogger("Policy")
import torch
import torch.nn as nn
from gymnasium import spaces

from ..policies.base_policy import BasePolicy
from ..utils.logger import Logger, configure_logger
from ..utils.callbacks import BaseCallback, CallbackList
from ..utils.episode_success import classify_episode
from ..utils.schedules import get_schedule_fn

__layer__ = (4, "Algorithm")

TRAINING_SIM_HZ = 10.0
EPISODE_LOG_WINDOW = 100
SCENARIO_LOG_WINDOW = 20


class BaseAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    
    Provides common functionality:
    - Environment management
    - Model save/load
    - Logging
    - Callbacks
    """
    
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union['VecEnv', None],
        learning_rate: Union[float, callable],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = 'auto',
        seed: Optional[int] = None,
        config: Optional[Union[Dict, Any]] = None,
        _init_setup_model: bool = True,
    ):
        """
        Initialize algorithm.
        
        Args:
            policy: Policy class to use.
            env: Environment to train on.
            learning_rate: Learning rate or schedule function.
            policy_kwargs: Additional arguments for policy.
            tensorboard_log: Directory for TensorBoard logs.
            verbose: Verbosity level.
            device: Device for training.
            seed: Random seed.
            config: Full configuration (for saving/loading).
            _init_setup_model: Whether to setup model immediately.
        """
        self.policy_class = policy
        self.env = env
        self.learning_rate = learning_rate
        self.policy_kwargs = policy_kwargs or {}
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.config = config  # Full config for checkpoint saving
        
        # Determine device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Initialize attributes
        self.policy: Optional[BasePolicy] = None
        self.logger: Optional[Logger] = None
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._num_timesteps_at_start = 0
        self.start_time: Optional[float] = None
        
        # Environment attributes
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None
        self.n_envs = 0
        
        # Callbacks
        self._current_progress_remaining = 1.0
        self._episode_num = 0
        self._ep_reward_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        self._ep_length_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        self._ep_success_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        self._ep_route_completion_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        self._scenario_episode_buffers: Dict[str, deque] = {}
        self._scenario_episode_counts: Dict[str, int] = {}
        
        # Get spaces from environment
        if env is not None:
            self._setup_env(env)
        
        # Set seed
        if seed is not None:
            self.set_random_seed(seed)
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_env(self, env) -> None:
        """
        Setup environment and get spaces.
        
        Args:
            env: Environment to setup.
        """
        self.env = env
        self.n_envs = getattr(env, 'num_envs', 1)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    @abstractmethod
    def _setup_model(self) -> None:
        """Setup policy and other model components."""
        raise NotImplementedError
    
    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
    ) -> 'BaseAlgorithm':
        """
        Train the algorithm.

        Args:
            total_timesteps: Total timesteps to train.
            callback: Callbacks to run during training.
            log_interval: Logging frequency.
            reset_num_timesteps: Whether to reset timestep counter.

        Returns:
            self
        """
        raise NotImplementedError
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get action for observation.
        
        Args:
            observation: Current observation.
            state: RNN state.
            episode_start: Whether this is the start of an episode.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Tuple of (action, state).
        """
        return self.policy.predict(observation, state, episode_start, deterministic)
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.env is not None and hasattr(self.env, 'seed'):
            self.env.seed(seed)
    
    def _setup_learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
        reset_num_timesteps: bool = True,
    ) -> Tuple[int, BaseCallback]:
        """
        Setup before learning.
        
        Args:
            total_timesteps: Total timesteps to train.
            callback: Callbacks.
            reset_num_timesteps: Whether to reset counter.
            
        Returns:
            Tuple of (total_timesteps, callback).
        """
        self.start_time = time.time()
        
        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
            self._reset_episode_metric_buffers()
        
        self._num_timesteps_at_start = self.num_timesteps
        self._total_timesteps = total_timesteps
        
        # Setup logger
        if self.logger is None:
            self.logger = configure_logger(
                self.tensorboard_log or './logs',
                ['stdout', 'tensorboard'] if self.tensorboard_log else ['stdout'],
                self.verbose,
            )
        
        # Setup callbacks
        if callback is None:
            callback_list = CallbackList([])
        elif isinstance(callback, list):
            callback_list = CallbackList(callback)
        else:
            callback_list = CallbackList([callback])
        
        callback_list.init_callback(self)
        
        return total_timesteps, callback_list

    def _reset_episode_metric_buffers(self) -> None:
        self._ep_reward_buffer.clear()
        self._ep_length_buffer.clear()
        self._ep_success_buffer.clear()
        self._ep_route_completion_buffer.clear()
        self._scenario_episode_buffers.clear()
        self._scenario_episode_counts.clear()

    def _reset_rollout_training_timer(self) -> None:
        """Start wall-clock accounting after environments are ready."""
        self.start_time = time.time()
        self._num_timesteps_at_start = self.num_timesteps

    def _get_elapsed_sec(self) -> float:
        if self.start_time is None:
            return 0.0
        return max(time.time() - self.start_time, sys.float_info.epsilon)

    def _get_sim_real_ratio(self, elapsed_sec: Optional[float] = None) -> float:
        elapsed = self._get_elapsed_sec() if elapsed_sec is None else elapsed_sec
        if elapsed <= 0.0:
            return 0.0
        simulated_seconds = max(
            0.0,
            float(self.num_timesteps - self._num_timesteps_at_start) / TRAINING_SIM_HZ,
        )
        return simulated_seconds / elapsed

    @staticmethod
    def _format_progress_metric(value: Optional[float], precision: int = 1) -> str:
        if value is None:
            return "nan"
        return f"{float(value):.{precision}f}"

    def _format_progress_line(
        self,
        total_timesteps: int,
        sim_real_ratio: float,
        reward_mean: Optional[float],
        length_mean: Optional[float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        total = max(int(total_timesteps), 1)
        progress_pct = 100.0 * float(self.num_timesteps) / float(total)
        parts = [
            "[progress]",
            f"algo={self.__class__.__name__.upper()}",
            f"iter={self._iteration}",
            f"steps={self.num_timesteps:,}/{int(total_timesteps):,}",
            f"pct={progress_pct:.1f}%",
            f"sim_real={sim_real_ratio:.2f}x",
            f"episodes={self._episode_num}",
            f"reward_mean={self._format_progress_metric(reward_mean)}",
            f"length_mean={self._format_progress_metric(length_mean)}",
        ]
        for key, value in (extra or {}).items():
            parts.append(f"{key}={value}")
        return " ".join(parts)
    
    def _update_current_progress_remaining(
        self,
        num_timesteps: int,
        total_timesteps: int,
    ) -> None:
        """Update progress for learning rate schedules."""
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    @staticmethod
    def _get_scenario_name(info: Optional[Dict[str, Any]]) -> str:
        if not isinstance(info, dict):
            return "unknown"
        scenario_name = info.get("scenario_name")
        if scenario_name is None:
            return "unknown"
        scenario_name = str(scenario_name).strip()
        return scenario_name or "unknown"

    @staticmethod
    def _format_episode_context(info: Optional[Dict[str, Any]]) -> str:
        if not isinstance(info, dict):
            return ""

        route_id = str(info.get("route_id") or "").strip()
        scenario_name = str(info.get("scenario_name") or "").strip()
        scenario_instance_name = str(info.get("scenario_instance_name") or "").strip()
        town = str(info.get("town") or "").strip()

        scenario_display = scenario_name or scenario_instance_name
        if scenario_name and scenario_instance_name and scenario_instance_name != scenario_name:
            scenario_display = f"{scenario_display} ({scenario_instance_name})"

        parts = []
        if route_id:
            parts.append(f"route={route_id}")
        if scenario_display:
            parts.append(f"scenario={scenario_display}")
        if town:
            parts.append(f"town={town}")
        return " | ".join(parts)

    def _record_episode_metrics(
        self,
        info: Optional[Dict[str, Any]],
        episode_reward: float,
        episode_length: float,
    ) -> str:
        scenario_name = self._get_scenario_name(info)
        classification = classify_episode([info] if isinstance(info, dict) else [])
        route_completion = max(
            0.0,
            min(float(classification.get("route_completed", 0.0)) / 100.0, 1.0),
        )
        success = 1.0 if classification.get("is_success", False) else 0.0

        self._ep_reward_buffer.append(float(episode_reward))
        self._ep_length_buffer.append(float(episode_length))
        self._ep_success_buffer.append(success)
        self._ep_route_completion_buffer.append(route_completion)

        buffer = self._scenario_episode_buffers.get(scenario_name)
        if buffer is None:
            buffer = deque(maxlen=SCENARIO_LOG_WINDOW)
            self._scenario_episode_buffers[scenario_name] = buffer
        buffer.append({
            "reward": float(episode_reward),
            "success": success,
            "route_completion": route_completion,
        })
        self._scenario_episode_counts[scenario_name] = (
            self._scenario_episode_counts.get(scenario_name, 0) + 1
        )
        return scenario_name

    def _record_episode_tensorboard_metrics(self) -> None:
        if self.logger is None:
            return

        episode_window_suffix = f"last_{EPISODE_LOG_WINDOW}_episodes"
        scenario_window_suffix = f"last_{SCENARIO_LOG_WINDOW}_episodes"

        if self._ep_reward_buffer:
            self.logger.record(
                f'episode/reward_mean_{episode_window_suffix}',
                float(np.mean(self._ep_reward_buffer)),
            )
        if self._ep_length_buffer:
            self.logger.record(
                f'episode/length_mean_{episode_window_suffix}',
                float(np.mean(self._ep_length_buffer)),
            )
        if self._ep_success_buffer:
            self.logger.record(
                f'episode/success_rate_mean_{episode_window_suffix}',
                float(np.mean(self._ep_success_buffer)),
            )
        if self._ep_route_completion_buffer:
            self.logger.record(
                f'episode/route_completion_mean_{episode_window_suffix}',
                float(np.mean(self._ep_route_completion_buffer)),
            )

        for scenario_name in sorted(self._scenario_episode_buffers):
            episodes = self._scenario_episode_buffers[scenario_name]
            if not episodes:
                continue
            tag = scenario_name
            rewards = [item["reward"] for item in episodes]
            successes = [item["success"] for item in episodes]
            route_completions = [item["route_completion"] for item in episodes]
            self.logger.record(
                f"scenario/reward_mean_{scenario_window_suffix}/{tag}",
                float(np.mean(rewards)),
                exclude='stdout',
            )
            self.logger.record(
                f"scenario/success_rate_mean_{scenario_window_suffix}/{tag}",
                float(np.mean(successes)),
                exclude='stdout',
            )
            self.logger.record(
                f"scenario/route_completion_mean_{scenario_window_suffix}/{tag}",
                float(np.mean(route_completions)),
                exclude='stdout',
            )
            self.logger.record(
                f"scenario/episode_count/{tag}",
                self._scenario_episode_counts.get(scenario_name, len(episodes)),
                exclude='stdout',
            )
    
    def get_lr_schedule_fn(self) -> callable:
        """Get learning rate as a schedule function."""
        return get_schedule_fn(self.learning_rate)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to a folder.
        
        The folder contains:
        - config.yaml: Full configuration (algorithm + env)
        - policy.pth: Policy state dict
        - training_state.pth: Optimizer state dict when available
        - metadata.json: Training metadata (timesteps, etc.)
        
        Args:
            path: Path to checkpoint folder.
        """
        import yaml
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        if self.config is not None:
            config_to_save = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
            with open(save_dir / 'config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
        
        # Save policy weights
        torch.save(self.policy.state_dict(), save_dir / 'policy.pth')

        # Save PPO/A2C optimizer state when present.
        optimizer = getattr(self.policy, 'optimizer', None)
        if optimizer is not None:
            torch.save(
                {
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                save_dir / 'training_state.pth',
            )
        
        # Save metadata
        metadata = {
            'policy_class': self.policy_class.__name__ if hasattr(self.policy_class, '__name__') else str(self.policy_class),
            'num_timesteps': self.num_timesteps,
            'total_timesteps': self._total_timesteps,
            'n_envs': self.n_envs,
            'episode_num': self._episode_num,
            'n_updates': int(getattr(self, '_n_updates', 0)),
            'scenario_episode_counts': dict(self._scenario_episode_counts),
        }
        metadata.update(self._get_save_data())
        with open(save_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(
            "[artifact] type=checkpoint path=%s steps=%d",
            save_dir,
            self.num_timesteps,
        )
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data for saving. Override in subclasses."""
        return {}
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env = None,
        device: Union[str, torch.device] = 'auto',
        **kwargs,
    ) -> 'BaseAlgorithm':
        """
        Load model from checkpoint folder.
        
        The folder should contain:
        - config.yaml: Full configuration
        - policy.pth: Policy state dict
        - training_state.pth: Optimizer state dict when available
        - metadata.json: Training metadata
        
        Args:
            path: Path to checkpoint folder.
            env: Environment to use.
            device: Device to load to.
            **kwargs: Additional arguments (will override checkpoint config).
                Required: policy (policy class)
                Optional: learning_rate, n_steps, batch_size, etc.
            
        Returns:
            Loaded algorithm.
        """
        import yaml
        
        load_dir = Path(path).expanduser().resolve()
        if not load_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {load_dir}")

        policy_path = load_dir / 'policy.pth'
        if not policy_path.is_file():
            raise FileNotFoundError(f"Checkpoint policy weights not found: {policy_path}")
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load configuration from checkpoint (for reference)
        config_path = load_dir / 'config.yaml'
        checkpoint_config = None
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                checkpoint_config = yaml.safe_load(f)
        
        # Load metadata
        metadata_path = load_dir / 'metadata.json'
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Load policy weights
        policy_state_dict = torch.load(policy_path, map_location=device, weights_only=False)

        # Optional for older/evaluation-only checkpoints.
        training_state_path = load_dir / 'training_state.pth'
        training_state = None
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=device, weights_only=False)
        
        # Use caller-provided config, fallback to checkpoint config
        config = kwargs.pop('config', checkpoint_config)
        
        # Get policy class (must be provided by caller as actual class, not string)
        policy = kwargs.pop('policy', None)
        if policy is None:
            raise ValueError(
                "Policy class must be provided when loading. "
                "Pass policy=ActorCriticPolicyV2 to load()."
            )
        
        # Create algorithm instance with kwargs
        model = cls(
            policy=policy,
            env=env,
            device=device,
            config=config,
            _init_setup_model=False,
            **kwargs,
        )
        
        # Restore num_timesteps from metadata
        model.num_timesteps = metadata.get('num_timesteps', 0)
        model._episode_num = metadata.get('episode_num', 0)
        model._scenario_episode_counts = {
            str(name): int(count)
            for name, count in (metadata.get('scenario_episode_counts') or {}).items()
        }
        
        # Setup model and load weights
        model._setup_model()
        model.policy.load_state_dict(policy_state_dict)
        optimizer = getattr(model.policy, 'optimizer', None)
        if (
            optimizer is not None
            and isinstance(training_state, dict)
            and 'optimizer_state_dict' in training_state
        ):
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
        elif optimizer is not None and not training_state_path.exists():
            logger.info(
                "Checkpoint has no training_state.pth; optimizer state was not restored"
            )
        if hasattr(model, '_n_updates'):
            model._n_updates = int(metadata.get('n_updates', getattr(model, '_n_updates', 0)))
        model._total_timesteps = int(
            metadata.get('total_timesteps', getattr(model, '_total_timesteps', 0))
        )
        if hasattr(model, '_load_save_data'):
            model._load_save_data(metadata)
        
        return model
    
    def get_env(self):
        """Get environment."""
        return self.env
    
    def set_env(self, env) -> None:
        """Set environment."""
        self._setup_env(env)
    
    def _excluded_save_params(self) -> List[str]:
        """Parameters to exclude from saving."""
        return ['policy', 'env', 'logger']
    
    @property
    def _episode_count(self) -> int:
        """Get episode count."""
        return self._episode_num
