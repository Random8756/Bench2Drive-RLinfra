"""TD3 learner for the off-policy training stack."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from .off_policy_algorithm import OffPolicyAlgorithm
from ..policies.td3_policy import TD3Policy

__layer__ = (4, "Algorithm")
from ..utils.noise import ActionNoise, VectorizedActionNoise
from ..utils.obs_utils import ObsUtils

logger = logging.getLogger("Policy")


class TD3(OffPolicyAlgorithm):
    """Twin Delayed DDPG with optional BC."""

    policy: TD3Policy

    def __init__(
        self,
        policy: Type[TD3Policy] = TD3Policy,
        env=None,
        learning_rate: Union[float, callable] = 1e-3,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: Union[float, List[float]] = 0.2,
        target_noise_clip: Union[float, List[float]] = 0.5,
        action_noise: Optional[ActionNoise] = None,
        visualizer: Optional[Any] = None,
        min_ready: int = 1,
        warmup_source: str = 'random',
        exploration_mode: str = 'gaussian',
        epsilon_greedy: Optional[Any] = None,
        explore_action_repeat: int = 1,
        exploration_noise_decay_steps: int = 0,
        exploration_noise_final_scale: float = 1.0,
        moe_aux_loss_weight: float = 0.0,
        # ---- TD3 + BC ----
        bc_enabled: bool = False,
        bc_lambda_initial: float = 1.0,
        bc_lambda_decay_steps: int = 200_000,
        bc_lambda_final: float = 0.0,
        bc_prior_action: Optional[List[float]] = None,
        bc_source: str = 'fixed',
        # ---- PER ----
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_annealing_steps: int = 100_000,
        per_min_priority: float = 1e-6,
        # ---- Plumbing ----
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = 'auto',
        seed: Optional[int] = None,
        config: Optional[Any] = None,
        static_buffer_config: Optional[Dict[str, Any]] = None,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            visualizer=visualizer,
            min_ready=min_ready,
            warmup_source=warmup_source,
            exploration_mode=exploration_mode,
            epsilon_greedy=epsilon_greedy,
            explore_action_repeat=explore_action_repeat,
            bc_source=bc_source,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_beta_annealing_steps=per_beta_annealing_steps,
            per_min_priority=per_min_priority,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            config=config,
            static_buffer_config=static_buffer_config,
            _init_setup_model=False,
        )

        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.action_noise = action_noise
        self.exploration_noise_decay_steps = int(exploration_noise_decay_steps)
        self.exploration_noise_final_scale = float(exploration_noise_final_scale)
        self.moe_aux_loss_weight = float(moe_aux_loss_weight)

        self.bc_enabled = bc_enabled
        self.bc_lambda_initial = bc_lambda_initial
        self.bc_lambda_decay_steps = bc_lambda_decay_steps
        self.bc_lambda_final = bc_lambda_final
        self.bc_prior_action = bc_prior_action or [0.6, 0.0]
        self.bc_source = bc_source

        self._n_updates = 0

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        action_dim = int(np.prod(self.action_space.shape))

        if isinstance(self.target_policy_noise, (int, float)):
            self.target_policy_noise = np.full(action_dim, self.target_policy_noise, dtype=np.float32)
        else:
            self.target_policy_noise = np.array(self.target_policy_noise, dtype=np.float32)
            assert len(self.target_policy_noise) == action_dim, (
                f"target_policy_noise length ({len(self.target_policy_noise)}) "
                f"must match action_dim ({action_dim})"
            )

        if isinstance(self.target_noise_clip, (int, float)):
            self.target_noise_clip = np.full(action_dim, self.target_noise_clip, dtype=np.float32)
        else:
            self.target_noise_clip = np.array(self.target_noise_clip, dtype=np.float32)
            assert len(self.target_noise_clip) == action_dim, (
                f"target_noise_clip length ({len(self.target_noise_clip)}) "
                f"must match action_dim ({action_dim})"
            )

        # Vectorise the noise generator across parallel envs so each worker
        # has its own independent noise state.
        if self.action_noise is not None and self.n_envs > 1:
            if not isinstance(self.action_noise, VectorizedActionNoise):
                self.action_noise = VectorizedActionNoise(self.action_noise, self.n_envs)

        if self.bc_enabled:
            if self.bc_source == 'lqr':
                logger.info("[TD3+BC] BC auxiliary loss enabled (bc_source='lqr')")
            else:
                logger.info("[TD3+BC] BC auxiliary loss enabled (bc_source='fixed')")
                logger.info("[TD3+BC] prior action (env space): %s", self.bc_prior_action)
            logger.info(
                "[TD3+BC] lambda: %s -> %s over %s steps",
                self.bc_lambda_initial, self.bc_lambda_final, self.bc_lambda_decay_steps,
            )
        if self.exploration_noise_decay_steps > 0:
            logger.info(
                "[TD3] exploration noise sigma decay: 1.0 -> %sx over %s steps (post-warmup)",
                self.exploration_noise_final_scale, self.exploration_noise_decay_steps,
            )
        logger.info("[PER] alpha=%s, beta=%s -> 1.0", self.per_alpha, self.per_beta)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_action(
        self,
        obs,
        deterministic: bool = False,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample TD3 action and return ``(env_action, buffer_action)``."""
        if self.num_timesteps < self.learning_starts and not deterministic:
            return self._random_warmup_action(obs)

        with torch.no_grad():
            obs_keys = ObsUtils.get_obs_keys(self.observation_space)
            obs_tensor = ObsUtils.obs_to_tensor(
                obs, device=self.device, keys=obs_keys,
                observation_space=self.observation_space,
            )
            action = self.policy.forward(obs_tensor)
            action = action.cpu().numpy()

        if not deterministic:
            repeat_n = self._explore_action_repeat
            if env_indices is None:
                env_indices = list(range(action.shape[0]))

            if (self._exploration_mode == 'epsilon_greedy'
                    and self._epsilon_greedy is not None):
                action = self._apply_epsilon_greedy_with_repeat(
                    action, env_indices, repeat_n
                )
            else:
                if self.action_noise is not None:
                    if isinstance(self.action_noise, VectorizedActionNoise) and env_indices is not None:
                        noise = self.action_noise(indices=env_indices)
                    else:
                        noise = self.action_noise()
                        if noise.ndim < action.ndim:
                            noise = np.tile(noise, (action.shape[0], 1))
                    noise = noise * self._get_exploration_noise_scale()
                    action = np.clip(action + noise, -1, 1)
                if repeat_n > 1:
                    action = self._apply_gaussian_action_repeat(
                        action, env_indices, repeat_n
                    )

        scaled_action = self.policy.scale_action(action)
        return scaled_action, action

    def _random_warmup_action(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(obs, dict):
            batch_size = len(next(iter(obs.values())))
        else:
            batch_size = len(obs) if obs.ndim > 1 else 1
        action_dim = int(np.prod(self.action_space.shape))

        action = np.zeros((batch_size, action_dim), dtype=np.float32)
        throttle_samples = np.random.normal(0.6, 1, size=batch_size)
        action[:, 0] = np.clip(throttle_samples, self.action_space.low[0], self.action_space.high[0])
        if action_dim > 1:
            steering_samples = np.random.normal(0, 0.25, size=batch_size)
            action[:, 1] = np.clip(steering_samples, self.action_space.low[1], self.action_space.high[1])
            for i in range(2, action_dim):
                action[:, i] = np.random.uniform(
                    self.action_space.low[i], self.action_space.high[i], size=batch_size,
                )

        unscaled = self.policy.unscale_action(action)
        return action, unscaled

    def _get_exploration_noise_scale(self) -> float:
        """Linearly decay sigma from 1.0 to ``exploration_noise_final_scale``.

        Decay is measured from the end of warmup (``learning_starts``).
        Returns 1.0 if decay is disabled (``exploration_noise_decay_steps == 0``).
        """
        decay_steps = self.exploration_noise_decay_steps
        if decay_steps <= 0:
            return 1.0
        elapsed = max(0, self.num_timesteps - self.learning_starts)
        progress = min(1.0, elapsed / decay_steps)
        return 1.0 + (self.exploration_noise_final_scale - 1.0) * progress

    # ------------------------------------------------------------------
    # Train sub-process configuration
    # ------------------------------------------------------------------

    def _build_train_config(self):
        from .train_process import OffPolicyTrainConfig

        target_policy_noise = (
            self.target_policy_noise.tolist()
            if isinstance(self.target_policy_noise, np.ndarray)
            else list(self.target_policy_noise)
        )
        target_noise_clip = (
            self.target_noise_clip.tolist()
            if isinstance(self.target_noise_clip, np.ndarray)
            else list(self.target_noise_clip)
        )
        return OffPolicyTrainConfig(
            algorithm_type='td3',
            gamma=self.gamma,
            tau=self.tau,
            bc_enabled=self.bc_enabled,
            bc_lambda_initial=self.bc_lambda_initial,
            bc_lambda_decay_steps=self.bc_lambda_decay_steps,
            bc_lambda_final=self.bc_lambda_final,
            bc_prior_action=list(self.bc_prior_action),
            bc_source=self.bc_source,
            policy_delay=self.policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            moe_aux_loss_weight=self.moe_aux_loss_weight,
            n_updates=self._n_updates,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------

    def _get_save_data(self) -> Dict[str, Any]:
        data = super()._get_save_data()
        target_policy_noise = (
            self.target_policy_noise.tolist()
            if isinstance(self.target_policy_noise, np.ndarray)
            else list(self.target_policy_noise)
        )
        target_noise_clip = (
            self.target_noise_clip.tolist()
            if isinstance(self.target_noise_clip, np.ndarray)
            else list(self.target_noise_clip)
        )
        data.update({
            'policy_delay': self.policy_delay,
            'target_policy_noise': target_policy_noise,
            'target_noise_clip': target_noise_clip,
            'moe_aux_loss_weight': self.moe_aux_loss_weight,
            '_n_updates': self._n_updates,
            'bc_enabled': self.bc_enabled,
            'bc_lambda_initial': self.bc_lambda_initial,
            'bc_lambda_decay_steps': self.bc_lambda_decay_steps,
            'bc_lambda_final': self.bc_lambda_final,
            'bc_prior_action': self.bc_prior_action,
            'bc_source': self.bc_source,
            'per_alpha': self.per_alpha,
            'per_beta': self.per_beta,
        })
        return data
