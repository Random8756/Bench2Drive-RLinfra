"""Soft Actor-Critic learner for the off-policy training stack."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

from .off_policy_algorithm import OffPolicyAlgorithm
from ..policies.sac_policy import SACPolicy
from ..utils.obs_utils import ObsUtils

__layer__ = (4, "Algorithm")

logger = logging.getLogger("Policy")


class SAC(OffPolicyAlgorithm):
    """Soft Actor-Critic with optional BC and MoE auxiliary loss.

    SAC+BC loss::

        L_actor = (alpha * log_prob - Q) / |Q|.detach() + lambda * MSE(a, a_prior)

    ``lambda`` linearly decays from ``bc_lambda_initial`` to ``bc_lambda_final``
    over ``bc_lambda_decay_steps`` environment steps.
    """

    policy: SACPolicy

    def __init__(
        self,
        policy: Type[SACPolicy] = SACPolicy,
        env=None,
        learning_rate: Union[float, callable] = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        visualizer: Optional[Any] = None,
        min_ready: int = 1,
        warmup_source: str = 'random',
        # ---- Exploration ----
        exploration_mode: str = 'gaussian',
        epsilon_greedy: Optional[Any] = None,
        explore_action_repeat: int = 1,
        # ---- SAC + BC ----
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
        # ---- Mixture-of-Experts auxiliary loss ----
        moe_aux_loss_weight: float = 0.0,
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

        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy

        self.bc_enabled = bc_enabled
        self.bc_lambda_initial = bc_lambda_initial
        self.bc_lambda_decay_steps = bc_lambda_decay_steps
        self.bc_lambda_final = bc_lambda_final
        self.bc_prior_action = bc_prior_action or [0.6, 0.0]
        self.bc_source = bc_source

        self.moe_aux_loss_weight = float(moe_aux_loss_weight)

        # Entropy coefficient state lives in the train sub-process; the main
        # process keeps these as None for type compatibility / save data.
        self.log_ent_coef: Optional[torch.Tensor] = None
        self.ent_coef_tensor: Optional[torch.Tensor] = None
        self.ent_coef_optimizer: Optional[torch.optim.Adam] = None

        self._n_updates = 0

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self._setup_target_entropy()
        self._setup_bc_prior_tensor()

        if self.bc_enabled:
            if self.bc_source == 'lqr':
                logger.info("[SAC+BC] BC auxiliary loss enabled (bc_source='lqr')")
            else:
                logger.info("[SAC+BC] BC auxiliary loss enabled (bc_source='fixed')")
                logger.info("[SAC+BC] prior action (env space): %s", self.bc_prior_action)
            logger.info(
                "[SAC+BC] lambda: %s -> %s over %s steps",
                self.bc_lambda_initial, self.bc_lambda_final, self.bc_lambda_decay_steps,
            )
        logger.info("[PER] alpha=%s, beta=%s -> 1.0", self.per_alpha, self.per_beta)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _setup_target_entropy(self) -> None:
        """Resolve ``target_entropy`` to a float (used by the train process)."""
        if self.target_entropy == "auto":
            self.target_entropy = -float(np.prod(self.action_space.shape))
        else:
            self.target_entropy = float(self.target_entropy)

    def _setup_bc_prior_tensor(self) -> None:
        """Cache the prior action tensor for `bc_source='fixed'` exploration use."""
        if self.bc_enabled and self.bc_source != 'lqr':
            prior_env = np.array(self.bc_prior_action, dtype=np.float32)
            prior_normalized = self.policy.unscale_action(prior_env)
            self._bc_prior_tensor = torch.as_tensor(
                prior_normalized, dtype=torch.float32, device=self.device,
            )
        else:
            self._bc_prior_tensor = None

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_action(
        self,
        obs,
        deterministic: bool = False,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(env_action, buffer_action)`` with exploration applied."""
        in_warmup = self.num_timesteps < self.learning_starts

        if in_warmup and self._warmup_source == 'random' and not deterministic:
            return self._random_warmup_action(obs)

        warmup_policy = getattr(self, '_warmup_policy', None)
        if (in_warmup
                and self._warmup_source == 'actor'
                and warmup_policy is not None
                and not deterministic):
            forward_policy = warmup_policy
        else:
            forward_policy = self.policy

        with torch.no_grad():
            obs_keys = ObsUtils.get_obs_keys(self.observation_space)
            obs_tensor = ObsUtils.obs_to_tensor(
                obs, device=self.device, keys=obs_keys,
                observation_space=self.observation_space,
            )
            if (self._exploration_mode == 'epsilon_greedy'
                    and self._epsilon_greedy is not None
                    and not deterministic):
                # Epsilon-greedy: use the deterministic policy output; the
                # exploration comes from the predefined discrete actions.
                action = forward_policy.forward(obs_tensor, deterministic=True)
            else:
                action = forward_policy.forward(obs_tensor, deterministic=deterministic)
            action = action.cpu().numpy()

        # Once warmup ends, drop any temporary `_warmup_policy` to free GPU.
        if (not in_warmup) and warmup_policy is not None:
            self._warmup_policy = None

        # ---- Exploration + action repeat ----
        if not deterministic:
            repeat_n = self._explore_action_repeat
            batch_size = action.shape[0]
            if env_indices is None:
                env_indices = list(range(batch_size))

            if (self._exploration_mode == 'epsilon_greedy'
                    and self._epsilon_greedy is not None):
                action = self._apply_epsilon_greedy_with_repeat(
                    action, env_indices, repeat_n
                )
            elif repeat_n > 1:
                # Gaussian / stochastic policy + action repeat: each worker
                # caches its sampled action and reuses it for the next
                # (repeat_n - 1) steps.
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
        throttle_samples = np.random.normal(0.7, 0.6, size=batch_size)
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

    # ------------------------------------------------------------------
    # Train sub-process configuration
    # ------------------------------------------------------------------

    def _build_train_config(self):
        from .train_process import OffPolicyTrainConfig

        return OffPolicyTrainConfig(
            algorithm_type='sac',
            gamma=self.gamma,
            tau=self.tau,
            bc_enabled=self.bc_enabled,
            bc_lambda_initial=self.bc_lambda_initial,
            bc_lambda_decay_steps=self.bc_lambda_decay_steps,
            bc_lambda_final=self.bc_lambda_final,
            bc_prior_action=list(self.bc_prior_action),
            bc_source=self.bc_source,
            target_update_interval=self.target_update_interval,
            target_entropy=self.target_entropy,
            ent_coef=self.ent_coef,
            moe_aux_loss_weight=self.moe_aux_loss_weight,
            n_updates=self._n_updates,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # Save data
    # ------------------------------------------------------------------

    def _get_save_data(self) -> Dict[str, Any]:
        data = super()._get_save_data()
        data.update({
            'ent_coef': self.ent_coef,
            'target_update_interval': self.target_update_interval,
            'target_entropy': self.target_entropy,
            '_n_updates': self._n_updates,
            'bc_enabled': self.bc_enabled,
            'bc_lambda_initial': self.bc_lambda_initial,
            'bc_lambda_decay_steps': self.bc_lambda_decay_steps,
            'bc_lambda_final': self.bc_lambda_final,
            'bc_prior_action': self.bc_prior_action,
            'bc_source': self.bc_source,
            'per_alpha': self.per_alpha,
            'per_beta': self.per_beta,
            'moe_aux_loss_weight': self.moe_aux_loss_weight,
        })
        return data
