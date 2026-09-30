"""Advantage Actor-Critic (A2C).

Vanilla advantage actor-critic operating on the per-worker rollout buffer.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Type, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from gymnasium import spaces

from .on_policy_algorithm import OnPolicyAlgorithm
from ..policies.actor_critic_policy_v2 import ActorCriticPolicyV2
from ..utils.schedules import get_schedule_fn
from ..utils.value_repr import (
    apply_value_transform,
    categorical_distribution_entropy,
    categorical_value_loss,
    categorical_value_stats,
    scalar_to_twohot,
)

if TYPE_CHECKING:
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer

# Use a dedicated logger name to avoid clashing with the envs layer.
logger = logging.getLogger("Policy")

__layer__ = (4, "Algorithm")


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    
    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    
    Args:
        y_pred: the prediction
        y_true: the expected value
        
    Returns:
        explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) algorithm.
    
    A2C is a synchronous, deterministic variant of Asynchronous Advantage Actor-Critic (A3C).
    It uses the advantage function to reduce variance of policy gradient estimates.
    
    Key differences from PPO:
    - No clipping of policy objective
    - Single gradient update per rollout (no epochs/minibatches)
    - Default gae_lambda=1.0 (full returns) instead of 0.95
    
    Features:
    - Value function for baseline
    - Entropy regularization
    - GAE for advantage estimation
    """
    
    def __init__(
        self,
        policy: Type[ActorCriticPolicyV2] = ActorCriticPolicyV2,
        env = None,
        learning_rate: Union[float, callable] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: Union[float, callable] = 0.0,
        ent_coef_final: Optional[float] = None,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = False,
        min_ready: int = 1,
        truncation_steps: Optional[int] = None,
        visualizer: Optional["TrainingVisualizer"] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = 'auto',
        seed: Optional[int] = None,
        config: Optional[Any] = None,
        success_trajectory_config: Optional[Dict[str, Any]] = None,
        success_trajectory_log_dir: Optional[Union[str, Any]] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        _init_setup_model: bool = True,
    ):
        """
        Initialize A2C algorithm.
        
        Args:
            policy: Policy class.
            env: Environment.
            learning_rate: Learning rate or schedule.
            n_steps: Number of steps per rollout (total across all workers).
            gamma: Discount factor.
            gae_lambda: GAE lambda (1.0 = full returns, no GAE).
            ent_coef: Entropy coefficient (float) or schedule callable.
            ent_coef_final: Optional final entropy coefficient for linear decay.
            vf_coef: Value function coefficient.
            max_grad_norm: Maximum gradient norm.
            normalize_advantage: Whether to normalize advantages.
            min_ready: For Standard mode, minimum workers to wait for per step.
            truncation_steps: Algorithm-internal episode slicing step count (None = disabled).
            visualizer: TrainingVisualizer for recording training episodes (optional).
            policy_kwargs: Additional policy arguments.
            tensorboard_log: TensorBoard log directory.
            verbose: Verbosity level.
            device: Device for training.
            seed: Random seed.
            config: Full configuration for checkpoint saving.
            success_trajectory_config: Optional successful trajectory export config.
            success_trajectory_log_dir: Current run directory for default exports.
            config_path: Source config path for trajectory metadata.
            checkpoint_path: Source checkpoint path for trajectory metadata.
            _init_setup_model: Whether to setup model.
        """
        self.normalize_advantage = normalize_advantage
        self.ent_coef_final = ent_coef_final
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            min_ready=min_ready,
            truncation_steps=truncation_steps,
            visualizer=visualizer,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            config=config,
            success_trajectory_config=success_trajectory_config,
            success_trajectory_log_dir=success_trajectory_log_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            _init_setup_model=False,
        )
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """Setup policy and initialize training state."""
        super()._setup_model()
        if (
            self.ent_coef_final is not None
            and isinstance(self.ent_coef, (int, float, np.floating))
        ):
            self.ent_coef_schedule = get_schedule_fn(
                lambda progress_remaining: float(self.ent_coef_final)
                + progress_remaining * (float(self.ent_coef) - float(self.ent_coef_final))
            )
        else:
            self.ent_coef_schedule = get_schedule_fn(self.ent_coef)
        
        # Training statistics
        self._n_updates = 0
    
    @property
    def uses_categorical_value_head(self) -> bool:
        """Whether A2C is using a categorical critic head."""
        return getattr(self.policy, 'uses_categorical_value_head', False)

    def _build_categorical_targets(
        self,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build two-hot categorical targets from raw scalar returns."""
        support = self.policy.value_support.to(device=returns.device, dtype=returns.dtype)
        transformed_returns = apply_value_transform(returns, self.policy.value_transform)
        clipped_returns = transformed_returns.clamp(min=support[0], max=support[-1])
        target_probs = scalar_to_twohot(clipped_returns, support)
        clip_frac = (
            (transformed_returns < support[0]) | (transformed_returns > support[-1])
        ).float().mean()
        return target_probs, clipped_returns, clip_frac
    
    def train(self) -> None:
        """
        Update policy using collected rollouts.
        
        A2C performs a single gradient update over the entire rollout buffer
        (no epochs, no minibatches).
        """
        self.policy.set_training_mode(True)
        
        # Update learning rate
        lr = self.get_lr_schedule_fn()(self._current_progress_remaining)
        for param_group in self.policy.optimizer.param_groups:
            param_group['lr'] = lr
        current_ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        
        # Training statistics
        entropy_losses = []
        entropy_values = []
        pg_losses = []
        value_losses = []
        grad_norms = []
        value_pred_means_raw = []
        value_pred_means_transformed = []
        value_target_means_raw = []
        value_target_means_transformed = []
        value_support_clip_fracs = []
        value_dist_entropies = []
        nan_batches_skipped = 0

        def _has_nan_or_inf(data):
            if isinstance(data, dict):
                return any(
                    torch.isnan(v).any() or torch.isinf(v).any()
                    for v in data.values()
                )
            return torch.isnan(data).any() or torch.isinf(data).any()
        
        # A2C uses full buffer (batch_size=None means get all data at once)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Check for NaN in input data
            if (_has_nan_or_inf(rollout_data.observations) or
                _has_nan_or_inf(rollout_data.advantages) or
                _has_nan_or_inf(rollout_data.returns)):
                nan_batches_skipped += 1
                logger.warning("Skipping batch with NaN/Inf values in rollout data")
                continue
            
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()
            
            # Evaluate actions
            try:
                if self.uses_categorical_value_head:
                    values, value_logits, log_prob, entropy = (
                        self.policy.evaluate_actions_with_value_logits(
                            rollout_data.observations,
                            actions,
                        )
                    )
                else:
                    value_logits = None
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                    )
            except ValueError as e:
                # Catch NaN errors from Categorical distribution
                if "invalid values" in str(e) or "nan" in str(e).lower():
                    nan_batches_skipped += 1
                    logger.warning("Skipping batch due to NaN in policy output")
                    continue
                raise
            
            values = values.flatten()
            
            # Check for NaN in policy output
            if _has_nan_or_inf(values) or _has_nan_or_inf(log_prob):
                nan_batches_skipped += 1
                logger.warning("NaN detected in values/log_prob, skipping batch")
                continue
            if value_logits is not None and _has_nan_or_inf(value_logits):
                nan_batches_skipped += 1
                logger.warning("NaN detected in categorical value logits, skipping batch")
                continue
            
            # Normalize advantages (not present in original A2C, but can be enabled)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()
            
            # Policy gradient loss (simple, no clipping)
            # L_pi = -E[A * log(pi)]
            policy_loss = -(advantages * log_prob).mean()
            
            # Check for NaN in loss
            if torch.isnan(policy_loss):
                nan_batches_skipped += 1
                logger.warning("NaN in policy loss, skipping batch")
                continue
            
            pg_losses.append(policy_loss.item())
            
            # Value loss
            if self.uses_categorical_value_head:
                target_probs, _, support_clip_frac = (
                    self._build_categorical_targets(rollout_data.returns)
                )
                value_loss = categorical_value_loss(value_logits, target_probs)
                value_losses.append(value_loss.item())

                with torch.no_grad():
                    pred_mean_raw, pred_mean_transformed, target_mean_raw, target_mean_transformed = (
                        categorical_value_stats(
                            value_logits,
                            self.policy.value_support.to(
                                device=value_logits.device,
                                dtype=value_logits.dtype,
                            ),
                            self.policy.value_transform,
                            rollout_data.returns,
                        )
                    )
                    value_pred_means_raw.append(pred_mean_raw.item())
                    value_pred_means_transformed.append(pred_mean_transformed.item())
                    value_target_means_raw.append(target_mean_raw.item())
                    value_target_means_transformed.append(target_mean_transformed.item())
                    value_support_clip_fracs.append(support_clip_frac.item())
                    value_dist_entropies.append(
                        categorical_distribution_entropy(value_logits).item()
                    )
            else:
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())
            
            # Entropy loss (encourages exploration)
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
                entropy_value = (-log_prob).mean().item()
            else:
                entropy_loss = -torch.mean(entropy)
                entropy_value = entropy.mean().item()
            
            entropy_losses.append(entropy_loss.item())
            entropy_values.append(entropy_value)
            
            # Total loss
            loss = policy_loss + current_ent_coef * entropy_loss + self.vf_coef * value_loss
            
            # Final NaN check before backward
            if torch.isnan(loss):
                nan_batches_skipped += 1
                logger.warning("NaN in total loss, skipping batch")
                continue
            
            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.policy.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                nan_batches_skipped += 1
                logger.warning("NaN gradients detected, skipping update")
                self.policy.optimizer.zero_grad()
                continue
            
            # Clip gradients and record the norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            grad_norms.append(grad_norm.item())
            
            self.policy.optimizer.step()
        
        self._n_updates += 1
        
        # Log NaN statistics
        if nan_batches_skipped > 0:
            logger.warning(f"Skipped {nan_batches_skipped} batches due to NaN/Inf values this iteration")
        
        # Compute explained variance
        explained_var = np.nan
        if self.rollout_buffer.values is not None and self.rollout_buffer.returns is not None:
            values_flat = self.rollout_buffer.values.flatten()
            returns_flat = self.rollout_buffer.returns.flatten()
            if len(values_flat) > 0 and len(returns_flat) > 0:
                explained_var = explained_variance(values_flat, returns_flat)
        
        # Logging
        if self.logger is not None:
            # Record NaN statistics for monitoring
            if nan_batches_skipped > 0:
                self.logger.record('train/nan_batches_skipped', nan_batches_skipped)
            
            # Core training metrics
            self.logger.record('train/entropy_loss', np.mean(entropy_losses) if entropy_losses else 0.0)
            self.logger.record('train/entropy', np.mean(entropy_values) if entropy_values else 0.0)
            self.logger.record('train/policy_loss', np.mean(pg_losses) if pg_losses else 0.0)
            self.logger.record('train/value_loss', np.mean(value_losses) if value_losses else 0.0)
            self.logger.record('train/loss', loss.item() if 'loss' in dir() else 0.0)
            
            # Explained variance
            self.logger.record('train/explained_variance', explained_var)
            
            # Gradient norms
            self.logger.record('train/grad_norm', np.mean(grad_norms) if grad_norms else 0.0)
            
            # Training progress
            self.logger.record('train/n_updates', self._n_updates)
            self.logger.record('train/learning_rate', lr)
            self.logger.record('train/ent_coef', current_ent_coef)
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data for saving."""
        return {
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'ent_coef': (
                float(self.ent_coef)
                if isinstance(self.ent_coef, (int, float, np.floating))
                else None
            ),
            'ent_coef_final': self.ent_coef_final,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'normalize_advantage': self.normalize_advantage,
        }
