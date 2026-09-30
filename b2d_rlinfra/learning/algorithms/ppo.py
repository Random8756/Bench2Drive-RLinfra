"""Proximal Policy Optimization (PPO).

Clipped objective + GAE, operating on the per-worker rollout buffer
collected through ``StandardEnvAdapter``.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Type, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

from gymnasium import spaces

from .on_policy_algorithm import OnPolicyAlgorithm
from ..policies.actor_critic_policy_v2 import ActorCriticPolicyV2
from ..utils.schedules import get_schedule_fn
from ..utils.value_repr import (
    apply_value_transform,
    categorical_distribution_entropy,
    categorical_value_stats,
    scalar_to_twohot,
)

__layer__ = (4, "Algorithm")

if TYPE_CHECKING:
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer

# Use a dedicated logger name to avoid clashing with the envs layer.
logger = logging.getLogger("Policy")


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


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm.
    
    Features:
    - Clipped surrogate objective
    - Entropy regularization
    - GAE for advantage estimation
    """
    
    def __init__(
        self,
        policy: Type[ActorCriticPolicyV2] = ActorCriticPolicyV2,
        env = None,
        learning_rate: Union[float, callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, callable] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: Union[float, callable] = 0.0,
        ent_coef_final: Optional[float] = None,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
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
        Initialize PPO algorithm.
        
        Args:
            policy: Policy class.
            env: Environment.
            learning_rate: Learning rate or schedule.
            n_steps: Number of steps per rollout (total across all workers).
            batch_size: Minibatch size.
            n_epochs: Number of epochs per update.
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            clip_range: Clipping range for policy objective.
            normalize_advantage: Whether to normalize advantages.
            ent_coef: Entropy coefficient (float) or schedule callable.
            ent_coef_final: Optional final entropy coefficient for linear decay.
            vf_coef: Value function coefficient.
            max_grad_norm: Maximum gradient norm.
            target_kl: Target KL divergence (None = disabled).
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
        
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.ent_coef_final = ent_coef_final
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """Setup policy and initialize schedules."""
        super()._setup_model()
        
        # Setup clip range schedule
        self.clip_range_schedule = get_schedule_fn(self.clip_range)
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

        # Log hyperparameters once at startup
        hp_lines = [
            f"gamma={self.gamma}, gae_lambda={self.gae_lambda}, "
            f"vf_coef={self.vf_coef}, max_grad_norm={self.max_grad_norm}",
            f"batch_size={self.batch_size}, n_epochs={self.n_epochs}, n_steps={self.n_steps}",
            f"normalize_advantage={self.normalize_advantage}",
        ]
        if self.target_kl is not None:
            hp_lines.append(f"target_kl={self.target_kl}")
        logger.info("[PPO] Hyperparameters: %s", " | ".join(hp_lines))

    @property
    def uses_categorical_value_head(self) -> bool:
        """Whether PPO is using a categorical critic head."""
        return getattr(self.policy, 'uses_categorical_value_head', False)

    @property
    def uses_beta_distribution(self) -> bool:
        """Whether PPO is using a bounded Beta action distribution."""
        return getattr(self.policy, 'uses_beta_distribution', False)

    @property
    def uses_signed_pedal_action_semantics(self) -> bool:
        """Whether action dimensions are ``[signed_pedal, steer]``."""
        env_config = getattr(self.config, 'env_config', None)
        if env_config is None and isinstance(self.config, dict):
            env_config = self.config.get('env', self.config)
        if not isinstance(env_config, dict):
            return False
        action_type = (env_config.get('action_space') or {}).get('type')
        return action_type in (
            'continuous_signed_pedal_steer',
            'continuous_accelerate_steering_rate',
        )

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
        
        Performs n_epochs of optimization on the rollout buffer.
        """
        self.policy.set_training_mode(True)
        
        # Update learning rate
        lr = self.get_lr_schedule_fn()(self._current_progress_remaining)
        for param_group in self.policy.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get current clip range
        clip_range = self.clip_range_schedule(self._current_progress_remaining)
        current_ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        
        # Training statistics
        entropy_losses = []
        entropy_values = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        approx_kl_divs = []
        exact_kl_divs = []
        grad_norms = []
        value_drifts = []
        value_pred_means_raw = []
        value_pred_means_transformed = []
        value_target_means_raw = []
        value_target_means_transformed = []
        value_support_clip_fracs = []
        value_dist_entropies = []
        nan_batches_skipped = 0
        
        continue_training = True
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            if not continue_training:
                break
            
            # Iterate over minibatches
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Check for NaN in input data and skip problematic batches
                def _has_nan_or_inf(data):
                    if isinstance(data, dict):
                        return any(
                            torch.isnan(v).any() or torch.isinf(v).any()
                            for v in data.values()
                        )
                    return torch.isnan(data).any() or torch.isinf(data).any()
                
                if (_has_nan_or_inf(rollout_data.observations) or
                    _has_nan_or_inf(rollout_data.advantages) or
                    _has_nan_or_inf(rollout_data.returns)):
                    nan_batches_skipped += 1
                    if nan_batches_skipped == 1:
                        logger.warning("Skipping batch with NaN/Inf values in rollout data")
                    continue
                
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                
                # Evaluate actions with try-except to catch NaN errors
                try:
                    if self.uses_categorical_value_head:
                        if self.uses_beta_distribution:
                            values, value_logits, log_prob, entropy, action_dist_params = (
                                self.policy.evaluate_actions_with_value_logits_and_dist_params(
                                    rollout_data.observations,
                                    actions,
                                )
                            )
                        else:
                            action_dist_params = None
                            values, value_logits, log_prob, entropy = (
                                self.policy.evaluate_actions_with_value_logits(
                                    rollout_data.observations,
                                    actions,
                                )
                            )
                    else:
                        value_logits = None
                        if self.uses_beta_distribution:
                            values, log_prob, entropy, action_dist_params = (
                                self.policy.evaluate_actions_with_dist_params(
                                    rollout_data.observations,
                                    actions,
                                )
                            )
                        else:
                            action_dist_params = None
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
                
                # Check for NaN in policy output
                if _has_nan_or_inf(values) or _has_nan_or_inf(log_prob):
                    nan_batches_skipped += 1
                    logger.warning("NaN detected in values/log_prob, skipping batch")
                    continue
                if value_logits is not None and _has_nan_or_inf(value_logits):
                    nan_batches_skipped += 1
                    logger.warning("NaN detected in categorical value logits, skipping batch")
                    continue

                exact_kl = None
                if self.uses_beta_distribution:
                    if rollout_data.old_action_dist_params is None or action_dist_params is None:
                        raise RuntimeError(
                            "Beta PPO requires old and current action distribution parameters "
                            "for exact KL computation"
                        )
                    with torch.no_grad():
                        old_alpha = rollout_data.old_action_dist_params[:, 0, :]
                        old_beta = rollout_data.old_action_dist_params[:, 1, :]
                        new_alpha = action_dist_params[:, 0, :]
                        new_beta = action_dist_params[:, 1, :]
                        exact_kl = kl_divergence(
                            Beta(old_alpha, old_beta),
                            Beta(new_alpha, new_beta),
                        ).sum(dim=-1).mean().item()
                    if np.isfinite(exact_kl):
                        exact_kl_divs.append(exact_kl)
                    else:
                        nan_batches_skipped += 1
                        logger.warning("Non-finite exact Beta KL detected, skipping batch")
                        continue

                    if self.target_kl is not None and exact_kl > 1.5 * self.target_kl:
                        continue_training = False
                        logger.info(
                            f"Early stopping at epoch {epoch} before optimizer step due "
                            f"to reaching max exact Beta KL: {exact_kl:.4f}"
                        )
                        break
                
                # Track value drift: |V_current - V_old|
                with torch.no_grad():
                    value_drift = torch.abs(values.flatten() - rollout_data.old_values).mean().item()
                    if not np.isnan(value_drift):
                        value_drifts.append(value_drift)
                
                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    adv_std = advantages.std()
                    if adv_std > 1e-8:  # Avoid division by very small numbers
                        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                    else:
                        advantages = advantages - advantages.mean()
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                
                
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Check for NaN in loss
                if torch.isnan(policy_loss):
                    nan_batches_skipped += 1
                    logger.warning("NaN in policy loss, skipping batch")
                    continue
                
                pg_losses.append(policy_loss.item())
                
                # Clip fraction (for logging)
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Value loss
                if self.uses_categorical_value_head:
                    target_probs, _, support_clip_frac = (
                        self._build_categorical_targets(rollout_data.returns)
                    )
                    value_log_probs = F.log_softmax(value_logits, dim=-1)
                    per_sample_value_loss = -(
                        target_probs * value_log_probs
                    ).sum(dim=-1)
                    value_loss = per_sample_value_loss.mean()
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
                    per_sample_value_loss = F.mse_loss(
                        values,
                        rollout_data.returns,
                        reduction='none',
                    )
                    value_loss = per_sample_value_loss.mean()
                    value_losses.append(value_loss.item())

                    with torch.no_grad():
                        value_pred_means_raw.append(values.mean().item())
                        value_pred_means_transformed.append(
                            apply_value_transform(
                                values,
                                self.policy.value_transform,
                            ).mean().item()
                        )
                        value_target_means_raw.append(rollout_data.returns.mean().item())
                        value_target_means_transformed.append(
                            apply_value_transform(
                                rollout_data.returns,
                                self.policy.value_transform,
                            ).mean().item()
                        )
                
                # Entropy loss
                if entropy is None:
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
                
                # Check for NaN gradients and skip update if found
                has_nan_grad = False
                for param in self.policy.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    nan_batches_skipped += 1
                    logger.warning("NaN gradients detected, skipping update")
                    self.policy.optimizer.zero_grad()  # Clear the bad gradients
                    continue
                
                # Clip gradients and record the norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                grad_norms.append(grad_norm.item())
                
                self.policy.optimizer.step()
                
                # Compute approximate KL divergence
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    if not np.isnan(approx_kl):
                        approx_kl_divs.append(approx_kl)
                
                # Non-Beta policies retain the sample-based PPO approximation
                # and its existing post-step stopping behavior.
                if (
                    exact_kl is None
                    and self.target_kl is not None
                    and approx_kl > 1.5 * self.target_kl
                ):
                    continue_training = False
                    logger.info(
                        f"Early stopping at epoch {epoch} due to reaching max "
                        f"approximate KL: {approx_kl:.4f}"
                    )
                    break
            
        self._n_updates += 1
        
        # Log NaN statistics
        if nan_batches_skipped > 0:
            logger.warning(f"Skipped {nan_batches_skipped} batches due to NaN/Inf values this iteration")
        
        # Compute explained variance (how well value function predicts returns)
        explained_var = np.nan
        if self.rollout_buffer.values is not None and self.rollout_buffer.returns is not None:
            values_flat = self.rollout_buffer.values.flatten()
            returns_flat = self.rollout_buffer.returns.flatten()
            if len(values_flat) > 0 and len(returns_flat) > 0:
                explained_var = explained_variance(values_flat, returns_flat)
        
        # Compute advantages_tail_cv (coefficient of variation for last 100 advantages)
        advantages_tail_cv = np.nan
        if self.rollout_buffer.advantages is not None:
            adv_flat = self.rollout_buffer.advantages.flatten()
            if len(adv_flat) >= 100:
                adv_tail = adv_flat[-100:]
                adv_mean = np.mean(adv_tail)
                adv_std = np.std(adv_tail)
                if abs(adv_mean) > 1e-8:
                    advantages_tail_cv = adv_std / abs(adv_mean)

        beta_metrics = {}
        if (
            self.uses_beta_distribution
            and self.rollout_buffer.actions is not None
            and self.rollout_buffer.action_dist_params is not None
        ):
            actions_flat = torch.as_tensor(self.rollout_buffer.actions).reshape(
                -1, self.policy.action_dim,
            )
            params_flat = torch.as_tensor(self.rollout_buffer.action_dist_params)
            alpha_flat = params_flat[:, 0, :]
            beta_flat = params_flat[:, 1, :]
            action_low = self.policy.action_low.detach().cpu()
            action_scale = self.policy.action_scale.detach().cpu()
            normalized_actions = (actions_flat - action_low) / action_scale
            beta_metrics = {
                'train/beta_alpha_mean': alpha_flat.mean().item(),
                'train/beta_alpha_min': alpha_flat.min().item(),
                'train/beta_beta_mean': beta_flat.mean().item(),
                'train/beta_beta_min': beta_flat.min().item(),
                'rollout/action_near_bound_fraction': (
                    (normalized_actions < 0.025) | (normalized_actions > 0.975)
                ).float().mean().item(),
            }
            for action_idx in range(actions_flat.shape[1]):
                beta_metrics[f'rollout/action_{action_idx}_mean'] = (
                    actions_flat[:, action_idx].mean().item()
                )
                beta_metrics[f'rollout/action_{action_idx}_std'] = (
                    actions_flat[:, action_idx].std(unbiased=False).item()
                )
            if (
                self.uses_signed_pedal_action_semantics
                and actions_flat.shape[1] >= 2
            ):
                signed_pedal = actions_flat[:, 0]
                steer = actions_flat[:, 1]
                beta_metrics['rollout/brake_fraction'] = (
                    signed_pedal < 0.0
                ).float().mean().item()
                beta_metrics['rollout/signed_pedal_mean'] = signed_pedal.mean().item()
                beta_metrics['rollout/signed_pedal_std'] = (
                    signed_pedal.std(unbiased=False).item()
                )
                beta_metrics['rollout/steer_mean'] = steer.mean().item()
                beta_metrics['rollout/steer_std'] = steer.std(unbiased=False).item()
                beta_metrics['rollout/high_throttle_high_steer_fraction'] = (
                    (signed_pedal > 0.7) & (steer.abs() > 0.7)
                ).float().mean().item()
        
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
            self.logger.record('train/approx_kl', np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
            if self.uses_beta_distribution:
                self.logger.record('train/exact_kl', np.mean(exact_kl_divs) if exact_kl_divs else 0.0)
            self.logger.record('train/clip_fraction', np.mean(clip_fractions) if clip_fractions else 0.0)
            self.logger.record('train/loss', loss.item() if 'loss' in dir() else 0.0)
            
            # Explained variance
            self.logger.record('train/explained_variance', explained_var)
            
            # Gradient norms
            self.logger.record('train/grad_norm', np.mean(grad_norms) if grad_norms else 0.0)
            
            # Training progress
            self.logger.record('train/n_updates', self._n_updates)
            self.logger.record('train/learning_rate', lr)
            self.logger.record('train/ent_coef', current_ent_coef)
            self.logger.record('train/clip_range', clip_range)

            for metric_name, metric_value in beta_metrics.items():
                self.logger.record(metric_name, metric_value)
            
            # Hyperparameters are logged once at _setup_model(), not per iteration
    
    def _get_save_data(self) -> Dict[str, Any]:
        """Get algorithm-specific data for saving."""
        return {
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'normalize_advantage': self.normalize_advantage,
            'ent_coef': (
                float(self.ent_coef)
                if isinstance(self.ent_coef, (int, float, np.floating))
                else None
            ),
            'ent_coef_final': self.ent_coef_final,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
        }
