"""Training sub-process for asynchronous SAC / TD3 updates.

Receives train commands from the collect process, samples shared replay, and
publishes updated weights through shared-memory files.
"""

from __future__ import annotations

import logging
import mmap
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger("Training Loop")

__layer__ = (4, "Algorithm")



class TrainCmd(Enum):
    """Commands sent from main process to train process."""
    TRAIN = auto()      # Start a training round
    SHUTDOWN = auto()   # Gracefully shut down


class TrainResult(Enum):
    """Results sent from train process back to main."""
    DONE = auto()       # Training round completed
    ERROR = auto()      # Training round failed


@dataclass
class OffPolicyTrainConfig:
    """All algorithm hyper-parameters needed by the train sub-process.

    The main process builds this once at start-up (in
    ``OffPolicyAlgorithm._ensure_train_process``) and passes it through
    ``mp.Process(kwargs=...)``. After that the only per-round parameters are
    the ones in the ``TrainCmd.TRAIN`` message: ``batch_size``,
    ``gradient_steps``, ``num_timesteps``, ``learning_rate``.

    All fields are picklable (basic types or lists of floats) so the
    dataclass can cross the spawn boundary.
    """
    algorithm_type: str  # 'sac' or 'td3'
    gamma: float
    tau: float

    # BC loss
    bc_enabled: bool = False
    bc_lambda_initial: float = 1.0
    bc_lambda_decay_steps: int = 200_000
    bc_lambda_final: float = 0.0
    bc_prior_action: List[float] = field(default_factory=lambda: [0.6, 0.0])
    bc_source: str = 'fixed'  # 'fixed' | 'lqr'

    # SAC
    target_update_interval: int = 1
    target_entropy: Union[str, float] = 'auto'
    ent_coef: Union[str, float] = 'auto'
    moe_aux_loss_weight: float = 0.0

    # TD3
    policy_delay: int = 2
    target_policy_noise: List[float] = field(default_factory=list)
    target_noise_clip: List[float] = field(default_factory=list)

    # Initial subprocess state
    n_updates: int = 0
    verbose: int = 1


def _log_level_from_verbose(verbose: int) -> int:
    try:
        verbose_value = int(verbose)
    except (TypeError, ValueError):
        verbose_value = 1
    return {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(verbose_value, logging.INFO)



def _train_td3(
    policy,
    buffer_view,
    batch_size: int,
    gradient_steps: int,
    num_timesteps: int,
    learning_rate: float,
    device: torch.device,
    cfg: OffPolicyTrainConfig,
    n_updates: int,
):
    """TD3 training loop: twin critics, delayed actor update, target smoothing."""
    import torch.nn.functional as F

    actor = policy.actor
    actor_target = policy.actor_target
    critic = policy.critic
    critic_target = policy.critic_target

    moe_apply = cfg.moe_aux_loss_weight > 0.0

    target_policy_noise = np.array(cfg.target_policy_noise, dtype=np.float32)
    target_noise_clip = np.array(cfg.target_noise_clip, dtype=np.float32)

    bc_prior_tensor = None
    if cfg.bc_enabled and cfg.bc_source != 'lqr':
        prior_env = np.array(cfg.bc_prior_action, dtype=np.float32)
        prior_normalized = policy.unscale_action(prior_env)
        bc_prior_tensor = torch.as_tensor(prior_normalized, dtype=torch.float32, device=device)

    if cfg.bc_enabled:
        progress = min(1.0, num_timesteps / max(1, cfg.bc_lambda_decay_steps))
        bc_lambda = cfg.bc_lambda_initial + (cfg.bc_lambda_final - cfg.bc_lambda_initial) * progress
    else:
        bc_lambda = 0.0

    for pg in policy.actor_optimizer.param_groups:
        pg['lr'] = learning_rate
    for pg in policy.critic_optimizer.param_groups:
        pg['lr'] = learning_rate

    actor_losses: List[float] = []
    critic_losses: List[float] = []
    bc_losses: List[float] = []
    q_preds: List[np.ndarray] = []
    q_tgts: List[np.ndarray] = []
    actor_moe_aux_losses: List[float] = []
    critic_moe_aux_losses: List[float] = []
    last_actor_moe_stats: Optional[Dict[str, Any]] = None
    last_critic_moe_stats: Optional[Dict[str, Any]] = None
    t_sample_total = 0.0
    sample_count = 0

    for _step in range(gradient_steps):
        n_updates += 1

        t_s = time.perf_counter()
        replay_data = buffer_view.sample(batch_size)
        t_sample_total += time.perf_counter() - t_s
        sample_count += 1

        is_weights = replay_data.weights
        tree_indices = replay_data.indices

        with torch.no_grad():
            noise_std = torch.as_tensor(target_policy_noise, device=device)
            noise_clip_t = torch.as_tensor(target_noise_clip, device=device)
            noise = torch.randn_like(replay_data.actions) * noise_std
            noise = noise.clamp(-noise_clip_t, noise_clip_t)

            next_features = policy.features_extractor(replay_data.next_observations)
            next_actions = (actor_target(next_features) + noise).clamp(-1, 1)

            next_q_values = critic_target.q_values(next_features, next_actions)
            next_q_values = torch.cat(next_q_values, dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)

            target_q_values = replay_data.rewards + (1 - replay_data.dones) * cfg.gamma * next_q_values

        current_features = policy.features_extractor(replay_data.observations)
        current_q_outputs = critic(current_features, replay_data.actions)

        use_dist = critic.use_distributional
        if use_dist:
            dist_helper = critic.distributional
            critic_loss = sum(
                dist_helper.loss(q_logits, target_q_values, weights=is_weights)
                for q_logits in current_q_outputs
            )
        else:
            critic_loss = sum(
                (is_weights * F.mse_loss(cq, target_q_values, reduction='none')).mean()
                for cq in current_q_outputs
            )
        critic_losses.append(critic_loss.item())

        if moe_apply:
            critic_aux = policy.critic_moe_aux_loss()
            if critic_aux is not None:
                critic_loss = critic_loss + cfg.moe_aux_loss_weight * critic_aux
                critic_moe_aux_losses.append(float(critic_aux.item()))
                last_critic_moe_stats = policy.critic_moe_stats()

        if tree_indices is not None:
            with torch.no_grad():
                if use_dist:
                    current_q_decoded = dist_helper.decode(current_q_outputs[0])
                else:
                    current_q_decoded = current_q_outputs[0]
                td_errors = (current_q_decoded - target_q_values).abs().cpu().numpy()
            buffer_view.update_priorities(tree_indices, td_errors)

        with torch.no_grad():
            if use_dist:
                q_preds.append(dist_helper.decode(current_q_outputs[0]).cpu().numpy())
            else:
                q_preds.append(current_q_outputs[0].cpu().numpy())
            q_tgts.append(target_q_values.cpu().numpy())

        policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        policy.critic_optimizer.step()

        if n_updates % cfg.policy_delay == 0:
            current_features = policy.features_extractor(replay_data.observations)
            actor_actions = actor(current_features)
            # for_actor=True: decode in symlog space (no symexp) for stable gradients
            q1_values = critic.q1_forward(current_features, actor_actions, for_actor=True)
            td3_actor_loss = -q1_values.mean()

            # Always normalize RL loss by |Q| (TD3+BC style) for a stable
            # actor-gradient scale across training.
            with torch.no_grad():
                q_abs_mean = torch.abs(q1_values).mean().clamp(min=1.0)
            actor_loss = td3_actor_loss / q_abs_mean

            if cfg.bc_enabled and bc_lambda > 0:
                if cfg.bc_source == 'lqr' and replay_data.expert_actions is not None:
                    prior = replay_data.expert_actions
                else:
                    prior = bc_prior_tensor.unsqueeze(0).expand_as(actor_actions)
                bc_loss = F.mse_loss(actor_actions, prior)
                bc_losses.append(bc_loss.item())
                actor_loss = actor_loss + bc_lambda * bc_loss

            if moe_apply:
                actor_aux = policy.actor_moe_aux_loss()
                if actor_aux is not None:
                    actor_loss = actor_loss + cfg.moe_aux_loss_weight * actor_aux
                    actor_moe_aux_losses.append(float(actor_aux.item()))
                    last_actor_moe_stats = policy.actor_moe_stats()

            actor_losses.append(td3_actor_loss.item())

            policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            policy.actor_optimizer.step()

            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

    y_pred = np.concatenate(q_preds, axis=0).flatten()
    y_true = np.concatenate(q_tgts, axis=0).flatten()
    var_y = np.var(y_true)
    explained_var = (1 - np.var(y_true - y_pred) / var_y) if var_y > 1e-8 else float('nan')

    result: Dict[str, Any] = {
        'n_updates': n_updates,
        'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
        'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
        'bc_loss': float(np.mean(bc_losses)) if bc_losses else 0.0,
        'bc_lambda': bc_lambda,
        'explained_var': explained_var,
        'time_sample': t_sample_total,
        'sample_count': sample_count,
    }
    if actor_moe_aux_losses or critic_moe_aux_losses:
        result['moe_aux_loss_weight'] = cfg.moe_aux_loss_weight
    if actor_moe_aux_losses:
        result['moe_actor_aux_loss'] = float(np.mean(actor_moe_aux_losses))
        if last_actor_moe_stats is not None:
            result['moe_actor_gate_entropy'] = float(last_actor_moe_stats['gate_entropy'])
            result['moe_actor_usage'] = list(last_actor_moe_stats['usage'])
    if critic_moe_aux_losses:
        result['moe_critic_aux_loss'] = float(np.mean(critic_moe_aux_losses))
        if last_critic_moe_stats is not None:
            result['moe_critic_gate_entropy'] = float(last_critic_moe_stats['gate_entropy'])
            result['moe_critic_usage'] = list(last_critic_moe_stats['usage'])
    return result


def _train_sac(
    policy,
    buffer_view,
    batch_size: int,
    gradient_steps: int,
    num_timesteps: int,
    learning_rate: float,
    device: torch.device,
    cfg: OffPolicyTrainConfig,
    n_updates: int,
    persistent_state: Dict[str, Any],
):
    """SAC training loop: entropy regularisation, twin critics, stochastic actor.

    Args:
        persistent_state: Mutable dict that survives between training rounds.
            Used to keep ``log_ent_coef`` and its optimizer alive so the
            entropy coefficient adapts continuously across rounds.
    """
    import torch.nn.functional as F

    actor = policy.actor
    critic = policy.critic
    critic_target = policy.critic_target

    moe_apply = cfg.moe_aux_loss_weight > 0.0

    if cfg.target_entropy == "auto":
        target_entropy = -float(np.prod(policy.action_space.shape))
    else:
        target_entropy = float(cfg.target_entropy)

    if 'log_ent_coef' in persistent_state:
        log_ent_coef = persistent_state['log_ent_coef']
        ent_coef_optimizer = persistent_state['ent_coef_optimizer']
        ent_coef_tensor = persistent_state.get('ent_coef_tensor', None)
        if ent_coef_optimizer is not None:
            for pg in ent_coef_optimizer.param_groups:
                pg['lr'] = learning_rate
    else:
        ent_coef_setting = cfg.ent_coef
        if isinstance(ent_coef_setting, str) and ent_coef_setting.startswith("auto"):
            init_value = 1.0
            if "_" in ent_coef_setting:
                init_value = float(ent_coef_setting.split("_")[1])
            log_ent_coef = torch.log(
                torch.ones(1, device=device) * init_value
            ).requires_grad_(True)
            ent_coef_optimizer = torch.optim.Adam([log_ent_coef], lr=learning_rate)
            ent_coef_tensor = None
        else:
            log_ent_coef = None
            ent_coef_optimizer = None
            ent_coef_tensor = torch.tensor(float(ent_coef_setting), device=device)
        persistent_state['log_ent_coef'] = log_ent_coef
        persistent_state['ent_coef_optimizer'] = ent_coef_optimizer
        persistent_state['ent_coef_tensor'] = ent_coef_tensor

    bc_prior_tensor = None
    if cfg.bc_enabled and cfg.bc_source != 'lqr':
        prior_env = np.array(cfg.bc_prior_action, dtype=np.float32)
        prior_normalized = policy.unscale_action(prior_env)
        bc_prior_tensor = torch.as_tensor(prior_normalized, dtype=torch.float32, device=device)

    if cfg.bc_enabled:
        progress = min(1.0, num_timesteps / max(1, cfg.bc_lambda_decay_steps))
        bc_lambda = cfg.bc_lambda_initial + (cfg.bc_lambda_final - cfg.bc_lambda_initial) * progress
    else:
        bc_lambda = 0.0

    for pg in policy.actor_optimizer.param_groups:
        pg['lr'] = learning_rate
    for pg in policy.critic_optimizer.param_groups:
        pg['lr'] = learning_rate
    if ent_coef_optimizer is not None:
        for pg in ent_coef_optimizer.param_groups:
            pg['lr'] = learning_rate

    ent_coef_losses: List[float] = []
    ent_coefs: List[float] = []
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    bc_losses: List[float] = []
    q_preds: List[np.ndarray] = []
    q_tgts: List[np.ndarray] = []
    actor_moe_aux_losses: List[float] = []
    critic_moe_aux_losses: List[float] = []
    last_actor_moe_stats: Optional[Dict[str, Any]] = None
    last_critic_moe_stats: Optional[Dict[str, Any]] = None
    t_sample_total = 0.0
    sample_count = 0

    for step_idx in range(gradient_steps):
        n_updates += 1

        t_s = time.perf_counter()
        replay_data = buffer_view.sample(batch_size)
        t_sample_total += time.perf_counter() - t_s
        sample_count += 1

        is_weights = replay_data.weights
        tree_indices = replay_data.indices

        if ent_coef_optimizer is not None and log_ent_coef is not None:
            ent_coef = torch.exp(log_ent_coef.detach())
        else:
            ent_coef = ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Entropy coefficient
        features = policy.features_extractor(replay_data.observations)
        actions_pi, log_prob = actor.action_log_prob(features)
        log_prob = log_prob.reshape(-1, 1)

        if ent_coef_optimizer is not None and log_ent_coef is not None:
            ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            ent_coef_optimizer.step()

        # Critic
        with torch.no_grad():
            next_features = policy.features_extractor(replay_data.next_observations)
            next_actions, next_log_prob = actor.action_log_prob(next_features)
            next_log_prob = next_log_prob.reshape(-1, 1)

            next_q_values = critic_target.q_values(next_features, next_actions)
            next_q_values = torch.cat(next_q_values, dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)

            next_q_values = next_q_values - ent_coef * next_log_prob

            target_q_values = replay_data.rewards + (1 - replay_data.dones) * cfg.gamma * next_q_values

        current_features = policy.features_extractor(replay_data.observations)
        current_q_outputs = critic(current_features, replay_data.actions)

        use_dist = critic.use_distributional
        if use_dist:
            dist_helper = critic.distributional
            critic_loss = 0.5 * sum(
                dist_helper.loss(q_logits, target_q_values, weights=is_weights)
                for q_logits in current_q_outputs
            )
        else:
            critic_loss = 0.5 * sum(
                (is_weights * F.mse_loss(cq, target_q_values, reduction='none')).mean()
                for cq in current_q_outputs
            )
        critic_losses.append(critic_loss.item())

        if moe_apply:
            critic_aux = policy.critic_moe_aux_loss()
            if critic_aux is not None:
                critic_loss = critic_loss + cfg.moe_aux_loss_weight * critic_aux
                critic_moe_aux_losses.append(float(critic_aux.item()))
                last_critic_moe_stats = policy.critic_moe_stats()

        if tree_indices is not None:
            with torch.no_grad():
                if use_dist:
                    current_q_decoded = dist_helper.decode(current_q_outputs[0])
                else:
                    current_q_decoded = current_q_outputs[0]
                td_errors = (current_q_decoded - target_q_values).abs().cpu().numpy()
            buffer_view.update_priorities(tree_indices, td_errors)

        with torch.no_grad():
            if use_dist:
                q_preds.append(dist_helper.decode(current_q_outputs[0]).cpu().numpy())
            else:
                q_preds.append(current_q_outputs[0].cpu().numpy())
            q_tgts.append(target_q_values.cpu().numpy())

        policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        policy.critic_optimizer.step()

        # Actor
        current_features = policy.features_extractor(replay_data.observations)
        actions_pi, log_prob = actor.action_log_prob(current_features)
        log_prob = log_prob.reshape(-1, 1)

        # for_actor=True: decode in symlog space (no symexp) for stable gradients
        q_values_pi = critic.q_values(current_features, actions_pi, for_actor=True)
        q_values_pi = torch.cat(q_values_pi, dim=1)
        min_q_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)

        sac_actor_loss = (ent_coef * log_prob - min_q_pi).mean()

        with torch.no_grad():
            q_abs_mean = torch.abs(min_q_pi).mean().clamp(min=1.0)
        actor_loss = sac_actor_loss / q_abs_mean

        if cfg.bc_enabled and bc_lambda > 0:
            if cfg.bc_source == 'lqr' and replay_data.expert_actions is not None:
                prior = replay_data.expert_actions
            else:
                prior = bc_prior_tensor.unsqueeze(0).expand_as(actions_pi)
            bc_loss = F.mse_loss(actions_pi, prior)
            bc_losses.append(bc_loss.item())
            actor_loss = actor_loss + bc_lambda * bc_loss

        if moe_apply:
            actor_aux = policy.actor_moe_aux_loss()
            if actor_aux is not None:
                actor_loss = actor_loss + cfg.moe_aux_loss_weight * actor_aux
                actor_moe_aux_losses.append(float(actor_aux.item()))
                last_actor_moe_stats = policy.actor_moe_stats()

        actor_losses.append(sac_actor_loss.item())

        policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        policy.actor_optimizer.step()

        if step_idx % cfg.target_update_interval == 0:
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

    y_pred = np.concatenate(q_preds, axis=0).flatten()
    y_true = np.concatenate(q_tgts, axis=0).flatten()
    var_y = np.var(y_true)
    explained_var = (1 - np.var(y_true - y_pred) / var_y) if var_y > 1e-8 else float('nan')

    result: Dict[str, Any] = {
        'n_updates': n_updates,
        'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
        'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
        'bc_loss': float(np.mean(bc_losses)) if bc_losses else 0.0,
        'bc_lambda': bc_lambda,
        'ent_coef': float(np.mean(ent_coefs)) if ent_coefs else 0.0,
        'explained_var': explained_var,
        'time_sample': t_sample_total,
        'sample_count': sample_count,
    }
    if actor_moe_aux_losses or critic_moe_aux_losses:
        result['moe_aux_loss_weight'] = cfg.moe_aux_loss_weight
    if actor_moe_aux_losses:
        result['moe_actor_aux_loss'] = float(np.mean(actor_moe_aux_losses))
        if last_actor_moe_stats is not None:
            result['moe_actor_gate_entropy'] = float(last_actor_moe_stats['gate_entropy'])
            result['moe_actor_usage'] = list(last_actor_moe_stats['usage'])
    if critic_moe_aux_losses:
        result['moe_critic_aux_loss'] = float(np.mean(critic_moe_aux_losses))
        if last_critic_moe_stats is not None:
            result['moe_critic_gate_entropy'] = float(last_critic_moe_stats['gate_entropy'])
            result['moe_critic_usage'] = list(last_critic_moe_stats['usage'])
    return result


_TRAIN_FN = {
    'td3': _train_td3,
    'sac': _train_sac,
}


# ---------------------------------------------------------------------------
# Train process entry point
# ---------------------------------------------------------------------------


def _train_process_entry(
    cmd_queue,
    result_queue,
    buffer_config,
    policy_class,
    policy_kwargs,
    observation_space,
    action_space,
    device_str: str,
    weight_shm_name: str,
    weight_keys,
    weight_shapes,
    weight_dtypes,
    weight_total_elements: int,
    train_config: OffPolicyTrainConfig,
):
    """Entry point for the training sub-process (spawned)."""
    try:
        log_level = _log_level_from_verbose(getattr(train_config, 'verbose', 1))
        logging.basicConfig(
            level=log_level,
            format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
        for logger_name in ("Training Loop", "Policy"):
            logging.getLogger(logger_name).setLevel(log_level)
        proc_logger = logging.getLogger("Training Loop")

        algorithm_type = train_config.algorithm_type
        proc_logger.info(
            "Train process started, PID=%d, device=%s, algorithm=%s",
            os.getpid(), device_str, algorithm_type,
        )

        device = torch.device(device_str)
        if device.type == 'cuda':
            dev_idx = device.index if device.index is not None else 0
            torch.cuda.set_device(dev_idx)
            torch.cuda.empty_cache()
            device = torch.device('cuda', dev_idx)

        # ``buffer_config`` is either a raw SharedPrioritizedReplayBuffer
        # config, or a MixedReplayBuffer config (detected by the presence of
        # a nested 'dynamic' dict).
        from b2d_rlinfra.learning.buffers.shared_buffer import SharedPrioritizedReplayBuffer
        is_mixed_buffer = isinstance(buffer_config, dict) and ("dynamic" in buffer_config)
        if is_mixed_buffer:
            from b2d_rlinfra.learning.buffers.mixed_buffer import MixedReplayBuffer
            buffer_view = MixedReplayBuffer.from_shared(buffer_config, device=device)
            proc_logger.info(
                "Attached to MixedReplayBuffer, dynamic_size=%d, static_scenarios=%d",
                buffer_view.size(), len(buffer_view.static_buffers),
            )
        else:
            buffer_view = SharedPrioritizedReplayBuffer.from_shared(buffer_config, device=device)
            proc_logger.debug("Attached to shared buffer, size=%d", buffer_view.size())

        lr_schedule = lambda _: 1.0  # value overridden each round
        policy = policy_class(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **policy_kwargs,
        )
        policy.to(device)

        nbytes = weight_total_elements * 4  # float32
        weight_path = os.path.join('/dev/shm', weight_shm_name)
        weight_fd = os.open(weight_path, os.O_RDWR)
        weight_mm = mmap.mmap(weight_fd, nbytes)
        weight_flat = np.frombuffer(weight_mm, dtype=np.float32)

        def _read_weights() -> Dict[str, torch.Tensor]:
            sd: Dict[str, torch.Tensor] = {}
            offset = 0
            for key in weight_keys:
                shape = weight_shapes[key]
                n = int(np.prod(shape))
                arr = weight_flat[offset:offset + n].copy()
                tensor = torch.from_numpy(arr).reshape(shape)
                dtype_str = weight_dtypes[key]
                orig_dtype = (
                    getattr(torch, dtype_str.split('.')[-1])
                    if '.' in dtype_str
                    else torch.float32
                )
                if tensor.dtype != orig_dtype:
                    tensor = tensor.to(orig_dtype)
                sd[key] = tensor.to(device)
                offset += n
            return sd

        def _write_weights(sd: Dict[str, torch.Tensor]) -> None:
            offset = 0
            for key in weight_keys:
                flat = sd[key].detach().cpu().numpy().astype(np.float32).ravel()
                n = flat.size
                weight_flat[offset:offset + n] = flat
                offset += n

        policy.load_state_dict(_read_weights())
        proc_logger.info("Loaded initial policy weights from shared memory")

        train_fn = _TRAIN_FN.get(algorithm_type)
        if train_fn is None:
            raise ValueError(
                f"Unknown algorithm_type: {algorithm_type!r}. "
                f"Supported: {list(_TRAIN_FN.keys())}"
            )

        n_updates = train_config.n_updates
        persistent_state: Dict[str, Any] = {}

        proc_logger.info("Train process ready, waiting for commands...")

        while True:
            cmd_msg = cmd_queue.get()
            cmd = cmd_msg['cmd']

            if cmd == TrainCmd.SHUTDOWN:
                proc_logger.info("Received SHUTDOWN command, exiting")
                break

            if cmd != TrainCmd.TRAIN:
                continue

            batch_size = cmd_msg['batch_size']
            gradient_steps = cmd_msg['gradient_steps']
            num_timesteps = cmd_msg['num_timesteps']
            learning_rate = cmd_msg['learning_rate']
            static_ratio = cmd_msg.get('static_ratio', None)

            t_start = time.perf_counter()

            try:
                policy.train()

                # Propagate current dynamic/static split to the view.
                if is_mixed_buffer:
                    buffer_view.set_view_ratio_override(static_ratio)

                train_kwargs = dict(
                    policy=policy,
                    buffer_view=buffer_view,
                    batch_size=batch_size,
                    gradient_steps=gradient_steps,
                    num_timesteps=num_timesteps,
                    learning_rate=learning_rate,
                    device=device,
                    cfg=train_config,
                    n_updates=n_updates,
                )
                if algorithm_type == 'sac':
                    train_kwargs['persistent_state'] = persistent_state

                result = train_fn(**train_kwargs)
                n_updates = result['n_updates']

                _write_weights(policy.state_dict())

                t_total = time.perf_counter() - t_start
                msg: Dict[str, Any] = {
                    'result': TrainResult.DONE,
                    'time_total': t_total,
                    'time_sample': result.get('time_sample', 0.0),
                    'sample_count': result.get('sample_count', 0),
                    'n_updates': n_updates,
                    'critic_loss': result.get('critic_loss', 0.0),
                    'actor_loss': result.get('actor_loss', 0.0),
                    'bc_loss': result.get('bc_loss', 0.0),
                    'bc_lambda': result.get('bc_lambda', 0.0),
                    'ent_coef': result.get('ent_coef', None),
                    'explained_var': result.get('explained_var', None),
                    'learning_rate': learning_rate,
                }
                for moe_key in (
                    'moe_aux_loss_weight',
                    'moe_actor_aux_loss',
                    'moe_critic_aux_loss',
                    'moe_actor_gate_entropy',
                    'moe_critic_gate_entropy',
                    'moe_actor_usage',
                    'moe_critic_usage',
                ):
                    if moe_key in result:
                        msg[moe_key] = result[moe_key]
                result_queue.put(msg)

            except Exception as e:
                proc_logger.error("Training error: %s\n%s", e, traceback.format_exc())
                result_queue.put({
                    'result': TrainResult.ERROR,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })

        buffer_view.cleanup()
        try:
            weight_mm.close()
        except Exception:
            pass
        try:
            os.close(weight_fd)
        except Exception:
            pass
        proc_logger.info("Train process exiting cleanly")

    except Exception as e:
        logging.error("Fatal error in train process: %s\n%s", e, traceback.format_exc())
        try:
            result_queue.put({
                'result': TrainResult.ERROR,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })
        except Exception:
            pass


# Re-export for callers that want to inspect the dataclass externally.
__all__ = [
    'OffPolicyTrainConfig',
    'TrainCmd',
    'TrainResult',
    '_train_process_entry',
]


# Helper used only by tests / introspection.
def train_config_to_dict(cfg: OffPolicyTrainConfig) -> Dict[str, Any]:
    return asdict(cfg)
