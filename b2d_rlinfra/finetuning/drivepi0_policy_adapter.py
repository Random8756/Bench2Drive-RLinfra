"""DrivePi0 policy adapter for rl_finetune."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from b2d_rlinfra.finetuning.drivepi0_core import DrivePi0Runtime
from b2d_rlinfra.finetuning.drivepi0_obs_adapter import require_drivepi0_obs
from b2d_rlinfra.finetuning.policy_adapter import (
    DDPOptions,
    LearnerOutput,
    LearnerSpec,
    PolicyAdapter,
    PolicyStep,
)


class _DrivePi0LearnerModule(torch.nn.Module):
    """Explicit DrivePi0 PPO graph exposed to DDP by this adapter."""

    def __init__(self, adapter: "DrivePi0PolicyAdapter"):
        super().__init__()
        # Register the concrete model graph; the runtime reference is kept
        # outside nn.Module registration and only supplies model-specific
        # preprocessing/evaluation helpers.
        self.drivepi0_model = adapter.runtime.drivepi0_model
        self.value_net = adapter.runtime.value_net
        self.log_std = adapter.runtime.log_std
        object.__setattr__(self, "_adapter", adapter)

    def forward(self, batch: Dict[str, Any]) -> LearnerOutput:
        return self._adapter._learner_forward(batch)


class DrivePi0PolicyAdapter(PolicyAdapter):
    def __init__(
        self,
        *,
        runtime: DrivePi0Runtime,
        config: Mapping[str, Any],
        trajectory_shape: Tuple[int, int],
    ):
        self.runtime = runtime
        self.config = dict(config or {})
        self.device = runtime.device
        self.trajectory_shape = tuple(int(v) for v in trajectory_shape)
        self.policy_version = 0
        self.rgb_key = runtime.rgb_key
        self.state_key = runtime.state_key
        ppo_cfg = dict(self.config.get("ppo", {}) or {})
        self.mean_ref_l2_coef = float(
            ppo_cfg.get("mean_ref_l2_coef", self.config.get("mean_ref_l2_coef", 0.0))
        )
        self._learner_module = _DrivePi0LearnerModule(self)
        self._learner_spec = LearnerSpec(
            module=self._learner_module,
            optimizer=runtime.optimizer,
            precision="bf16" if bool(self.config.get("use_bf16", False)) else None,
            ddp_options=DDPOptions(broadcast_buffers=False),
            aux_loss_weights=(
                {"mean_ref_l2": self.mean_ref_l2_coef}
                if self.mean_ref_l2_coef != 0.0
                else {}
            ),
        )
        self._learner_spec.validate()

    @classmethod
    def load_initial(
        cls,
        config: Mapping[str, Any],
        checkpoint: Optional[str],
        device: torch.device,
    ) -> "DrivePi0PolicyAdapter":
        cfg = dict(config or {})
        if checkpoint:
            cfg["checkpoint_path"] = checkpoint
        traj_cfg = dict(cfg.get("trajectory", {}) or {})
        num_points = int(traj_cfg.get("num_points", 10))
        state_dim = int(traj_cfg.get("state_dim", 2))
        action_dim = num_points * state_dim
        runtime = DrivePi0Runtime(
            config=cfg,
            device=device,
            action_dim=action_dim,
            value_hidden_dim=int(cfg.get("value_hidden_dim", 512)),
            log_std_init=float(cfg.get("log_std_init", -2.0)),
            learning_rate=float(cfg.get("learning_rate", 1.0e-5)),
        )
        adapter = cls(
            runtime=runtime,
            config=cfg,
            trajectory_shape=(num_points, state_dim),
        )
        return adapter

    @torch.no_grad()
    def collect_step(self, obs: Any) -> PolicyStep:
        self.runtime.set_train(False)
        obs_mapping = require_drivepi0_obs(obs, rgb_key=self.rgb_key, state_key=self.state_key)
        policy_input_state = self.runtime.encode_observation(obs_mapping)
        dist = self.runtime.distribution_from_state(policy_input_state)
        mean_norm = dist.distribution.mean.detach()
        action_flat = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action_flat.unsqueeze(0)).squeeze(0)
        value = self.runtime.value_from_state(policy_input_state).squeeze(0)
        action_np = action_flat.detach().cpu().numpy().astype(np.float32)
        env_action = self.runtime.denormalize_trajectory(
            action_flat.reshape(1, *self.trajectory_shape)
        ).squeeze(0)
        env_action_np = env_action.detach().cpu().numpy().astype(np.float32)
        return PolicyStep(
            policy_input_state=policy_input_state,
            train_action=action_np,
            env_action=env_action_np,
            value=float(value.detach().cpu().item()),
            old_action_log_prob=float(log_prob.detach().cpu().item()),
            action_logprob_info={
                "mean_norm": mean_norm.squeeze(0).cpu().numpy().astype(np.float32),
            },
        )

    @torch.no_grad()
    def value_from_obs(self, obs: Any) -> float:
        obs_mapping = require_drivepi0_obs(obs, rgb_key=self.rgb_key, state_key=self.state_key)
        policy_input_state = self.runtime.encode_observation(obs_mapping)
        return float(self.runtime.value_from_state(policy_input_state).squeeze(0).detach().cpu().item())

    def _learner_forward(self, batch: Dict[str, Any]) -> LearnerOutput:
        self.runtime.set_train(True)
        # DiagGaussianDistribution is stateful: proba_distribution stores
        # Normal(mean, std) on self.action_dist.distribution. mean carries the
        # full DrivePi0 autograd graph. target-KL early-stop skips backward, so
        # that cached Normal would retain the graph across the next forward and
        # OOM (~2x peak activation). Always drop it after extracting tensors;
        # The returned tensors still hold the graph for a real backward path.
        try:
            policy_state = batch["policy_input_state"]
            dist = self.runtime.distribution_from_batch(policy_state)
            current_mean_norm = dist.distribution.mean
            actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            if actions.ndim == 3:
                actions = actions.reshape(actions.shape[0], -1)
            log_probs = dist.log_prob(actions)
            values = self.runtime.value_from_state(policy_state)
            if values.ndim == 0:
                values = values.unsqueeze(0)
            entropy = dist.entropy()
            aux_losses: Dict[str, torch.Tensor] = {}
            ref_mean_norm = batch.get("mean_norm", batch.get("action_logprob_info_mean_norm"))
            if self.mean_ref_l2_coef != 0.0 and ref_mean_norm is not None:
                ref_mean_norm = torch.as_tensor(ref_mean_norm, dtype=torch.float32, device=self.device)
                if ref_mean_norm.ndim == 1:
                    ref_mean_norm = ref_mean_norm.unsqueeze(0)
                if ref_mean_norm.ndim == 3:
                    ref_mean_norm = ref_mean_norm.reshape(ref_mean_norm.shape[0], -1)
                aux_losses["mean_ref_l2"] = torch.mean((current_mean_norm.float() - ref_mean_norm) ** 2)
            return {
                "log_probs": log_probs,
                "values": values,
                "entropy": entropy,
                "aux_losses": aux_losses,
                "aux_logs": {},
            }
        finally:
            self.runtime.action_dist.distribution = None

    def learner_spec(self) -> LearnerSpec:
        return self._learner_spec

    def trainable_state_dict(self) -> Dict[str, Any]:
        return self.runtime.trainable_state_dict()

    def load_trainable_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.runtime.load_trainable_state_dict(state_dict)

    def trainable_components(self) -> List[str]:
        return self.runtime.trainable_components()

    def set_train(self, mode: bool = True) -> None:
        self.runtime.set_train(mode)
