"""Coordinator for rl finetune PPO updates."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from b2d_rlinfra.finetuning.collector import Collector, collector_entry, rl_ppo_config
from b2d_rlinfra.finetuning.config_schema import normalize_rl_finetune_config
from b2d_rlinfra.finetuning.coordination import (
    HeartbeatMonitor,
    QueueEventBus,
    atomic_torch_save,
    publish_coordinator_exit_state,
    read_control_state,
)
from b2d_rlinfra.finetuning.learner_distributed import (
    LearnerCommandBus,
    LearnerContext,
    wrap_training_module,
)
from b2d_rlinfra.finetuning.policy_adapter import LearnerOutput, LearnerSpec, resolve_policy_adapter
from b2d_rlinfra.finetuning.rollout_file_dataset import RolloutFileDataset
from b2d_rlinfra.finetuning.rollout_file_store import (
    RolloutManifest,
    select_ready_with_policy_lag,
    write_update_index,
)
from b2d_rlinfra.finetuning.tensorboard_logger import RLFineTuneTensorboardLogger
from b2d_rlinfra.finetuning.topology import TopologyPlan, build_topology
from b2d_rlinfra.finetuning.weight_store import WeightStore

logger = logging.getLogger("RLFinetune.Coordinator")
_RUN_ENV_KEY = "B2D_RL_FINETUNE_RUN_DIR"
_FP16_PRECISIONS = {"fp16", "float16"}


class _CoordinatorStopRequested(RuntimeError):
    pass


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _full_log_probs_from_output(learner_output: LearnerOutput) -> Optional[torch.Tensor]:
    aux_logs = dict(learner_output.get("aux_logs") or {})
    full_log_probs = aux_logs.get("full_log_probs")
    if full_log_probs is None:
        return None
    tensor = full_log_probs if torch.is_tensor(full_log_probs) else torch.as_tensor(full_log_probs)
    if tensor.ndim < 2:
        return None
    tensor = tensor.detach().float()
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor


def _ref_log_probs_from_batch(batch: Mapping[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    ref_log_probs = batch.get("ref_log_probs", batch.get("action_logprob_info_ref_log_probs"))
    if ref_log_probs is None:
        return None
    tensor = ref_log_probs if torch.is_tensor(ref_log_probs) else torch.as_tensor(ref_log_probs)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim < 2:
        return None
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor.detach().to(device=device, dtype=torch.float32)


def _selected_action_probs(log_probs: torch.Tensor, actions: torch.Tensor) -> Optional[torch.Tensor]:
    if log_probs.ndim != 2:
        return None
    flat_actions = actions.to(device=log_probs.device, dtype=torch.long).view(-1, 1)
    if flat_actions.shape[0] != log_probs.shape[0]:
        return None
    if torch.any(flat_actions < 0) or torch.any(flat_actions >= log_probs.shape[1]):
        return None
    return torch.exp(log_probs).gather(1, flat_actions).squeeze(1)


def _distribution_drift_metrics(
    *,
    before_log_probs: Optional[torch.Tensor],
    after_log_probs: Optional[torch.Tensor],
    ref_log_probs: Optional[torch.Tensor],
    actions: torch.Tensor,
) -> Dict[str, float]:
    if before_log_probs is None or after_log_probs is None:
        return {}
    if before_log_probs.shape != after_log_probs.shape:
        return {}

    metrics: Dict[str, float] = {}
    with torch.no_grad():
        before = before_log_probs.detach().float()
        after = after_log_probs.detach().to(device=before.device, dtype=torch.float32)
        if ref_log_probs is not None and ref_log_probs.shape == before.shape:
            ref = ref_log_probs.detach().to(device=before.device, dtype=torch.float32)
            kl_before = F.kl_div(before, ref, log_target=True, reduction="batchmean")
            kl_after = F.kl_div(after, ref, log_target=True, reduction="batchmean")
            metrics["full_ref_kl_before"] = float(kl_before.cpu().item())
            metrics["full_ref_kl_after"] = float(kl_after.cpu().item())
            metrics["full_ref_kl_delta"] = float((kl_after - kl_before).cpu().item())

        top1_before = torch.argmax(before, dim=1)
        top1_after = torch.argmax(after, dim=1)
        metrics["top1_action_flip_rate"] = float((top1_before != top1_after).float().mean().cpu().item())

        selected_before = _selected_action_probs(before, actions)
        selected_after = _selected_action_probs(after, actions)
        if selected_before is not None and selected_after is not None:
            selected_delta = selected_after - selected_before
            metrics["selected_action_prob_before"] = float(selected_before.mean().cpu().item())
            metrics["selected_action_prob_after"] = float(selected_after.mean().cpu().item())
            metrics["selected_action_prob_delta"] = float(selected_delta.mean().cpu().item())
            metrics["selected_action_prob_abs_delta"] = float(selected_delta.abs().mean().cpu().item())
    return metrics


def _format_rollout_progress(meta: Mapping[str, Any]) -> str:
    reward = meta.get("reward_sum")
    try:
        reward_text = f"{float(reward):.3f}"
    except (TypeError, ValueError):
        reward_text = "n/a"

    route_completion = meta.get("route_completion_ratio")
    try:
        rc_text = f"{float(route_completion) * 100.0:.1f}%"
    except (TypeError, ValueError):
        rc_text = "n/a"

    return f"reward={reward_text} rc={rc_text}"


def _validate_learner_output(result: Any) -> LearnerOutput:
    if not isinstance(result, Mapping):
        raise TypeError(
            "LearnerSpec.module must return a mapping, "
            f"got {type(result).__name__}"
        )

    allowed_keys = {"log_probs", "values", "entropy", "aux_losses", "aux_logs"}
    missing_keys = sorted({"log_probs", "values"} - set(result))
    if missing_keys:
        raise ValueError(
            "LearnerSpec.module output is missing required field(s): "
            + ", ".join(missing_keys)
        )
    unknown_keys = sorted(str(key) for key in set(result) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "LearnerSpec.module output contains unknown field(s): "
            + ", ".join(unknown_keys)
        )

    log_probs = result["log_probs"]
    values = result["values"]
    entropy = result.get("entropy")
    if not torch.is_tensor(log_probs):
        raise TypeError("LearnerSpec.module output 'log_probs' must be a torch.Tensor")
    if not torch.is_tensor(values):
        raise TypeError("LearnerSpec.module output 'values' must be a torch.Tensor")
    if entropy is not None and not torch.is_tensor(entropy):
        raise TypeError("LearnerSpec.module output 'entropy' must be a torch.Tensor or None")

    aux_losses = result.get("aux_losses")
    aux_logs = result.get("aux_logs")
    if aux_losses is None:
        aux_losses = {}
    if aux_logs is None:
        aux_logs = {}
    if not isinstance(aux_losses, Mapping):
        raise TypeError("LearnerSpec.module output 'aux_losses' must be a mapping")
    if not isinstance(aux_logs, Mapping):
        raise TypeError("LearnerSpec.module output 'aux_logs' must be a mapping")
    for name, value in aux_losses.items():
        if not isinstance(name, str):
            raise TypeError("LearnerSpec.module output 'aux_losses' keys must be strings")
        if not torch.is_tensor(value):
            raise TypeError(
                f"LearnerSpec.module output auxiliary loss {name!r} must be a torch.Tensor"
            )
    if any(not isinstance(name, str) for name in aux_logs):
        raise TypeError("LearnerSpec.module output 'aux_logs' keys must be strings")

    return {
        "log_probs": log_probs,
        "values": values,
        "entropy": entropy,
        "aux_losses": dict(aux_losses),
        "aux_logs": dict(aux_logs),
    }


class RLPPOUpdater:
    def __init__(
        self,
        *,
        policy: Any,
        algo_config: Any,
        device: torch.device,
        learner_spec: Optional[LearnerSpec] = None,
        training_module: Optional[torch.nn.Module] = None,
        learner_context: Optional[LearnerContext] = None,
    ):
        self.policy = policy
        self.algo = algo_config
        self.device = torch.device(device)
        self.learner_spec = learner_spec or self.policy.learner_spec()
        self.learner_spec.validate()
        self.training_module = training_module or self.learner_spec.module
        self.learner_context = learner_context or LearnerContext(device=self.device)
        precision = str(self.learner_spec.precision or "").lower()
        self.grad_scaler = (
            torch.amp.GradScaler("cuda")
            if self.device.type == "cuda" and precision in _FP16_PRECISIONS
            else None
        )

    def _global_float(self, value: torch.Tensor) -> float:
        return float(self.learner_context.mean_tensor(value.detach().float()).cpu().item())

    def _global_explained_variance(
        self,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> float:
        detached_returns = returns.detach().to(dtype=torch.float64)
        residuals = detached_returns - values.detach().to(dtype=torch.float64)
        moments = torch.stack(
            (
                torch.as_tensor(
                    detached_returns.numel(),
                    device=detached_returns.device,
                    dtype=torch.float64,
                ),
                detached_returns.sum(),
                torch.square(detached_returns).sum(),
                residuals.sum(),
                torch.square(residuals).sum(),
            )
        )
        count, returns_sum, returns_sum_sq, residual_sum, residual_sum_sq = (
            self.learner_context.sum_tensor(moments)
        )
        count_value = float(count.item())
        if count_value <= 0.0:
            return 0.0
        returns_var = torch.clamp(
            returns_sum_sq / count - torch.square(returns_sum / count),
            min=0.0,
        )
        if float(returns_var.item()) <= 1e-8:
            return 0.0
        residual_var = torch.clamp(
            residual_sum_sq / count - torch.square(residual_sum / count),
            min=0.0,
        )
        return float((1.0 - residual_var / (returns_var + 1e-8)).item())

    def _finish_ddp_reduction_without_step(self, *loss_terms: Any) -> None:
        """Complete a DDP forward when this batch will not run a real optimizer step.

        DistributedDataParallel requires a matching backward after each synced
        forward. A recoverable target-KL early-stop skips the real step, so a
        zeroed backward finishes the reducer before the next update.
        """
        if int(self.learner_context.world_size) <= 1:
            return
        if not isinstance(self.training_module, torch.nn.parallel.DistributedDataParallel):
            return
        dummy: Optional[torch.Tensor] = None
        for term in loss_terms:
            if not torch.is_tensor(term) or not term.requires_grad:
                continue
            contrib = term.reshape(-1).sum() * 0.0
            dummy = contrib if dummy is None else dummy + contrib
        if dummy is None:
            return
        optimizer = self.learner_spec.optimizer
        optimizer.zero_grad(set_to_none=True)
        dummy.backward()
        optimizer.zero_grad(set_to_none=True)

    def update(self, dataset: RolloutFileDataset, *, sampler_seed: int = 0) -> Dict[str, float]:
        self.training_module.train(True)
        self.policy.set_train(True)
        optimizer = self.learner_spec.optimizer
        params = [param for param in self.learner_spec.module.parameters() if param.requires_grad]
        batch_size = int(getattr(self.algo, "batch_size", 128))
        n_epochs = int(getattr(self.algo, "n_epochs", 1))
        clip_range = float(getattr(self.algo, "clip_range", 0.2))
        vf_coef = float(getattr(self.algo, "vf_coef", 0.5))
        ent_coef = float(getattr(self.algo, "ent_coef", 0.0))
        max_grad_norm = float(getattr(self.algo, "max_grad_norm", 0.5))
        normalize_advantage = bool(getattr(self.algo, "normalize_advantage", True))
        target_kl = getattr(self.algo, "target_kl", None)
        update_advantage_mean, update_advantage_std = dataset.advantage_statistics(
            self.learner_context
        )
        normalize_update_advantage = bool(normalize_advantage and len(dataset) > 1)

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_losses: List[float] = []
        aux_loss_logs: Dict[str, List[float]] = {}
        approx_kls: List[float] = []
        clip_fractions: List[float] = []
        grad_norms: List[float] = []
        ratio_means: List[float] = []
        advantage_means: List[float] = []
        return_means: List[float] = []
        value_means: List[float] = []
        explained_variances: List[float] = []
        drift_metric_logs: Dict[str, List[float]] = {}
        batches = 0
        trained_sample_visits = 0
        continue_training = True
        first_batch_pre_step_approx_kl: Optional[float] = None
        early_stop_approx_kl: Optional[float] = None
        early_stop_batch_index: Optional[int] = None
        skipped_by_target_kl = False

        for epoch in range(n_epochs):
            if not continue_training:
                break
            for batch in dataset.iter_rank_batches(
                batch_size,
                rank=self.learner_context.rank,
                world_size=self.learner_context.world_size,
                seed=int(sampler_seed),
                epoch=epoch,
                shuffle=True,
                device=self.device,
            ):
                learner_output = _validate_learner_output(self.training_module(batch))
                before_full_log_probs = _full_log_probs_from_output(learner_output)
                ref_full_log_probs = _ref_log_probs_from_batch(batch, self.device)
                local_aux_losses = {
                    str(name): aux_loss
                    for name, aux_loss in learner_output["aux_losses"].items()
                    if aux_loss is not None
                }
                aux_names = tuple(sorted(local_aux_losses))
                has_full_log_probs = before_full_log_probs is not None
                current_log_probs = learner_output["log_probs"]
                values = learner_output["values"]
                entropy = learner_output["entropy"]
                current_log_probs = current_log_probs.flatten()
                values = values.flatten()
                old_log_probs = batch["old_action_log_probs"].float().flatten()
                returns = batch["returns"].float().flatten()
                advantages = batch["advantages"].float().flatten()
                raw_advantages = advantages.detach().float()

                if normalize_update_advantage:
                    if update_advantage_std > 1e-8:
                        advantages = (advantages - update_advantage_mean) / (update_advantage_std + 1e-8)
                    else:
                        advantages = advantages - update_advantage_mean

                log_ratio = current_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                value_loss = F.mse_loss(values, returns)
                if entropy is None:
                    entropy_loss = -torch.mean(-current_log_probs)
                    entropy_value = torch.mean(-current_log_probs)
                else:
                    entropy_loss = -entropy.flatten().mean()
                    entropy_value = entropy.flatten().mean()

                with torch.no_grad():
                    detached_ratio = ratio.detach().float()
                    detached_log_ratio = log_ratio.detach().float()
                    detached_returns = returns.detach().float()
                    detached_values = values.detach().float()
                    approx_kl_tensor = torch.mean(
                        (torch.exp(detached_log_ratio) - 1.0) - detached_log_ratio
                    )
                    approx_kl = self._global_float(approx_kl_tensor)
                    clip_fraction = self._global_float(
                        torch.mean((torch.abs(detached_ratio - 1.0) > clip_range).float())
                    )
                    explained_variance = self._global_explained_variance(
                        detached_returns,
                        detached_values,
                    )

                if first_batch_pre_step_approx_kl is None:
                    first_batch_pre_step_approx_kl = float(approx_kl)
                if target_kl is not None and approx_kl > 1.5 * float(target_kl):
                    continue_training = False
                    early_stop_approx_kl = float(approx_kl)
                    early_stop_batch_index = int(batches)
                    skipped_by_target_kl = batches == 0
                    logger.info(
                        (
                            "Early stopping PPO update before optimizer step because "
                            "approx_kl=%.6f target_kl=%s updated_batches=%s"
                        ),
                        approx_kl,
                        target_kl,
                        batches,
                    )
                    # approx_kl is already all-reduced, so all ranks take this branch.
                    self._finish_ddp_reduction_without_step(
                        policy_loss,
                        value_loss,
                        entropy_loss,
                        *local_aux_losses.values(),
                    )
                    break

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                aux_weights = dict(self.learner_spec.aux_loss_weights)
                aux_log_tensors = []
                for name in aux_names:
                    aux_loss = local_aux_losses[name]
                    aux_tensor = aux_loss if torch.is_tensor(aux_loss) else torch.as_tensor(aux_loss, device=self.device)
                    aux_tensor = aux_tensor.mean()
                    weight = float(aux_weights.get(name, 1.0))
                    loss = loss + weight * aux_tensor
                    aux_log_tensors.append(aux_tensor.detach().float())
                if aux_log_tensors:
                    global_aux_logs = self.learner_context.mean_tensor(
                        torch.stack(aux_log_tensors)
                    ).cpu()
                    for name, value in zip(aux_names, global_aux_logs.tolist()):
                        aux_loss_logs.setdefault(name, []).append(float(value))

                if not self.learner_context.all_true(bool(torch.isfinite(loss).detach().item())):
                    raise FloatingPointError("non-finite PPO loss")

                optimizer.zero_grad(set_to_none=True)
                if self.grad_scaler is not None:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    if not self.learner_context.all_true(
                        bool(torch.isfinite(torch.as_tensor(grad_norm)).detach().item())
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("non-finite PPO gradient norm")
                    optimizer.step()

                if has_full_log_probs:
                    with torch.no_grad():
                        after_output = _validate_learner_output(self.learner_spec.module(batch))
                    after_full_log_probs = _full_log_probs_from_output(after_output)
                    drift_metrics = _distribution_drift_metrics(
                        before_log_probs=before_full_log_probs,
                        after_log_probs=after_full_log_probs,
                        ref_log_probs=ref_full_log_probs,
                        actions=batch["actions"],
                    )
                    drift_names = tuple(sorted(drift_metrics))
                    if drift_names:
                        local_drift = torch.tensor(
                            [drift_metrics[name] for name in drift_names],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        global_drift = self.learner_context.mean_tensor(local_drift).cpu()
                        for name, value in zip(drift_names, global_drift.tolist()):
                            drift_metric_logs.setdefault(name, []).append(float(value))

                policy_losses.append(self._global_float(policy_loss))
                value_losses.append(self._global_float(value_loss))
                entropy_losses.append(self._global_float(entropy_loss))
                approx_kls.append(float(approx_kl))
                clip_fractions.append(float(clip_fraction))
                grad_norms.append(float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm))
                ratio_means.append(self._global_float(detached_ratio.mean()))
                advantage_means.append(self._global_float(raw_advantages.mean()))
                return_means.append(self._global_float(detached_returns.mean()))
                value_means.append(self._global_float(detached_values.mean()))
                explained_variances.append(float(explained_variance))
                batches += 1
                trained_sample_visits += int(old_log_probs.numel()) * self.learner_context.world_size

        stats = {
            "batches": float(batches),
            "samples": float(len(dataset)),
            "samples_before_filter": float(getattr(dataset, "samples_before_filter", len(dataset))),
            "samples_after_filter": float(getattr(dataset, "samples_after_filter", len(dataset))),
            "samples_after_sharding": float(len(dataset) - dataset.distributed_dropped_samples),
            "samples_trained": float(trained_sample_visits),
            "dropped_samples": float(dataset.distributed_dropped_samples),
            "policy_loss": _mean_or_zero(policy_losses),
            "value_loss": _mean_or_zero(value_losses),
            "entropy_loss": _mean_or_zero(entropy_losses),
            "approx_kl": _mean_or_zero(approx_kls),
            "clip_fraction": _mean_or_zero(clip_fractions),
            "grad_norm": _mean_or_zero(grad_norms),
            "ratio_mean": _mean_or_zero(ratio_means),
            "advantage_mean": _mean_or_zero(advantage_means),
            "return_mean": _mean_or_zero(return_means),
            "value_mean": _mean_or_zero(value_means),
            "explained_variance": _mean_or_zero(explained_variances),
            "first_batch_pre_step_approx_kl": float(first_batch_pre_step_approx_kl or 0.0),
            "stopped_by_target_kl": float(early_stop_approx_kl is not None),
            "skipped_by_target_kl": float(skipped_by_target_kl),
            "early_stop_approx_kl": float(early_stop_approx_kl or 0.0),
            "early_stop_batch_index": float(early_stop_batch_index if early_stop_batch_index is not None else -1),
        }
        for name, values in sorted(aux_loss_logs.items()):
            stats[f"aux_{name}"] = _mean_or_zero(values)
        for name, values in sorted(drift_metric_logs.items()):
            stats[name] = _mean_or_zero(values)
        for outcome, count in dict(getattr(dataset, "filtered_sample_counts_by_outcome", {}) or {}).items():
            stats[f"samples_after_filter_{outcome}"] = float(count)
        for outcome, value in dict(getattr(dataset, "filtered_advantage_mean_by_outcome", {}) or {}).items():
            stats[f"advantage_mean_{outcome}"] = float(value)
        if self.grad_scaler is not None:
            stats["amp_loss_scale"] = float(self.grad_scaler.get_scale())
        return stats

    def state_dict(self) -> Dict[str, Any]:
        if self.grad_scaler is None:
            return {}
        return {"grad_scaler": self.grad_scaler.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.grad_scaler is not None and state.get("grad_scaler") is not None:
            self.grad_scaler.load_state_dict(state["grad_scaler"])


def load_policy_for_learner(
    raw_config: Mapping[str, Any],
    algo_config: Any,
    device: torch.device,
) -> Any:
    rl_config = dict(raw_config.get("rl_finetune", {}) or {})
    adapter_cfg = dict(raw_config.get("policy_adapter", {}) or {})
    adapter_type_cfg = dict(adapter_cfg.get("config", {}) or {})
    adapter_type_cfg.setdefault("learning_rate", float(getattr(algo_config, "learning_rate", 1.0e-4)))
    training_cfg = raw_config.get("training", {}) or {}
    adapter_type_cfg.setdefault("seed", rl_config.get("seed", training_cfg.get("seed", 0)))
    adapter_cls = resolve_policy_adapter(adapter_cfg)
    return adapter_cls.load_initial(adapter_type_cfg, adapter_cfg.get("checkpoint"), device)


def build_learner_updater(
    *,
    raw_config: Mapping[str, Any],
    algo_config: Any,
    device: torch.device,
    learner_context: LearnerContext,
) -> tuple[Any, RLPPOUpdater]:
    policy = load_policy_for_learner(raw_config, algo_config, device)
    learner_spec = policy.learner_spec()
    learner_spec.validate()
    training_module = wrap_training_module(learner_spec, learner_context)
    updater = RLPPOUpdater(
        policy=policy,
        algo_config=algo_config,
        device=device,
        learner_spec=learner_spec,
        training_module=training_module,
        learner_context=learner_context,
    )
    return policy, updater


def load_learner_checkpoint(
    checkpoint_path: str,
    *,
    policy: Any,
    updater: RLPPOUpdater,
    device: torch.device,
) -> tuple[int, int]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"--init-from-checkpoint must point to a checkpoint file, got {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid rl_finetune checkpoint payload: {path}")
    if "trainable_state_dict" not in payload:
        raise ValueError(f"Checkpoint does not contain trainable_state_dict: {path}")
    policy.load_trainable_state_dict(payload["trainable_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        updater.learner_spec.optimizer.load_state_dict(optimizer_state)
    updater.load_state_dict(dict(payload.get("updater_state_dict") or {}))
    policy_version = int(payload.get("policy_version", getattr(policy, "policy_version", 0)))
    policy.policy_version = policy_version
    if payload.get("updates_completed") is not None:
        update_id = int(payload["updates_completed"])
    elif payload.get("last_update_id") is not None:
        update_id = int(payload["last_update_id"]) + 1
    else:
        update_id = 0
    return policy_version, update_id


def run_learner_follower(
    *,
    raw_config: Mapping[str, Any],
    device: torch.device,
    learner_context: LearnerContext,
    command_bus: LearnerCommandBus,
    init_from_checkpoint: Optional[str] = None,
) -> None:
    algo_config = rl_ppo_config(raw_config)
    policy, updater = build_learner_updater(
        raw_config=raw_config,
        algo_config=algo_config,
        device=device,
        learner_context=learner_context,
    )
    if init_from_checkpoint:
        load_learner_checkpoint(
            init_from_checkpoint,
            policy=policy,
            updater=updater,
            device=device,
        )
    learner_context.barrier()
    rl_config = dict(raw_config.get("rl_finetune", {}) or {})
    sample_filter = dict(rl_config.get("update_sample_filter", {}) or {})
    last_seq = -1
    while True:
        command = command_bus.wait_next(last_seq=last_seq)
        command_name = str(command.get("command", "")).lower()
        if command_name == "stop":
            return
        if command_name != "update":
            raise ValueError(f"unknown learner command: {command_name!r}")
        with RolloutFileDataset(
            str(command["index_path"]), sample_filter=sample_filter
        ) as dataset:
            updater.update(dataset, sampler_seed=int(command["sampler_seed"]))
        # Rollout files must remain valid until every rank has released its mmap.
        learner_context.barrier()
        last_seq = int(command["seq"])


class Coordinator:
    def __init__(
        self,
        *,
        config: Any,
        config_path: str,
        run_dir: str,
        device: torch.device,
        topology: Optional[TopologyPlan] = None,
        event_bus: Optional[Any] = None,
        control_plane: Optional[Any] = None,
        spawn_collectors: bool = True,
        init_from_checkpoint: Optional[str] = None,
        learner_context: Optional[LearnerContext] = None,
        learner_command_bus: Optional[LearnerCommandBus] = None,
    ):
        self.config = config
        self.config_path = str(config_path)
        self.raw_config = normalize_rl_finetune_config(config.to_dict() if hasattr(config, "to_dict") else dict(config))
        if hasattr(config, "raw_config"):
            config.raw_config = self.raw_config
        if hasattr(config, "env_config"):
            config.env_config = self.raw_config.get("env")
        self.rl_config = dict(self.raw_config.get("rl_finetune", {}) or {})
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.topology = topology or build_topology(self.raw_config)
        self.distributed = bool(self.topology.distributed)
        self.event_bus = event_bus
        self.control_plane = control_plane
        self.spawn_collectors = bool(spawn_collectors)
        self.init_from_checkpoint = str(init_from_checkpoint) if init_from_checkpoint else None
        self.learner_context = learner_context or LearnerContext(device=self.device)
        self.learner_command_bus = learner_command_bus
        self.rollout_dir = Path(self.rl_config.get("rollout_dir") or (self.run_dir / "rollouts"))
        self.weight_dir = Path(self.rl_config.get("weight_dir") or (self.run_dir / "weights"))
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = RolloutManifest(str(self.rollout_dir / "manifest.jsonl"))
        policy_sync = dict(self.rl_config.get("policy_sync", {}) or {})
        self.weight_store = WeightStore(str(self.weight_dir), filename=policy_sync.get("filename", "policy_latest.pt"))
        self.algo_config = rl_ppo_config(self.raw_config)
        self.policy, self.updater = build_learner_updater(
            raw_config=self.raw_config,
            algo_config=self.algo_config,
            device=self.device,
            learner_context=self.learner_context,
        )
        self.policy_version = int(getattr(self.policy, "policy_version", 0))
        self.update_id = 0
        if self.init_from_checkpoint:
            self._load_rl_checkpoint(self.init_from_checkpoint)
        self.learner_context.barrier()
        self.tensorboard_logger = RLFineTuneTensorboardLogger.from_config(
            run_dir=self.run_dir,
            rl_config=self.rl_config,
        )
        self._collector_heartbeat_monitor: Optional[HeartbeatMonitor] = None
        self._heartbeat_grace_deadline = time.monotonic() + float(
            (self.rl_config.get("distributed", {}) or {}).get("heartbeat_timeout", 900.0)
        )
        if self.distributed:
            heartbeat_paths = [
                self.run_dir / "heartbeats" / f"collector_{plan.collector_id:03d}.json"
                for plan in self.topology.collectors()
            ]
            self._collector_heartbeat_monitor = HeartbeatMonitor(heartbeat_paths)

    def run(self, max_updates: Optional[int] = None) -> None:
        algo_name = str(getattr(self.algo_config, "name", "ppo")).lower()
        if algo_name != "ppo":
            raise ValueError(f"rl_finetune v1 only supports algorithm.name=ppo, got {algo_name!r}")
        max_updates = int(max_updates if max_updates is not None else self.rl_config.get("max_updates", 1))
        self._publish_latest()
        self._maybe_save_initial_checkpoint()

        completed = False
        stop_requested = False
        try:
            execution_mode = str(self.rl_config.get("execution_mode", "process")).lower()
            if not self.spawn_collectors:
                if self.event_bus is None:
                    raise ValueError("external collector mode requires an event_bus")
                self._run_external_collectors(max_updates)
            elif execution_mode == "inline":
                self._run_inline(max_updates)
            else:
                self._run_process_collectors(max_updates)
            completed = self.update_id >= max_updates
        except _CoordinatorStopRequested as exc:
            stop_requested = True
            logger.info("Coordinator stopping on control-plane request: %s", exc)
        finally:
            if self.learner_command_bus is not None:
                self.learner_command_bus.publish_stop(
                    seq=self.update_id + 1,
                    reason="completed" if completed else ("stop_requested" if stop_requested else "failed"),
                )
            if self.control_plane is not None:
                publish_coordinator_exit_state(
                    self.control_plane,
                    completed=completed,
                    stop_requested=stop_requested,
                )
            self.tensorboard_logger.close()

    def _run_inline(self, max_updates: int) -> None:
        collectors = [
            Collector(
                collector_id=plan.collector_id,
                config=self.config,
                rl_config=self.rl_config,
                rollout_dir=str(self.rollout_dir),
                weight_dir=str(self.weight_dir),
                device=torch.device(plan.device),
                collector_plan=plan.to_dict(),
                run_dir=str(self.run_dir),
            )
            for plan in self.topology.node(0).collectors
        ]
        try:
            while self.update_id < max_updates:
                self._collect_inline_until_ready(collectors)
                self._run_one_update()
        finally:
            for collector in collectors:
                collector.slot.close()

    def _collect_inline_until_ready(self, collectors: List[Collector]) -> None:
        while not self._ready_records():
            for collector in collectors:
                meta = collector.collect_one()
                self.manifest.add_ready(meta.to_dict())
                self.tensorboard_logger.add_rollout(meta.to_dict())
                logger.info(
                    "Collected rollout collector=%s episode=%s steps=%s %s",
                    meta.collector_id,
                    meta.episode_id,
                    meta.num_steps,
                    _format_rollout_progress(meta.to_dict()),
                )
                if self._ready_records():
                    break

    def _run_process_collectors(self, max_updates: int) -> None:
        ctx = mp.get_context(str(self.rl_config.get("start_method", "spawn")))
        event_queue = ctx.Queue(maxsize=int(self.rl_config.get("queue_maxsize", 256)))
        stop_event = ctx.Event()
        event_bus = QueueEventBus(event_queue)
        processes = []
        plans = list(self.topology.node(0).collectors)
        previous_run_env = os.environ.get(_RUN_ENV_KEY)
        os.environ[_RUN_ENV_KEY] = str(self.run_dir.resolve())
        try:
            for plan in plans:
                process = ctx.Process(
                    target=collector_entry,
                    args=(
                        plan.collector_id,
                        self.config,
                        self.rl_config,
                        str(self.rollout_dir),
                        str(self.weight_dir),
                        str(plan.device),
                        event_queue,
                        stop_event,
                        plan.to_dict(),
                        str(self.run_dir),
                        "queue",
                        None,
                    ),
                    daemon=False,
                )
                process.start()
                processes.append(process)
            while self.update_id < max_updates:
                self._wait_for_ready_events(event_bus, processes)
                self._run_one_update()
        finally:
            stop_event.set()
            stop_timeout = float(self.rl_config.get("shutdown_timeout", 120.0))
            for process in processes:
                process.join(timeout=stop_timeout)
                if process.is_alive():
                    logger.warning(
                        "Collector process pid=%s did not stop within %.1fs; terminating",
                        process.pid,
                        stop_timeout,
                    )
                    process.terminate()
                    process.join(timeout=5.0)
            self._cleanup_managed_carla_servers()
            self._cleanup_tagged_child_processes()
            if previous_run_env is None:
                os.environ.pop(_RUN_ENV_KEY, None)
            else:
                os.environ[_RUN_ENV_KEY] = previous_run_env

    def _run_external_collectors(self, max_updates: int) -> None:
        assert self.event_bus is not None
        if self.control_plane is not None:
            self.control_plane.publish_state("running", reason="learner_started")
            self.control_plane.heartbeat(phase="running")
        while self.update_id < max_updates:
            self._wait_for_ready_events(self.event_bus, processes=None)
            self._run_one_update()
            if self.control_plane is not None:
                self.control_plane.heartbeat(
                    phase="running",
                    extra={
                        "policy_version": self.policy_version,
                        "update_id": self.update_id,
                    },
                )

    def _cleanup_managed_carla_servers(self) -> None:
        sim_actor_cfg = dict(self.rl_config.get("sim_actor", {}) or {})
        if not bool(sim_actor_cfg.get("manage_servers", True)):
            return
        if str(self.rl_config.get("env_type", sim_actor_cfg.get("env_type", "carla"))).lower() == "fake":
            return

        env_config = dict(self.raw_config.get("env", {}) or {})
        carla_config = dict(env_config.get("carla", {}) or {})
        if not carla_config:
            return

        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "tools" / "runtime" / "kill_by_host.sh"
        if not script.exists():
            logger.warning("CARLA cleanup script not found: %s", script)
            return

        seen_hosts = []
        for node in self.topology.nodes:
            for plan in node.collectors:
                host = str(plan.host)
                if host not in seen_hosts:
                    seen_hosts.append(host)

        for host in seen_hosts:
            try:
                logger.info("Cleaning managed CARLA processes for host %s", host)
                subprocess.run(
                    ["bash", str(script), host],
                    cwd=str(project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=float(self.rl_config.get("carla_cleanup_timeout", 30.0)),
                    check=False,
                )
            except Exception as exc:
                logger.warning("Failed to cleanup CARLA processes for host %s: %s", host, exc)

    def _cleanup_tagged_child_processes(self) -> None:
        run_dir = str(self.run_dir.resolve())
        pids = self._find_processes_by_env(_RUN_ENV_KEY, run_dir)
        pids = [pid for pid in pids if pid != os.getpid()]
        if not pids:
            return

        logger.warning("Cleaning %s lingering rl_finetune child process(es): %s", len(pids), pids)
        for sig, wait_time in ((15, 1.0), (9, 0.0)):
            remaining = []
            for pid in pids:
                if not self._is_process_running(pid):
                    continue
                try:
                    os.kill(pid, sig)
                    remaining.append(pid)
                except ProcessLookupError:
                    continue
                except PermissionError as exc:
                    logger.warning("No permission to signal lingering process pid=%s: %s", pid, exc)
            if wait_time > 0.0:
                deadline = time.time() + wait_time
                while time.time() < deadline and any(self._is_process_running(pid) for pid in remaining):
                    time.sleep(0.05)
            pids = [pid for pid in remaining if self._is_process_running(pid)]
            if not pids:
                return
        if pids:
            logger.warning("Lingering rl_finetune child process(es) still alive after cleanup: %s", pids)

    @staticmethod
    def _find_processes_by_env(key: str, value: str) -> List[int]:
        marker = f"{key}={value}".encode()
        pids: List[int] = []
        proc_root = Path("/proc")
        for proc_dir in proc_root.iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                environ = (proc_dir / "environ").read_bytes()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if marker in environ.split(b"\0"):
                pids.append(int(proc_dir.name))
        return sorted(pids)

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _wait_for_ready_events(self, event_bus: Any, processes: Optional[List[mp.Process]] = None) -> None:
        poll_interval = float(self.rl_config.get("control_poll_interval", 1.0))
        timeout = self.rl_config.get("collect_timeout")
        deadline = None if timeout in (None, 0, "0", "") else time.time() + float(timeout)
        while not self._ready_records():
            self._raise_if_control_stop_requested()
            if deadline is not None and time.time() >= deadline:
                self._save_checkpoint_file(checkpoint_phase="collect_timeout", last_update_id=self.update_id - 1)
                if self.control_plane is not None:
                    self.control_plane.publish_state("stopping", reason="collect_timeout")
                raise TimeoutError("Timed out waiting for ready rl finetune rollouts")
            if self.control_plane is not None:
                self.control_plane.heartbeat(
                    phase="waiting_for_rollouts",
                    extra={"policy_version": self.policy_version, "update_id": self.update_id},
                )
            events = event_bus.poll(poll_interval)
            for event in events:
                self._handle_event(event)
            self._raise_if_control_stop_requested()
            self._raise_if_collectors_stopped(processes)

    def _raise_if_control_stop_requested(self) -> None:
        if self.control_plane is None:
            return
        state = read_control_state(self.control_plane)
        state_name = str(state.get("state", "")).lower()
        if state_name == "failed":
            reason = str(state.get("reason", "") or "unknown")
            raise RuntimeError(f"Distributed control plane entered failed state: {reason}")
        if not self.control_plane.should_stop():
            return
        raise _CoordinatorStopRequested(
            f"state={state.get('state', '')} reason={state.get('reason', '')}"
        )

    def _raise_if_collectors_stopped(self, processes: Optional[List[mp.Process]]) -> None:
        if self.distributed:
            active = self._active_collector_count()
            if active < int(self.topology.min_active_collectors):
                if self.control_plane is not None:
                    self.control_plane.publish_state(
                        "failed",
                        reason="min_active_collectors",
                        extra={
                            "active_collectors": active,
                            "min_active_collectors": int(self.topology.min_active_collectors),
                        },
                    )
                raise RuntimeError(
                    f"Active collectors dropped below threshold: active={active} "
                    f"min_active={self.topology.min_active_collectors}"
                )
            return
        if not processes:
            return
        stopped = [process for process in processes if not process.is_alive()]
        if not stopped:
            return
        exitcodes = [process.exitcode for process in stopped]
        raise RuntimeError(f"Collector process stopped before enough rollouts were ready; exitcodes={exitcodes}")

    def _active_collector_count(self) -> int:
        if self._collector_heartbeat_monitor is None:
            return int(self.topology.total_collectors)
        timeout = float((self.rl_config.get("distributed", {}) or {}).get("heartbeat_timeout", 900.0))
        alive = self._collector_heartbeat_monitor.alive_paths(timeout=timeout)
        if len(alive) < int(self.topology.min_active_collectors) and time.monotonic() < self._heartbeat_grace_deadline:
            return int(self.topology.total_collectors)
        return len(alive)

    def _handle_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        payload = dict(event.get("payload") or {})
        if event_type == "rollout":
            self.manifest.add_ready(payload)
            self.tensorboard_logger.add_rollout(payload)
            logger.info(
                "Rollout ready: %s steps=%s %s",
                payload.get("file_path"),
                payload.get("num_steps"),
                _format_rollout_progress(payload),
            )
        elif event_type == "crash":
            self.manifest.add_crash(payload)
            logger.warning(
                "Collector crash: collector=%s reason=%s crash_type=%s detail=%s",
                payload.get("collector_id"), payload.get("reason"),
                payload.get("crash_type", ""), payload.get("crash_detail", ""),
            )

    def _ready_records(self) -> List[Dict[str, Any]]:
        max_files = int(self.rl_config.get("rollouts_per_update", 1))
        stale_action = str(self.rl_config.get("stale_rollout_action", "drop")).lower()
        max_policy_lag = self.rl_config.get("max_policy_lag", 2)
        cleanup_stale = bool(self.rl_config.get("cleanup_stale_rollouts", True))
        selected, stale_count, fresh_count = select_ready_with_policy_lag(
            self.manifest,
            max_files=max_files,
            policy_version=self.policy_version,
            update_id=self.update_id,
            max_policy_lag=max_policy_lag,
            stale_action=stale_action,
            cleanup_stale=cleanup_stale,
        )
        if stale_count:
            logger.warning(
                "Observed stale rollout(s): stale=%s fresh=%s policy_version=%s max_policy_lag=%s action=%s",
                stale_count,
                fresh_count,
                self.policy_version,
                max_policy_lag,
                stale_action,
            )
        return selected

    def _run_one_update(self) -> None:
        selected = self._ready_records()
        if not selected:
            raise RuntimeError("no ready rollout files available for update")
        rollout_files = [record["file_path"] for record in selected]
        self.manifest.mark_selected(rollout_files, self.update_id)
        index_path = self.rollout_dir / f"update_{self.update_id:06d}_sample_plan.json"
        try:
            update_start = time.perf_counter()
            sampler_seed = int(self.rl_config.get("seed", 0)) + self.update_id * 1009
            write_update_index(
                str(index_path),
                self.update_id,
                rollout_files,
                sample_filter=dict(self.rl_config.get("update_sample_filter", {}) or {}),
            )
            if self.learner_command_bus is not None:
                self.learner_command_bus.publish_update(
                    update_id=self.update_id,
                    index_path=str(index_path),
                    sampler_seed=sampler_seed,
                )
            with RolloutFileDataset(str(index_path)) as dataset:
                stats = self.updater.update(dataset, sampler_seed=sampler_seed)
            # All learner ranks close mmap views before rollout cleanup.
            self.learner_context.barrier()
            update_seconds = time.perf_counter() - update_start
            self.manifest.mark_consumed(rollout_files, self.update_id)
            skipped_by_target_kl = bool(stats.get("skipped_by_target_kl", 0.0))
            if skipped_by_target_kl:
                self._cleanup_consumed_rollouts()
                self.tensorboard_logger.log_update(
                    update_id=self.update_id,
                    stats=stats,
                    optimizer=self.updater.learner_spec.optimizer,
                    update_seconds=update_seconds,
                )
                logger.warning(
                    (
                        "Update %06d skipped by target_kl before any optimizer step; "
                        "consumed rollout samples=%s/%s dropped=%s first_batch_pre_step_kl=%.6f "
                        "early_stop_kl=%.6f policy_version=%s sample_split=s/f/t:%s/%s/%s "
                        "adv_split=s/f/t:%.4f/%.4f/%.4f"
                    ),
                    self.update_id,
                    int(stats.get("samples_after_filter", stats.get("samples", 0))),
                    int(stats.get("samples_before_filter", stats.get("samples", 0))),
                    int(stats.get("dropped_samples", 0)),
                    stats.get("first_batch_pre_step_approx_kl", 0.0),
                    stats.get("early_stop_approx_kl", 0.0),
                    self.policy_version,
                    int(stats.get("samples_after_filter_success", 0)),
                    int(stats.get("samples_after_filter_failure", 0)),
                    int(stats.get("samples_after_filter_truncated", 0)),
                    stats.get("advantage_mean_success", 0.0),
                    stats.get("advantage_mean_failure", 0.0),
                    stats.get("advantage_mean_truncated", 0.0),
                )
                self.update_id += 1
                return
            self.policy_version += 1
            self.policy.policy_version = self.policy_version
            self._publish_latest()
            self._maybe_save_checkpoint(last_update_id=self.update_id)
            self._cleanup_consumed_rollouts()
            self.tensorboard_logger.log_update(
                update_id=self.update_id,
                stats=stats,
                optimizer=self.updater.learner_spec.optimizer,
                update_seconds=update_seconds,
            )
            logger.info(
                (
                    "Update %06d complete samples=%s/%s dropped=%s batches=%s policy_loss=%.6f "
                    "value_loss=%.6f approx_kl=%.6f clip_fraction=%.4f "
                    "ratio_mean=%.4f advantage_mean=%.4f return_mean=%.4f "
                    "value_mean=%.4f explained_var=%.4f full_ref_kl_before=%.6f "
                    "full_ref_kl_after=%.6f top1_flip=%.4f selected_prob_delta=%.6f "
                    "first_pre_kl=%.6f stopped_by_target_kl=%s early_stop_kl=%.6f "
                    "sample_split=s/f/t:%s/%s/%s adv_split=s/f/t:%.4f/%.4f/%.4f"
                ),
                self.update_id,
                int(stats.get("samples_after_filter", stats.get("samples", 0))),
                int(stats.get("samples_before_filter", stats.get("samples", 0))),
                int(stats.get("dropped_samples", 0)),
                int(stats.get("batches", 0)),
                stats.get("policy_loss", 0.0),
                stats.get("value_loss", 0.0),
                stats.get("approx_kl", 0.0),
                stats.get("clip_fraction", 0.0),
                stats.get("ratio_mean", 0.0),
                stats.get("advantage_mean", 0.0),
                stats.get("return_mean", 0.0),
                stats.get("value_mean", 0.0),
                stats.get("explained_variance", 0.0),
                stats.get("full_ref_kl_before", 0.0),
                stats.get("full_ref_kl_after", 0.0),
                stats.get("top1_action_flip_rate", 0.0),
                stats.get("selected_action_prob_delta", 0.0),
                stats.get("first_batch_pre_step_approx_kl", 0.0),
                int(stats.get("stopped_by_target_kl", 0.0)),
                stats.get("early_stop_approx_kl", 0.0),
                int(stats.get("samples_after_filter_success", 0)),
                int(stats.get("samples_after_filter_failure", 0)),
                int(stats.get("samples_after_filter_truncated", 0)),
                stats.get("advantage_mean_success", 0.0),
                stats.get("advantage_mean_failure", 0.0),
                stats.get("advantage_mean_truncated", 0.0),
            )
            self.update_id += 1
        except Exception:
            self.manifest.rollback_selected(rollout_files, self.update_id)
            raise

    def _cleanup_consumed_rollouts(self) -> None:
        if not bool(self.rl_config.get("cleanup_consumed_rollouts", False)):
            return
        keep_last = int(self.rl_config.get("keep_last_n_update_rollouts", 0))
        try:
            deleted = self.manifest.cleanup_consumed_files(keep_last_n_update_rollouts=keep_last)
        except Exception as exc:
            logger.warning("Failed to cleanup consumed rollout files: %s", exc)
            return
        if deleted:
            logger.info("Deleted %s consumed rollout file(s)", len(deleted))

    def _publish_latest(self) -> None:
        params = [
            param
            for param in self.updater.learner_spec.module.parameters()
            if param.requires_grad
        ]
        dtype = str(params[0].dtype) if params else "unknown"
        adapter_cfg = dict(self.raw_config.get("policy_adapter", {}) or {})
        self.weight_store.publish(
            policy_version=self.policy_version,
            base_checkpoint=adapter_cfg.get("checkpoint"),
            trainable_state_dict=self.policy.trainable_state_dict(),
            trainable_components=self.policy.trainable_components(),
            dtype=dtype,
        )
        logger.info("Published policy_version=%s to %s", self.policy_version, self.weight_store.latest_path)

    def _checkpoint_interval(self) -> int:
        return int(self.rl_config.get("checkpoint_interval", 1))

    def _checkpoint_payload(self, *, checkpoint_phase: str, last_update_id: Optional[int]) -> Dict[str, Any]:
        updates_completed = 0 if last_update_id is None else int(last_update_id) + 1
        return {
            "policy_version": self.policy_version,
            "update_id": last_update_id,
            "last_update_id": last_update_id,
            "updates_completed": updates_completed,
            "checkpoint_phase": checkpoint_phase,
            "trainable_state_dict": self.policy.trainable_state_dict(),
            "trainable_components": self.policy.trainable_components(),
            "optimizer_state_dict": self.updater.learner_spec.optimizer.state_dict(),
            "updater_state_dict": self.updater.state_dict(),
            "config_path": self.config_path,
        }

    def _maybe_save_initial_checkpoint(self) -> None:
        interval = self._checkpoint_interval()
        if interval <= 0 or not bool(self.rl_config.get("save_initial_checkpoint", True)):
            return
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint_file(checkpoint_phase="initial", last_update_id=None, filename="rl_finetune_initial.pt")

    def _maybe_save_checkpoint(self, *, last_update_id: int) -> None:
        interval = self._checkpoint_interval()
        updates_completed = int(last_update_id) + 1
        if interval <= 0 or updates_completed % interval != 0:
            return
        self._save_checkpoint_file(
            checkpoint_phase="post_update",
            last_update_id=last_update_id,
            filename=f"rl_finetune_update_{updates_completed:06d}.pt",
        )

    def _save_checkpoint_file(
        self,
        *,
        checkpoint_phase: str,
        last_update_id: Optional[int],
        filename: Optional[str] = None,
    ) -> Path:
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            updates_completed = max(0, int(last_update_id if last_update_id is not None else -1) + 1)
            filename = f"rl_finetune_{checkpoint_phase}_{updates_completed:06d}.pt"
        payload = self._checkpoint_payload(checkpoint_phase=checkpoint_phase, last_update_id=last_update_id)
        path = checkpoint_dir / filename
        atomic_torch_save(path, payload)
        logger.info("Saved rl_finetune checkpoint: %s", path)
        return path

    def _load_rl_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        self.policy_version, self.update_id = load_learner_checkpoint(
            checkpoint_path,
            policy=self.policy,
            updater=self.updater,
            device=self.device,
        )
        logger.info(
            "Initialized rl_finetune from checkpoint=%s policy_version=%s next_update_id=%s",
            path,
            self.policy_version,
            self.update_id,
        )
