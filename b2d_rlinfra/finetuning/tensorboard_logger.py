"""TensorBoard logging for rl finetune coordinator metrics."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional

import numpy as np

logger = logging.getLogger("RLFinetune.TensorBoard")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _parse_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class RLFineTuneTensorboardLogger:
    """Coordinator-owned TensorBoard writer with rollout window summaries."""

    def __init__(
        self,
        *,
        log_dir: str | Path,
        enabled: bool = True,
        window_size: int = 100,
        writer: Optional[Any] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self.window_size = max(1, int(window_size))
        self._rollouts: Deque[Dict[str, Any]] = deque(maxlen=self.window_size)
        self._writer = writer
        self._cumulative_steps = 0.0

        if not self.enabled or self._writer is not None:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard not available; rl_finetune tensorboard logging is disabled")
            self.enabled = False
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    @classmethod
    def from_config(cls, *, run_dir: str | Path, rl_config: Mapping[str, Any]) -> "RLFineTuneTensorboardLogger":
        tb_cfg = dict((rl_config or {}).get("tensorboard", {}) or {})
        enabled = bool(tb_cfg.get("enable", True))
        log_dir = tb_cfg.get("log_dir") or (Path(run_dir) / "tensorboard")
        window_size = int(tb_cfg.get("window_size", 100))
        return cls(log_dir=log_dir, enabled=enabled, window_size=window_size)

    def add_rollout(self, meta: Mapping[str, Any]) -> None:
        if not meta:
            return
        self._rollouts.append(dict(meta))

    def log_update(
        self,
        *,
        update_id: int,
        stats: Mapping[str, Any],
        optimizer: Optional[Any],
        update_seconds: float,
    ) -> None:
        if not self.enabled or self._writer is None:
            return

        step = int(update_id)
        update_samples = _to_float(stats.get("samples"))
        trained_samples = _to_float(stats.get("samples_trained", update_samples))
        sharded_samples = _to_float(stats.get("samples_after_sharding", update_samples))
        self._cumulative_steps += sharded_samples
        train_values = {
            "train/policy_loss": _to_float(stats.get("policy_loss")),
            "train/value_loss": _to_float(stats.get("value_loss")),
            "train/entropy_loss": _to_float(stats.get("entropy_loss")),
            "train/approx_kl": _to_float(stats.get("approx_kl")),
            "train/clip_fraction": _to_float(stats.get("clip_fraction")),
            "train/grad_norm": _to_float(stats.get("grad_norm")),
            "train/ratio_mean": _to_float(stats.get("ratio_mean")),
            "train/advantage_mean": _to_float(stats.get("advantage_mean")),
            "train/return_mean": _to_float(stats.get("return_mean")),
            "train/value_mean": _to_float(stats.get("value_mean")),
            "train/explained_variance": _to_float(stats.get("explained_variance")),
            "train/first_batch_pre_step_approx_kl": _to_float(stats.get("first_batch_pre_step_approx_kl")),
            "train/stopped_by_target_kl": _to_float(stats.get("stopped_by_target_kl")),
            "train/skipped_by_target_kl": _to_float(stats.get("skipped_by_target_kl")),
            "train/early_stop_approx_kl": _to_float(stats.get("early_stop_approx_kl")),
            "train/early_stop_batch_index": _to_float(stats.get("early_stop_batch_index"), -1.0),
            "train/batches": _to_float(stats.get("batches")),
            "train/samples": update_samples,
            "train/samples_trained": trained_samples,
            "train/samples_after_sharding": sharded_samples,
            "train/dropped_samples": _to_float(stats.get("dropped_samples")),
            "train/samples_before_filter": _to_float(stats.get("samples_before_filter", update_samples)),
            "train/samples_after_filter": _to_float(stats.get("samples_after_filter", update_samples)),
            "train/samples_after_filter_success": _to_float(stats.get("samples_after_filter_success")),
            "train/samples_after_filter_failure": _to_float(stats.get("samples_after_filter_failure")),
            "train/samples_after_filter_truncated": _to_float(stats.get("samples_after_filter_truncated")),
            "train/advantage_mean_success": _to_float(stats.get("advantage_mean_success")),
            "train/advantage_mean_failure": _to_float(stats.get("advantage_mean_failure")),
            "train/advantage_mean_truncated": _to_float(stats.get("advantage_mean_truncated")),
            "train/learning_rate": self._learning_rate(optimizer),
        }
        optional_train_metrics = {
            "full_ref_kl_before": "train/full_ref_kl_before",
            "full_ref_kl_after": "train/full_ref_kl_after",
            "full_ref_kl_delta": "train/full_ref_kl_delta",
            "top1_action_flip_rate": "train/top1_action_flip_rate",
            "selected_action_prob_before": "train/selected_action_prob_before",
            "selected_action_prob_after": "train/selected_action_prob_after",
            "selected_action_prob_delta": "train/selected_action_prob_delta",
            "selected_action_prob_abs_delta": "train/selected_action_prob_abs_delta",
        }
        for stat_name, tag in optional_train_metrics.items():
            value = _optional_float(stats.get(stat_name))
            if value is not None:
                train_values[tag] = value
        # Adapter-specific auxiliary losses (e.g. aux_kl_loss, aux_mean_ref_l2).
        for key, value in stats.items():
            if str(key).startswith("aux_"):
                train_values[f"train/{key}"] = _to_float(value)
        overview_values = {
            "overview/update_duration_minutes": _to_float(update_seconds) / 60.0,
            "overview/cumulative_steps": self._cumulative_steps,
            "overview/samples_per_second": self._samples_per_second(list(self._rollouts)),
        }
        rollout_values = self._rollout_window_values()
        try:
            for tag, value in {**train_values, **overview_values, **rollout_values}.items():
                self._writer.add_scalar(tag, value, step)
            flush = getattr(self._writer, "flush", None)
            if callable(flush):
                flush()
        except Exception as exc:
            logger.warning("TensorBoard write failed; disabling rl_finetune tensorboard logging: %s", exc)
            self.enabled = False
            self.close()

    @staticmethod
    def _learning_rate(optimizer: Optional[Any]) -> float:
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return 0.0
        return _to_float(param_groups[0].get("lr", 0.0))

    def _rollout_window_values(self) -> Dict[str, float]:
        rollouts = list(self._rollouts)
        if not rollouts:
            return {
                "rollout/reward_mean": 0.0,
                "rollout/route_completion_mean": 0.0,
                "rollout/success_rate": 0.0,
                "rollout/episode_length_mean": 0.0,
            }

        rewards = [_to_float(meta.get("reward_sum")) for meta in rollouts]
        route_completion = [_to_float(meta.get("route_completion_ratio")) for meta in rollouts]
        lengths = [_to_float(meta.get("num_steps")) for meta in rollouts]
        return {
            "rollout/reward_mean": float(np.mean(rewards)),
            "rollout/route_completion_mean": float(np.mean(route_completion)),
            "rollout/success_rate": float(np.mean([value >= 0.999 for value in route_completion])),
            "rollout/episode_length_mean": float(np.mean(lengths)),
        }

    @staticmethod
    def _samples_per_second(rollouts: list[Dict[str, Any]]) -> float:
        timestamped = []
        for meta in rollouts:
            timestamp = _parse_timestamp(meta.get("created_at"))
            if timestamp is None:
                continue
            timestamped.append((timestamp, _to_float(meta.get("num_steps"))))
        if len(timestamped) < 2:
            return 0.0
        duration = max(ts for ts, _steps in timestamped) - min(ts for ts, _steps in timestamped)
        if duration <= 0.0:
            return 0.0
        return float(sum(steps for _ts, steps in timestamped) / duration)

    def close(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        close = getattr(writer, "close", None)
        if callable(close):
            close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
