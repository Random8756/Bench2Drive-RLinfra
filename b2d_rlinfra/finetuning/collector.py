"""RL finetune collector: local model + one restartable env slot."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from b2d_rlinfra.finetuning.carla_slot import SlotCrashed, make_slot
from b2d_rlinfra.finetuning.config_schema import bind_environment_result_dir
from b2d_rlinfra.finetuning.coordination import FileControlPlane, FileEventBus, HeartbeatWriter
from b2d_rlinfra.finetuning.messages import CollectorEvent, CrashMeta, RolloutMeta
from b2d_rlinfra.finetuning.policy_adapter import (
    resolve_policy_adapter,
    stack_policy_states,
    to_numpy_tree,
)
from b2d_rlinfra.finetuning.rollout_file_store import (
    RolloutFileStore,
    compute_returns_and_advantages,
)
from b2d_rlinfra.finetuning.weight_store import WeightStore

logger = logging.getLogger("RLFinetune.Collector")


def rl_ppo_config(raw_config: Mapping[str, Any]) -> SimpleNamespace:
    """Build the PPO algorithm namespace from a raw rl_finetune config dict."""
    algo_cfg = raw_config.get("algorithm")
    if not isinstance(algo_cfg, Mapping) or not algo_cfg:
        raise ValueError("rl_finetune config must define a top-level algorithm section")
    return SimpleNamespace(**dict(algo_cfg))


class CollectorStopped(Exception):
    """Raised internally when the coordinator requests collector shutdown."""


def _stack_optional_infos(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    key_sets = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("action_logprob_info entries must be dicts")
        key_sets.append(set(item.keys()))
    keys = key_sets[0]
    if any(item_keys != keys for item_keys in key_sets[1:]):
        raise ValueError("optional action_logprob_info keys are inconsistent across the episode")
    stacked: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [item[key] for item in items]
        arrays = [np.asarray(value) for value in values]
        try:
            stacked[key] = np.stack(arrays, axis=0)
        except ValueError as exc:
            raise ValueError(f"optional action_logprob_info field {key!r} cannot be stacked") from exc
    return stacked


def _stack_rollout_items(items: List[Any]) -> Any:
    if not items:
        raise ValueError("cannot stack an empty rollout item list")
    first = items[0]
    if isinstance(first, Mapping):
        keys = set(first.keys())
        for item in items[1:]:
            if not isinstance(item, Mapping) or set(item.keys()) != keys:
                raise ValueError("mapping rollout item keys are inconsistent across the episode")
        return {
            key: _stack_rollout_items([item[key] for item in items])
            for key in sorted(keys)
        }
    return np.stack([np.asarray(item) for item in items], axis=0)


class EpisodeBuffer:
    def __init__(self) -> None:
        self.policy_input_states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.episode_starts: List[bool] = []
        self.values: List[float] = []
        self.old_action_log_probs: List[float] = []
        self.action_logprob_infos: List[Dict[str, Any]] = []

    def append(
        self,
        *,
        policy_input_state: Any,
        action: Any,
        reward: float,
        episode_start: bool,
        value: float,
        old_action_log_prob: float,
        action_logprob_info: Optional[Dict[str, Any]],
    ) -> None:
        self.policy_input_states.append(to_numpy_tree(policy_input_state))
        self.actions.append(to_numpy_tree(action))
        self.rewards.append(float(reward))
        self.episode_starts.append(bool(episode_start))
        self.values.append(float(value))
        self.old_action_log_probs.append(float(old_action_log_prob))
        self.action_logprob_infos.append(dict(action_logprob_info or {}))

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)

    def to_rollout_kwargs(self) -> Dict[str, Any]:
        optional = _stack_optional_infos(self.action_logprob_infos)
        if optional:
            prefixed = {}
            for key, value in optional.items():
                if key == "ref_log_probs":
                    prefixed[key] = value
                else:
                    prefixed[f"action_logprob_info_{key}"] = value
            optional = prefixed
        return {
            "actions": _stack_rollout_items(self.actions),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "episode_starts": np.asarray(self.episode_starts, dtype=np.bool_),
            "values": np.asarray(self.values, dtype=np.float32),
            "old_action_log_probs": np.asarray(self.old_action_log_probs, dtype=np.float32),
            "policy_input_state": stack_policy_states(self.policy_input_states),
            "optional_fields": optional,
        }


class Collector:
    def __init__(
        self,
        *,
        collector_id: int,
        config: Any,
        rl_config: Mapping[str, Any],
        rollout_dir: str,
        weight_dir: str,
        device: torch.device,
        event_queue: Optional[Any] = None,
        stop_event: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        control_plane: Optional[Any] = None,
        collector_plan: Optional[Mapping[str, Any]] = None,
        run_dir: Optional[str] = None,
        heartbeat_dir: Optional[str] = None,
    ):
        self.collector_id = int(collector_id)
        self.config = config
        self.raw_config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        self.rl_config = dict(rl_config or {})
        self.event_queue = event_queue
        self.event_bus = event_bus
        self.stop_event = stop_event
        self.control_plane = control_plane
        self.collector_plan = dict(collector_plan or {})
        self.run_dir = str(run_dir) if run_dir is not None else None
        self.device = torch.device(device)
        self.rollout_store = RolloutFileStore(rollout_dir)
        weight_filename = (
            self.rl_config.get("policy_sync", {}) or {}
        ).get("filename", "policy_latest.pt")
        self.weight_store = WeightStore(weight_dir, filename=weight_filename)
        self.episode_id = 0
        self.algo_config = rl_ppo_config(self.raw_config)

        adapter_cfg = dict(self.raw_config.get("policy_adapter", {}) or {})
        adapter_type_cfg = dict(adapter_cfg.get("config", {}) or {})
        adapter_type_cfg.setdefault("learning_rate", float(getattr(self.algo_config, "learning_rate", 1.0e-4)))
        training_cfg = self.raw_config.get("training", {}) or {}
        adapter_type_cfg.setdefault(
            "seed",
            int(self.rl_config.get("seed", training_cfg.get("seed", 0))) + self.collector_id,
        )
        adapter_cls = resolve_policy_adapter(adapter_cfg)
        self.policy = adapter_cls.load_initial(
            adapter_type_cfg,
            adapter_cfg.get("checkpoint"),
            self.device,
        )
        self.policy_version = int(getattr(self.policy, "policy_version", 0))
        self._synced_once = False
        self._current_obs: Any = None
        self._current_info: Optional[Dict[str, Any]] = None
        self._heartbeat_phase = "starting"
        self._heartbeat_steps = 0
        self._heartbeat: Optional[HeartbeatWriter] = None
        if heartbeat_dir:
            from pathlib import Path

            interval = float((self.rl_config.get("distributed", {}) or {}).get("heartbeat_interval", 10.0))
            self._heartbeat = HeartbeatWriter(
                Path(heartbeat_dir) / f"collector_{self.collector_id:03d}.json",
                interval=interval,
                payload_factory=self._heartbeat_payload,
            )

        env_config = dict(getattr(config, "env_config", None) or self.raw_config.get("env") or {})
        if self.run_dir is not None:
            env_config = bind_environment_result_dir(env_config, self.run_dir)
        self.slot = make_slot(
            collector_id=self.collector_id,
            env_config=env_config,
            algorithm_config=dict(self.raw_config["algorithm"]),
            rl_config=self.rl_config,
            collector_plan=self.collector_plan or None,
        )
        self.gamma = float(getattr(self.algo_config, "gamma", 0.99))
        self.gae_lambda = float(getattr(self.algo_config, "gae_lambda", 0.95))

    def run(self, max_episodes: Optional[int] = None) -> None:
        if self._heartbeat is not None:
            self._heartbeat.start()
        try:
            self._heartbeat_phase = "reset"
            obs, info = self._ensure_current_obs()
            completed = 0
            self._heartbeat_phase = "collecting"
            while not self._should_stop():
                try:
                    _meta, obs, info = self.collect_episode_from(obs, info)
                except CollectorStopped:
                    break
                completed += 1
                self._current_obs = obs
                self._current_info = info
                self._heartbeat_phase = "collecting"
                if max_episodes is not None and completed >= int(max_episodes):
                    break
        finally:
            self._heartbeat_phase = "closing"
            if self._heartbeat is not None:
                self._heartbeat.write_once()
            self.slot.close()
            if self._heartbeat is not None:
                self._heartbeat.stop()

    def collect_one(self) -> RolloutMeta:
        obs, info = self._ensure_current_obs()
        meta, self._current_obs, self._current_info = self.collect_episode_from(obs, info)
        return meta

    def collect_episode_from(self, obs: Any, info: Dict[str, Any]) -> tuple[RolloutMeta, Any, Dict[str, Any]]:
        """Collect one episode; returns (meta, next_obs, next_info) for the following episode."""
        while not self._should_stop():
            self._maybe_load_latest()
            episode_policy_version = int(self.policy_version)
            buffer = EpisodeBuffer()
            episode_start_info = dict(info or {})
            self.policy.on_episode_start(episode_start_info)
            self._heartbeat_phase = "episode"

            while not self._should_stop():
                policy_step = self.policy.collect_step(obs)
                try:
                    next_obs, reward, terminated, truncated, step_info = self.slot.step(policy_step.env_action)
                except SlotCrashed as exc:
                    buffer.clear()
                    crash_info = dict(getattr(exc, "info", {}) or {})
                    crash_reason = crash_info.get("crash_reason") or exc.reason
                    self._record_crash_and_advance(
                        {
                            **episode_start_info,
                            **crash_info,
                            "crash_reason": crash_reason,
                            "crash_type": crash_info.get("crash_type", exc.__class__.__name__),
                            "crash_detail": crash_info.get("crash_detail", str(exc)),
                        },
                        episode_policy_version,
                    )
                    obs, info = self._restart_after_carla_failure(
                        crash_reason,
                        episode_policy_version,
                    )
                    self._maybe_load_latest()
                    break
                except TimeoutError as exc:
                    buffer.clear()
                    crash_info = {
                        **episode_start_info,
                        "crash_reason": "slot_step_timeout",
                        "crash_type": exc.__class__.__name__,
                        "crash_detail": str(exc),
                    }
                    self._record_crash_and_advance(crash_info, episode_policy_version)
                    obs, info = self._restart_after_carla_failure(
                        "slot_step_timeout",
                        episode_policy_version,
                    )
                    self._maybe_load_latest()
                    break
                step_info = dict(step_info or {})

                if step_info.get("crashed", False):
                    buffer.clear()
                    self._record_crash_and_advance(
                        {**episode_start_info, **step_info},
                        episode_policy_version,
                    )
                    obs, info = self._restart_after_carla_failure(
                        step_info.get("crash_reason") or step_info.get("crash_type") or "crashed",
                        episode_policy_version,
                    )
                    self._maybe_load_latest()
                    break

                buffer.append(
                    policy_input_state=policy_step.policy_input_state,
                    action=policy_step.train_action,
                    reward=float(reward),
                    episode_start=bool(episode_start_info.get("episode_start", len(buffer) == 0)),
                    value=float(policy_step.value),
                    old_action_log_prob=float(policy_step.old_action_log_prob),
                    action_logprob_info=policy_step.action_logprob_info,
                )
                self._heartbeat_steps += 1

                if terminated or truncated:
                    bootstrap_non_terminal = bool(truncated and not terminated)
                    bootstrap_value = (
                        self.policy.value_from_obs(next_obs)
                        if bootstrap_non_terminal
                        else 0.0
                    )
                    returns, advantages = compute_returns_and_advantages(
                        buffer.rewards,
                        buffer.values,
                        gamma=self.gamma,
                        gae_lambda=self.gae_lambda,
                        bootstrap_value=bootstrap_value,
                        bootstrap_non_terminal=bootstrap_non_terminal,
                    )
                    rollout_kwargs = buffer.to_rollout_kwargs()
                    meta = self.rollout_store.write_episode(
                        collector_id=self.collector_id,
                        episode_id=self.episode_id,
                        policy_version=episode_policy_version,
                        returns=returns,
                        advantages=advantages,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=step_info,
                        **rollout_kwargs,
                    )
                    self._emit_rollout(meta)
                    self.episode_id += 1

                    if step_info.get("planned_recycle", False):
                        next_reset_obs, next_reset_info = self._restart_after_carla_failure(
                            step_info.get("planned_recycle_reason", "planned_recycle"),
                            episode_policy_version,
                        )
                    else:
                        try:
                            next_reset_obs, next_reset_info = self.slot.consume_reset_obs()
                        except SlotCrashed as exc:
                            crash_info = dict(getattr(exc, "info", {}) or {})
                            self._record_crash_and_advance(
                                {
                                    **step_info,
                                    **crash_info,
                                    "crash_reason": crash_info.get("crash_reason", "reset_obs_crashed"),
                                    "crash_type": crash_info.get("crash_type", exc.__class__.__name__),
                                    "crash_detail": str(exc),
                                },
                                episode_policy_version,
                            )
                            next_reset_obs, next_reset_info = self._restart_after_carla_failure(
                                exc.reason,
                                episode_policy_version,
                            )
                        except TimeoutError as exc:
                            self._record_crash_and_advance(
                                {
                                    **step_info,
                                    "crash_reason": "reset_obs_timeout",
                                    "crash_type": exc.__class__.__name__,
                                    "crash_detail": str(exc),
                                },
                                episode_policy_version,
                            )
                            next_reset_obs, next_reset_info = self._restart_after_carla_failure(
                                "reset_obs_timeout",
                                episode_policy_version,
                            )
                    self._maybe_load_latest()
                    return meta, next_reset_obs, next_reset_info

                obs = next_obs
                episode_start_info = {}

        raise CollectorStopped("collector stopped before completing an episode")

    def _emit_rollout(self, meta: RolloutMeta) -> None:
        event = CollectorEvent.rollout(meta).to_dict()
        if self.event_bus is not None:
            self.event_bus.emit(event)
        elif self.event_queue is not None:
            self.event_queue.put(event)

    def _emit_crash(self, info: Mapping[str, Any], policy_version: int) -> None:
        crash = CrashMeta(
            collector_id=self.collector_id,
            episode_id=self.episode_id,
            policy_version=int(policy_version),
            reason=str(info.get("crash_reason") or info.get("reason") or "crashed"),
            crash_type=str(info.get("crash_type", "")),
            crash_detail=str(info.get("crash_detail") or info.get("error") or ""),
            route_id=str(info.get("route_id", "")),
            scenario_name=str(info.get("scenario_name", "")),
            town=str(info.get("town", "")),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        event = CollectorEvent.crash(crash).to_dict()
        if self.event_bus is not None:
            self.event_bus.emit(event)
        elif self.event_queue is not None:
            self.event_queue.put(event)

    def _record_crash_and_advance(self, info: Mapping[str, Any], policy_version: int) -> None:
        self._emit_crash(info, policy_version)
        self.episode_id += 1

    def _restart_after_carla_failure(self, reason: str, policy_version: int) -> tuple[Any, Dict[str, Any]]:
        backoff = float(self.rl_config.get("restart_backoff", 5.0))
        max_backoff = float(self.rl_config.get("restart_max_backoff", 60.0))
        attempt = 0
        while not self._should_stop():
            try:
                return self.slot.restart(reason=reason)
            except SlotCrashed as exc:
                crash_info = dict(getattr(exc, "info", {}) or {})
                self._record_crash_and_advance(
                    {
                        **crash_info,
                        "crash_reason": crash_info.get("crash_reason", reason),
                        "crash_type": crash_info.get("crash_type", exc.__class__.__name__),
                        "crash_detail": str(exc),
                    },
                    policy_version,
                )
                reason = exc.reason
            except TimeoutError as exc:
                self._record_crash_and_advance(
                    {
                        "crash_reason": reason,
                        "crash_type": exc.__class__.__name__,
                        "crash_detail": str(exc),
                    },
                    policy_version,
                )
            attempt += 1
            delay = min(backoff * attempt, max_backoff)
            logger.warning(
                "Collector %s restart attempt %s failed (%s), retrying in %.1fs",
                self.collector_id, attempt, reason, delay,
            )
            time.sleep(delay)
        raise CollectorStopped("collector stopped while restarting after CARLA failure")

    def _maybe_load_latest(self) -> None:
        warned = False
        while True:
            version, changed = self.weight_store.maybe_load_into(
                self.policy,
                self.policy_version,
                map_location=self.device,
                force=not self._synced_once,
            )
            if changed:
                logger.info("Collector %s loaded policy_version=%s", self.collector_id, version)
                self.policy_version = int(version)
                self._synced_once = True
                return
            if self._synced_once:
                self.policy_version = int(version)
                return
            if self._should_stop():
                raise CollectorStopped("collector stopped before initial policy weights were published")
            if not warned:
                logger.info(
                    "Collector %s waiting for initial policy weights at %s",
                    self.collector_id,
                    self.weight_store.latest_path,
                )
                warned = True
            time.sleep(float(self.rl_config.get("control_poll_interval", 1.0)))

    def _should_stop(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return True
        if self.control_plane is not None and self.control_plane.should_stop():
            return True
        return False

    def _heartbeat_payload(self, seq: int) -> Dict[str, Any]:
        return {
            "seq": int(seq),
            "collector_id": self.collector_id,
            "episode_id": int(self.episode_id),
            "policy_version": int(self.policy_version),
            "phase": self._heartbeat_phase,
            "steps": int(self._heartbeat_steps),
            "pid": mp.current_process().pid,
        }

    def _ensure_current_obs(self) -> tuple[Any, Dict[str, Any]]:
        if self._current_obs is None or self._current_info is None:
            try:
                self._current_obs, self._current_info = self.slot.reset()
            except SlotCrashed as exc:
                crash_info = dict(getattr(exc, "info", {}) or {})
                self._record_crash_and_advance(
                    {
                        **crash_info,
                        "crash_reason": crash_info.get("crash_reason", "initial_reset_crashed"),
                        "crash_type": crash_info.get("crash_type", exc.__class__.__name__),
                        "crash_detail": str(exc),
                    },
                    self.policy_version,
                )
                self._current_obs, self._current_info = self._restart_after_carla_failure(
                    exc.reason,
                    self.policy_version,
                )
            except TimeoutError as exc:
                self._record_crash_and_advance(
                    {
                        "crash_reason": "initial_reset_timeout",
                        "crash_type": exc.__class__.__name__,
                        "crash_detail": str(exc),
                    },
                    self.policy_version,
                )
                self._current_obs, self._current_info = self._restart_after_carla_failure(
                    "initial_reset_timeout",
                    self.policy_version,
                )
        return self._current_obs, dict(self._current_info)


def collector_entry(
    collector_id: int,
    config: Any,
    rl_config: Dict[str, Any],
    rollout_dir: str,
    weight_dir: str,
    device: str,
    event_queue: Optional[mp.Queue],
    stop_event: Optional[mp.Event],
    collector_plan: Optional[Dict[str, Any]] = None,
    run_dir: Optional[str] = None,
    event_bus_kind: str = "queue",
    heartbeat_dir: Optional[str] = None,
) -> None:
    event_bus = None
    control_plane = None
    if event_bus_kind == "file":
        if run_dir is None:
            raise ValueError("file event bus requires run_dir")
        from pathlib import Path

        event_bus = FileEventBus(Path(run_dir) / "events", collector_id=collector_id)
        control_plane = FileControlPlane(Path(run_dir) / "control")
    collector = Collector(
        collector_id=collector_id,
        config=config,
        rl_config=rl_config,
        rollout_dir=rollout_dir,
        weight_dir=weight_dir,
        device=torch.device(device),
        event_queue=event_queue,
        stop_event=stop_event,
        event_bus=event_bus,
        control_plane=control_plane,
        collector_plan=collector_plan,
        run_dir=run_dir,
        heartbeat_dir=heartbeat_dir,
    )
    collector.run()
