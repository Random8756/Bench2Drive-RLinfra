"""Off-policy algorithm base class for SAC and TD3."""

from __future__ import annotations

import logging
import mmap
import os
import queue as _queue
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING

import multiprocessing as mp
import numpy as np
import torch
from gymnasium import spaces

from .base_algorithm import BaseAlgorithm, EPISODE_LOG_WINDOW
from ..policies.base_policy import BasePolicy
from ..buffers.episode_assembler import EpisodeAssembler
from ..buffers.mixed_buffer import (
    MixedReplayBuffer,
    _current_static_ratio as _compute_static_ratio,
    _is_scenario_simple,
)
from ..buffers.npz_transition_pool import NpzTransitionPool
from ..buffers.shared_buffer import SharedPrioritizedReplayBuffer
from ..buffers.static_buffer import StaticReplayBuffer
from ..utils.callbacks import BaseCallback
from ..utils.episode_success import classify_episode
from ..utils.obs_utils import ObsUtils

if TYPE_CHECKING:
    from b2d_rlinfra.evaluation.visualization import TrainingVisualizer
    from .train_process import OffPolicyTrainConfig

logger = logging.getLogger("Training Loop")

__layer__ = (4, "Algorithm")


class OffPolicyAlgorithm(BaseAlgorithm):
    """Base class for off-policy algorithms (SAC, TD3).

    The dynamic replay tier is shared-memory PER; optional static replay wraps
    it in ``MixedReplayBuffer``. Subclasses define ``_sample_action`` and
    ``_build_train_config``.
    """

    replay_buffer: SharedPrioritizedReplayBuffer
    policy: BasePolicy

    def __init__(
        self,
        policy: Type[BasePolicy],
        env,
        learning_rate: Union[float, callable],
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        min_ready: int = 1,
        visualizer: Optional["TrainingVisualizer"] = None,
        warmup_source: str = 'random',
        exploration_mode: str = 'gaussian',
        epsilon_greedy: Optional[Any] = None,
        explore_action_repeat: int = 1,
        bc_source: str = 'fixed',
        # Prioritized replay
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_annealing_steps: int = 100_000,
        per_min_priority: float = 1e-6,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[str, torch.device] = 'auto',
        seed: Optional[int] = None,
        config: Optional[Any] = None,
        static_buffer_config: Optional[Dict[str, Any]] = None,
        _init_setup_model: bool = True,
    ):
        """Initialize shared SAC/TD3 training state.

        Most arguments mirror the YAML algorithm section; ``static_buffer_config``
        enables the optional per-scenario static replay tier.
        """
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            config=config,
            _init_setup_model=False,
        )

        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.min_ready = min_ready
        self.l5_visualizer = visualizer

        self._warmup_source = warmup_source
        self._bc_source = bc_source

        # Exploration and action repeat
        # exploration_mode: 'gaussian' keeps the algorithm's native
        # exploration; 'epsilon_greedy' mixes deterministic policy output
        # with predefined exploration actions sampled by ``epsilon_greedy``.
        self._exploration_mode = exploration_mode
        self._epsilon_greedy = epsilon_greedy
        # When > 1, a sampled exploration action persists for N steps; for
        # epsilon_greedy this also forces N exploit steps right after the
        # repeated exploration window.
        self._explore_action_repeat = max(1, int(explore_action_repeat or 1))
        # Per-worker exploration state (repeat counters, cached actions).
        self._exploration_state: Dict[int, Dict[str, Any]] = {}

        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_annealing_steps = per_beta_annealing_steps
        self.per_min_priority = per_min_priority

        # LQR expert-action cache populated by collect_rollouts when
        # warmup_source / bc_source involves the rule-based expert.
        self._cached_expert_actions: Dict[int, Optional[np.ndarray]] = {}

        # Replay buffer is allocated in _setup_model.  When the static
        # replay buffer is enabled, ``self.replay_buffer`` becomes a
        # :class:`MixedReplayBuffer` wrapping both the dynamic PER buffer and
        # the dict of per-scenario static buffers.
        self.replay_buffer: Optional[SharedPrioritizedReplayBuffer] = None

        # Static replay
        # Parsed in ``_init_static_buffer_state`` so subclasses can rely on
        # ``self._static_buffer_enabled`` etc. without repeating parsing logic.
        self._static_buffer_raw_config = dict(static_buffer_config or {})
        self._init_static_buffer_state()

        # Collect-loop bookkeeping.
        self._last_obs: Union[Dict[int, np.ndarray], None] = None
        self._n_collected_steps = 0
        self._pending_obs: Dict[int, np.ndarray] = {}
        self._pending_actions: Dict[int, np.ndarray] = {}

        # Per-worker episode counts (visualizer continuity).
        self._worker_episode_counts: Dict[int, int] = {}

        # Smoothed episode statistics.
        self._ep_reward_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)
        self._ep_length_buffer: deque = deque(maxlen=EPISODE_LOG_WINDOW)

        # Timing accumulators.
        self._timing_env_step_total = 0.0
        self._timing_env_step_count = 0
        self._timing_buffer_add_total = 0.0
        self._timing_buffer_add_count = 0
        self._timing_buffer_sample_total = 0.0
        self._timing_buffer_sample_count = 0
        self._timing_train_total = 0.0
        self._timing_train_count = 0
        self._timing_action_sample_total = 0.0
        self._timing_action_sample_count = 0
        self._timing_obs_stack_total = 0.0
        self._timing_obs_stack_count = 0

        # Async training-process state.
        self._train_process: Optional[mp.Process] = None
        self._train_cmd_queue: Optional[mp.Queue] = None
        self._train_result_queue: Optional[mp.Queue] = None
        self._train_pending = False

        # Shared-memory weight transfer (/dev/shm + mmap).
        self._weight_shm_name: Optional[str] = None
        self._weight_fd: Optional[int] = None
        self._weight_mm: Optional[mmap.mmap] = None
        self._weight_flat: Optional[np.ndarray] = None
        self._weight_keys: Optional[List[str]] = None
        self._weight_shapes: Optional[Dict[str, tuple]] = None
        self._weight_dtypes: Optional[Dict[str, str]] = None
        self._weight_total_elements: int = 0

        self._last_train_result: Optional[Dict[str, Any]] = None

        if _init_setup_model:
            self._setup_model()

    # Static replay configuration

    def _init_static_buffer_state(self) -> None:
        """Parse optional per-scenario static replay configuration."""
        cfg = self._static_buffer_raw_config or {}
        self._static_buffer_enabled = bool(cfg.get("enabled", False))
        self._static_buffer_capacity = int(cfg.get("capacity", 5000))
        self._static_completion_threshold = float(cfg.get("completion_threshold", 90.0))
        self._static_strict_success_threshold = float(
            cfg.get("strict_success_threshold", 99.9)
        )
        self._static_simple_scenarios = list(
            cfg.get("simple_scenarios", [
                "VanillaSignalizedTurnEncounterGreenLight",
                "VanillaSignalizedTurnEncounterRedLight",
                "VanillaNonSignalizedTurn",
                "VanillaNonSignalizedTurnEncounterStopsign",
                "Vanilla*",
            ])
        )
        self._static_ratio_initial = float(cfg.get("static_ratio_initial", 0.0))
        self._static_ratio_final = float(cfg.get("static_ratio_final", 0.4))
        self._static_ratio_anneal_steps = int(cfg.get("static_ratio_anneal_steps", 3_000_000))
        self._static_require_clean_high_completion = bool(
            cfg.get("require_clean_for_high_completion", True)
        )
        # Hardcoded scenarios list takes precedence; otherwise we scan the
        # routes XML when building buffers.
        self._static_scenarios = list(cfg.get("scenarios", []) or [])
        self._static_routes_hint = cfg.get("routes_file")

        # Optional npz-backed reserved area
        npz_cfg = (cfg.get("npz_prefill") or {}) if isinstance(cfg, dict) else {}
        self._npz_prefill_enabled = bool(npz_cfg.get("enabled", False))
        self._npz_prefill_reserved_size = int(npz_cfg.get("reserved_size", 0))
        self._npz_prefill_per_scenario: Dict[str, str] = {}
        per_scen = npz_cfg.get("per_scenario", {}) or {}
        if isinstance(per_scen, dict):
            for k, v in per_scen.items():
                if v is None:
                    continue
                self._npz_prefill_per_scenario[str(k)] = str(v)
        elif isinstance(per_scen, list):
            # Accept also list of "key=value" strings (shell-friendly).
            for item in per_scen:
                if not isinstance(item, str) or "=" not in item:
                    continue
                k, _, v = item.partition("=")
                k = k.strip()
                v = v.strip()
                if k and v:
                    self._npz_prefill_per_scenario[k] = v

        # State that gets populated when the buffer is actually built.
        self._episode_assembler: Optional[EpisodeAssembler] = None
        self._mixed_buffer: Optional[MixedReplayBuffer] = None
        self._static_admitted_counts: Dict[str, int] = {}
        self._static_rejected_counts: Dict[str, int] = {}

    def _discover_scenarios_from_routes(self) -> List[str]:
        """Best-effort auto-detect scenario types from the routes XML.

        Returns an empty list when no routes file can be located.  The caller
        falls back to lazy-creation logging in that case.
        """
        import xml.etree.ElementTree as ET

        # Priority: explicit ``routes_file`` from static_buffer config.
        candidates: List[str] = []
        if self._static_routes_hint:
            candidates.append(str(self._static_routes_hint))

        # Fallback: look up ``env_config['routes']['route_files']`` from
        # ``self.config``. The attribute is ``env_config`` (not ``env``);
        # ``Config`` only renames it to ``env`` inside ``to_dict()``.
        if self.config is not None:
            env_cfg = (
                getattr(self.config, "env_config", None)
                or getattr(self.config, "env", None)
                or {}
            )
            if isinstance(env_cfg, dict):
                routes_cfg = env_cfg.get("routes", {}) or {}
                if isinstance(routes_cfg, dict):
                    files = routes_cfg.get("route_files") or []
                    if isinstance(files, (list, tuple)):
                        candidates.extend(str(f) for f in files)

        scenarios: set = set()
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                for scenario_el in root.iter("scenario"):
                    stype = scenario_el.get("type") or scenario_el.get("name")
                    if stype:
                        scenarios.add(str(stype).strip())
            except Exception as exc:
                logger.warning(
                    "[StaticBuffer] failed to parse routes file %s: %s",
                    path, exc,
                )
        return sorted(scenarios)

    def _build_static_buffers(self) -> Dict[str, StaticReplayBuffer]:
        """Allocate per-scenario static buffers for all non-simple scenarios."""
        if not self._static_buffer_enabled:
            return {}

        store_expert = getattr(self, "_bc_source", "fixed") == "lqr"

        # Determine scenario list.
        scenarios = list(self._static_scenarios)
        if not scenarios:
            scenarios = self._discover_scenarios_from_routes()
        if not scenarios:
            env_cfg = (
                getattr(self.config, "env_config", None)
                or getattr(self.config, "env", None)
                or {}
            )
            route_files = []
            if isinstance(env_cfg, dict):
                route_files = (env_cfg.get("routes", {}) or {}).get("route_files", []) or []
            logger.warning(
                "[StaticBuffer] No scenarios listed in config and routes "
                "auto-detect returned empty (hint=%s, route_files=%s) – "
                "static buffer will skip admission.",
                self._static_routes_hint, route_files,
            )
            return {}

        skipped_simple: List[str] = []
        buffers: Dict[str, StaticReplayBuffer] = {}

        npz_active = (
            self._npz_prefill_enabled
            and self._npz_prefill_reserved_size > 0
            and bool(self._npz_prefill_per_scenario)
        )
        npz_summary: List[str] = []
        # Sanity: warn (not error) about npz entries that don't match any
        # configured scenario.  Hard errors only fire when a scenario asks
        # for a reserved area but its npz dir is missing/empty.
        unmatched_npz = set(self._npz_prefill_per_scenario.keys())

        for name in scenarios:
            if _is_scenario_simple(name, self._static_simple_scenarios):
                skipped_simple.append(name)
                continue
            reserved_size = 0
            npz_pool: Optional[NpzTransitionPool] = None
            npz_dir: Optional[str] = None
            if npz_active and name in self._npz_prefill_per_scenario:
                npz_dir = self._npz_prefill_per_scenario[name]
                unmatched_npz.discard(name)
                reserved_size = self._npz_prefill_reserved_size
                if reserved_size > self._static_buffer_capacity:
                    raise ValueError(
                        f"[StaticBuffer] reserved_size={reserved_size} > "
                        f"capacity={self._static_buffer_capacity} for scenario "
                        f"'{name}' — shrink reserved_size or grow capacity."
                    )
                # NpzTransitionPool raises FileNotFoundError / RuntimeError on
                # missing directory or empty npz set; we propagate.
                npz_pool = NpzTransitionPool(npz_dir)
                npz_summary.append(
                    f"{name}: reserved={reserved_size} pool={len(npz_pool)} from {npz_dir}"
                )
            buffers[name] = StaticReplayBuffer(
                capacity=self._static_buffer_capacity,
                observation_space=self.observation_space,
                action_space=self.action_space,
                scenario_name=name,
                device=self.device,
                use_bitpack=True,
                store_expert_actions=store_expert,
                store_expert_dist=False,
                reserved_size=reserved_size,
                npz_pool=npz_pool,
                npz_dir=npz_dir,
            )
        logger.info(
            "[StaticBuffer] Allocated %d per-scenario buffers (capacity=%d each). "
            "Source=%s; skipped %d simple scenarios: %s",
            len(buffers), self._static_buffer_capacity,
            "config" if self._static_scenarios else "auto-detect",
            len(skipped_simple), skipped_simple,
        )
        if npz_active:
            logger.info(
                "[StaticBuffer] npz prefill enabled: reserved_size=%d "
                "scenarios_with_npz=%d",
                self._npz_prefill_reserved_size, len(npz_summary),
            )
            for line in npz_summary:
                logger.info("[StaticBuffer]   %s", line)
            if unmatched_npz:
                logger.warning(
                    "[StaticBuffer] npz_prefill specified npz dirs for "
                    "scenarios that did not match any allocated buffer: %s",
                    sorted(unmatched_npz),
                )
        return buffers

    def _wrap_with_mixed_buffer(self, dynamic_buffer) -> "MixedReplayBuffer":
        """Construct a MixedReplayBuffer around an existing dynamic PER buffer."""
        static_buffers = self._build_static_buffers()
        mixed = MixedReplayBuffer(
            dynamic_buffer=dynamic_buffer,
            static_buffers=static_buffers,
            simple_scenarios=self._static_simple_scenarios,
            completion_threshold=self._static_completion_threshold,
            static_ratio_initial=self._static_ratio_initial,
            static_ratio_final=self._static_ratio_final,
            static_ratio_anneal_steps=self._static_ratio_anneal_steps,
            num_timesteps_getter=lambda: int(self.num_timesteps),
        )
        self._mixed_buffer = mixed
        self._episode_assembler = EpisodeAssembler()
        return mixed

    def _current_static_ratio(self) -> float:
        if not self._static_buffer_enabled:
            return 0.0
        return _compute_static_ratio(
            int(self.num_timesteps),
            self._static_ratio_initial,
            self._static_ratio_final,
            self._static_ratio_anneal_steps,
        )

    # Model and buffer setup

    def _setup_model(self) -> None:
        """Create the policy and the shared-memory PER replay buffer."""
        lr_schedule = self.get_lr_schedule_fn()

        self.policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lr_schedule,
            **self.policy_kwargs,
        )
        self.policy.to(self.device)

        dynamic_buffer = SharedPrioritizedReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            alpha=self.per_alpha,
            beta=self.per_beta,
            beta_annealing_steps=self.per_beta_annealing_steps,
            min_priority=self.per_min_priority,
            store_expert_actions=(self._bc_source == 'lqr'),
        )

        if self._static_buffer_enabled:
            self.replay_buffer = self._wrap_with_mixed_buffer(dynamic_buffer)
            logger.info(
                "[StaticBuffer] enabled – wrapped PER in MixedReplayBuffer "
                "(capacity_per_scenario=%d, static_ratio %.2f->%.2f over %d steps)",
                self._static_buffer_capacity,
                self._static_ratio_initial,
                self._static_ratio_final,
                self._static_ratio_anneal_steps,
            )
        else:
            self.replay_buffer = dynamic_buffer

    # Subclass action sampling

    def _sample_action(
        self,
        obs,
        deterministic: bool = False,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(env_action, buffer_action)``; must be overridden."""
        raise NotImplementedError

    # Shared TD3/SAC exploration

    def _apply_epsilon_greedy_with_repeat(
        self,
        action: np.ndarray,
        env_indices: List[int],
        repeat_n: int,
    ) -> np.ndarray:
        """Apply epsilon-greedy exploration with optional action repeat.

        ``action`` holds normalized [-1, 1] deterministic policy outputs; rows
        for workers that draw "explore" are replaced by a predefined
        exploration action.

        State per worker:
          * explore_left: remaining repeats of the exploration action
          * cached_action: the exploration action being repeated
                                  (normalized [-1, 1])
          * force_exploit_left: forced exploit steps after the repeat window
        """
        for i, wid in enumerate(env_indices):
            state = self._exploration_state.setdefault(wid, {})
            explore_left = state.get('explore_left', 0)
            force_exploit_left = state.get('force_exploit_left', 0)

            if explore_left > 0:
                # Currently repeating an exploration action.
                action[i] = state['cached_action']
                explore_left -= 1
                state['explore_left'] = explore_left
                if explore_left == 0 and repeat_n > 1:
                    # Repeat window over; enter the forced-exploit phase.
                    state['force_exploit_left'] = repeat_n
            elif force_exploit_left > 0:
                # Forced exploit: keep the actor output (action[i] untouched).
                state['force_exploit_left'] = force_exploit_left - 1
            else:
                # Free phase: random explore/exploit decision.
                if not self._epsilon_greedy.should_exploit(self.num_timesteps):
                    explore_env = self._epsilon_greedy.sample_explore_action()
                    explore_normalized = self.policy.unscale_action(explore_env)
                    action[i] = explore_normalized
                    if repeat_n > 1:
                        # This step counts as the 1st of N repeats.
                        state['cached_action'] = explore_normalized.astype(np.float32).copy()
                        state['explore_left'] = repeat_n - 1
                # else: normal exploit, keep action[i] as-is.
        return np.clip(action, -1, 1)

    def _apply_gaussian_action_repeat(
        self,
        action: np.ndarray,
        env_indices: List[int],
        repeat_n: int,
    ) -> np.ndarray:
        """Make each stochastic/noisy sample persist for ``repeat_n`` steps.

        If a worker is inside a repeat window, return its cached action;
        otherwise cache the freshly sampled action for the next
        (repeat_n - 1) steps.
        """
        for i, wid in enumerate(env_indices):
            state = self._exploration_state.setdefault(wid, {})
            repeat_left = state.get('repeat_left', 0)
            if repeat_left > 0:
                action[i] = state['cached_action']
                state['repeat_left'] = repeat_left - 1
            else:
                state['cached_action'] = action[i].astype(np.float32).copy()
                state['repeat_left'] = repeat_n - 1
        return action

    # Subclass training-process configuration

    def _build_train_config(self) -> "OffPolicyTrainConfig":
        """Build the picklable training config sent to the sub-process."""
        raise NotImplementedError

    # Expert actions

    def _get_lqr_warmup_actions(
        self,
        valid_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached LQR expert actions for the given workers.

        Falls back to ``action_space.sample()`` for workers without cached
        actions. Both elements of the tuple are populated: the first goes to
        the env, the second to the replay buffer.
        """
        batch_size = len(valid_indices)
        action_dim = int(np.prod(self.action_space.shape))
        env_actions = np.zeros((batch_size, action_dim), dtype=np.float32)

        for i, worker_idx in enumerate(valid_indices):
            cached = self._cached_expert_actions.get(worker_idx)
            if cached is not None:
                env_actions[i] = cached
            else:
                env_actions[i] = self.action_space.sample()

        buffer_actions = self.policy.unscale_action(env_actions)
        return env_actions, buffer_actions

    def _cache_expert_actions_from_infos(self, infos: Dict[int, Dict[str, Any]]) -> None:
        """Cache ``info['expert_action']`` per worker_id for LQR warmup."""
        if not isinstance(infos, dict):
            return
        for wid, info in infos.items():
            if not isinstance(info, dict):
                continue
            ea = info.get('expert_action')
            if ea is not None:
                self._cached_expert_actions[wid] = np.asarray(ea, dtype=np.float32)

    # Rollout collection

    def collect_rollouts(
        self,
        env,
        callback: BaseCallback,
        n_steps: int,
    ) -> Tuple[int, int, bool]:
        """Run ``n_steps`` collect steps and trigger training when due.

        Returns ``(collected_steps, collected_episodes, continue_training)``.
        """
        self.policy.set_training_mode(False)
        callback.on_rollout_start()

        result = self._collect_rollouts_standard(env, callback, n_steps)

        callback.on_rollout_end()
        return result

    def _collect_rollouts_standard(
        self,
        env,
        callback: BaseCallback,
        n_steps: int,
    ) -> Tuple[int, int, bool]:
        obs_dict = self._last_obs if isinstance(self._last_obs, dict) else {}
        obs_keys = ObsUtils.get_obs_keys(self.observation_space)
        collected_steps = 0
        collected_episodes = 0

        def _process_step_outputs(
            new_obs_dict: Dict[int, Any],
            rewards: Dict[int, float],
            terminateds: Dict[int, bool],
            truncateds: Dict[int, bool],
            infos: Dict[int, Dict[str, Any]],
        ) -> Dict[int, Any]:
            nonlocal collected_steps, collected_episodes

            if self._warmup_source == 'lqr':
                self._cache_expert_actions_from_infos(infos)

            if self.l5_visualizer is not None:
                vis_actions = {wid: self._pending_actions.get(wid) for wid in rewards}
                vis_obs: Dict[int, Any] = {}
                for wid in rewards:
                    info = infos.get(wid, {})
                    obs_item = new_obs_dict.get(wid)
                    bev_image = info.get("bev_image")
                    if bev_image is not None:
                        if isinstance(obs_item, dict):
                            vis_obs[wid] = dict(obs_item)
                            vis_obs[wid]["bev_image"] = bev_image
                        else:
                            vis_obs[wid] = {"bev_image": bev_image}
                    else:
                        vis_obs[wid] = obs_item
                self.l5_visualizer.process_step_result(
                    actions=vis_actions,
                    obs=vis_obs,
                    rewards=rewards,
                    terminateds=terminateds,
                    truncateds=truncateds,
                    infos=infos,
                )

            for wid in rewards:
                info = infos.get(wid, {})

                if info.get("crashed", False):
                    self._episode_num += 1
                    collected_episodes += 1
                    self._worker_episode_counts[wid] = 0
                    had_pending = wid in self._pending_actions
                    self._pending_obs.pop(wid, None)
                    self._pending_actions.pop(wid, None)
                    # Drop any partially collected trajectory for static admission.
                    if self._episode_assembler is not None:
                        self._episode_assembler.discard(wid)
                    context = self._format_episode_context(info)
                    context_suffix = f" | {context}" if context else ""
                    logger.warning(
                        "Worker %d%s crashed (%s), skipping transition, had_pending_action=%s",
                        wid, context_suffix,
                        info.get('crash_type', 'unknown'),
                        had_pending,
                    )
                    continue

                if wid not in self._pending_actions:
                    logger.debug("Worker %d returned without pending action, skipping", wid)
                    continue

                is_terminated = terminateds.get(wid, False)
                is_truncated = truncateds.get(wid, False)
                done = is_terminated or is_truncated
                next_obs = new_obs_dict.get(wid, self._pending_obs[wid])

                expert_act: Optional[np.ndarray] = None
                if self._bc_source == 'lqr':
                    raw_expert = info.get('expert_action')
                    if raw_expert is not None:
                        raw_expert = np.asarray(raw_expert, dtype=np.float32)
                        expert_act = self.policy.unscale_action(raw_expert)

                _t_buf = time.perf_counter()
                self.replay_buffer.add(
                    obs=self._pending_obs[wid],
                    next_obs=next_obs,
                    action=self._pending_actions[wid],
                    reward=rewards.get(wid, 0.0),
                    terminated=is_terminated,
                    truncated=is_truncated,
                    expert_action=expert_act,
                )
                self._timing_buffer_add_total += time.perf_counter() - _t_buf
                self._timing_buffer_add_count += 1

                # Mirror the transition into the per-worker episode assembler
                # for the static-tier admission decision taken at episode end.
                if self._episode_assembler is not None:
                    self._episode_assembler.record(
                        worker_id=wid,
                        obs=self._pending_obs[wid],
                        next_obs=next_obs,
                        action=self._pending_actions[wid],
                        reward=rewards.get(wid, 0.0),
                        terminated=is_terminated,
                        truncated=is_truncated,
                        info=info,
                        expert_action=expert_act,
                    )

                collected_steps += 1
                self.num_timesteps += 1
                self._n_collected_steps += 1

                if done:
                    collected_episodes += 1
                    self._record_episode_end(
                        worker_id=wid,
                        info=info,
                        terminated=is_terminated,
                        truncated=is_truncated,
                    )
                    # Trajectory is now complete; attempt static admission.
                    self._maybe_flush_episode_to_static(wid, info)

                if self._should_train():
                    self._do_training()

                del self._pending_obs[wid]
                del self._pending_actions[wid]

            next_ready_obs = dict(new_obs_dict)
            for wid in list(next_ready_obs):
                if terminateds.get(wid, False) or truncateds.get(wid, False):
                    del next_ready_obs[wid]

            if self.l5_visualizer is not None:
                for wid in next_ready_obs:
                    info = infos.get(wid, {})
                    if info.get("from_reset", False) or info.get("episode_start", False):
                        self.l5_visualizer.on_episode_start(
                            wid,
                            episode_id=self._worker_episode_counts.get(wid, 0),
                        )

            return next_ready_obs

        while collected_steps < n_steps:
            if not obs_dict:
                new_obs_dict, rewards, terminateds, truncateds, infos = env.step(
                    {}, min_ready=self.min_ready,
                )
                if rewards:
                    obs_dict = _process_step_outputs(
                        new_obs_dict, rewards, terminateds, truncateds, infos,
                    )
                    callback.update_locals(locals())
                    if not callback.on_step():
                        self._last_obs = obs_dict
                        return collected_steps, collected_episodes, False
                    if obs_dict:
                        continue
                else:
                    obs_dict = new_obs_dict
                    if self.l5_visualizer is not None:
                        for wid in obs_dict:
                            self.l5_visualizer.on_episode_start(
                                wid,
                                episode_id=self._worker_episode_counts.get(wid, 0),
                            )
                    if len(obs_dict) < self.min_ready:
                        continue
                continue

            ready_workers = list(obs_dict.keys())
            _t_obs = time.perf_counter()
            stacked_obs = ObsUtils.stack_obs(
                [obs_dict[wid] for wid in ready_workers],
                keys=obs_keys,
                observation_space=self.observation_space,
            )
            self._timing_obs_stack_total += time.perf_counter() - _t_obs
            self._timing_obs_stack_count += 1

            _t_act = time.perf_counter()
            if (self.num_timesteps < self.learning_starts
                    and self._warmup_source == 'lqr'):
                actions_np, buffer_actions = self._get_lqr_warmup_actions(ready_workers)
            else:
                actions_np, buffer_actions = self._sample_action(
                    stacked_obs, env_indices=ready_workers,
                )
            self._timing_action_sample_total += time.perf_counter() - _t_act
            self._timing_action_sample_count += 1

            actions: Dict[int, Any] = {}
            for idx, wid in enumerate(ready_workers):
                action = (
                    int(actions_np[idx])
                    if isinstance(self.action_space, spaces.Discrete)
                    else actions_np[idx]
                )
                actions[wid] = action
                self._pending_obs[wid] = obs_dict[wid]
                self._pending_actions[wid] = buffer_actions[idx]

            _t_env = time.perf_counter()
            new_obs_dict, rewards, terminateds, truncateds, infos = env.step(
                actions, min_ready=self.min_ready,
            )
            self._timing_env_step_total += time.perf_counter() - _t_env
            self._timing_env_step_count += 1
            new_obs_dict = _process_step_outputs(
                new_obs_dict, rewards, terminateds, truncateds, infos,
            )

            callback.update_locals(locals())
            if not callback.on_step():
                self._last_obs = new_obs_dict
                return collected_steps, collected_episodes, False

            obs_dict = new_obs_dict

        self._last_obs = obs_dict
        return collected_steps, collected_episodes, True

    def _maybe_flush_episode_to_static(
        self,
        worker_id: int,
        info: Dict[str, Any],
    ) -> None:
        """Attempt to admit the just-finished trajectory into the static tier.

        No-op when the static buffer is disabled or the assembler is not set
        up.  Safe to call unconditionally at episode end.
        """
        if self._episode_assembler is None or self._mixed_buffer is None:
            return
        ep = self._episode_assembler.flush(worker_id)
        if ep is None or not ep.transitions:
            return

        scenario_name = self._get_scenario_name(info)
        if _is_scenario_simple(scenario_name, self._static_simple_scenarios):
            return

        classification = classify_episode(
            ep.info_sequence,
            completion_threshold=self._static_completion_threshold,
            strict_success_threshold=self._static_strict_success_threshold,
            require_clean_for_high_completion=self._static_require_clean_high_completion,
        )
        if not (classification["is_success"] or classification["is_high_completion"]):
            self._static_rejected_counts[scenario_name] = (
                self._static_rejected_counts.get(scenario_name, 0) + 1
            )
            return

        # Build the payload accepted by StaticReplayBuffer.try_add_episode.
        payload: List[Dict[str, Any]] = []
        for tr in ep.transitions:
            payload.append({
                "obs": tr.obs,
                "next_obs": tr.next_obs,
                "action": tr.action,
                "reward": tr.reward,
                "terminated": tr.terminated,
                "truncated": tr.truncated,
                "expert_action": tr.expert_action,
                "expert_mean": tr.expert_mean,
                "expert_log_std": tr.expert_log_std,
            })

        admitted = self._mixed_buffer.try_add_episode(
            scenario_name,
            payload,
            is_success=classification["is_success"],
            is_high_completion=classification["is_high_completion"],
            route_completed=classification["route_completed"],
        )
        if admitted:
            self._static_admitted_counts[scenario_name] = (
                self._static_admitted_counts.get(scenario_name, 0) + 1
            )
        else:
            self._static_rejected_counts[scenario_name] = (
                self._static_rejected_counts.get(scenario_name, 0) + 1
            )

    def _record_episode_end(
        self,
        worker_id: int,
        info: Dict[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Update per-episode counters and TensorBoard scenario buffers."""
        if not (terminated or truncated):
            return

        self._episode_num += 1
        self._worker_episode_counts[worker_id] = self._worker_episode_counts.get(worker_id, 0) + 1

        ep_reward = info.get('episode_reward', info.get('total_reward', 0))
        ep_length = info.get('episode_length', info.get('total_steps', 0))

        self._record_episode_metrics(info, ep_reward, ep_length)

    # Training process

    def _should_train(self) -> bool:
        return (
            self.num_timesteps > self.learning_starts
            and self._n_collected_steps % self.train_freq == 0
            and self.replay_buffer.size() >= self.batch_size
        )

    def _do_training(self) -> None:
        """Submit a training round to the sub-process (non-blocking)."""
        gradient_steps = (
            self._n_collected_steps if self.gradient_steps < 0
            else self.gradient_steps
        )
        if gradient_steps <= 0:
            return

        # Wait for the previous round (pipeline depth = 1) and reload weights
        # before submitting the next one.
        self._wait_for_training()
        self._ensure_train_process()

        from .train_process import TrainCmd

        lr = self.get_lr_schedule_fn()(self._current_progress_remaining)
        self._train_cmd_queue.put({
            'cmd': TrainCmd.TRAIN,
            'batch_size': self.batch_size,
            'gradient_steps': gradient_steps,
            'num_timesteps': self.num_timesteps,
            'learning_rate': lr,
            # Current dynamic/static batch split for the MixedReplayBuffer
            # view inside the train process (None-safe for plain PER).
            'static_ratio': self._current_static_ratio(),
        })
        self._train_pending = True

    def _ensure_train_process(self) -> None:
        if self._train_process is not None and self._train_process.is_alive():
            return

        from .train_process import _train_process_entry

        # Snapshot policy weights into a /dev/shm flat buffer that the
        # sub-process attaches to via mmap.
        state_dict = self.policy.state_dict()
        self._weight_keys = list(state_dict.keys())
        self._weight_shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
        self._weight_dtypes = {k: str(v.dtype) for k, v in state_dict.items()}
        self._weight_total_elements = sum(v.numel() for v in state_dict.values())

        self._weight_shm_name = f'rl_weights_{uuid.uuid4().hex[:12]}'
        nbytes = self._weight_total_elements * 4  # float32 flat layout
        weight_path = os.path.join('/dev/shm', self._weight_shm_name)
        self._weight_fd = os.open(weight_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self._weight_fd, nbytes)
        self._weight_mm = mmap.mmap(self._weight_fd, nbytes)
        self._weight_flat = np.frombuffer(self._weight_mm, dtype=np.float32)

        self._write_weights_to_shm(state_dict)

        train_config = self._build_train_config()
        buffer_config = self.replay_buffer.get_shared_config()

        if self.device.type == 'cuda':
            dev_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            device_str = f'cuda:{dev_idx}'
        else:
            device_str = str(self.device)

        ctx = mp.get_context('spawn')
        self._train_cmd_queue = ctx.Queue()
        self._train_result_queue = ctx.Queue()

        self._train_process = ctx.Process(
            target=_train_process_entry,
            kwargs={
                'cmd_queue': self._train_cmd_queue,
                'result_queue': self._train_result_queue,
                'buffer_config': buffer_config,
                'policy_class': type(self.policy),
                'policy_kwargs': self.policy_kwargs,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'device_str': device_str,
                'weight_shm_name': self._weight_shm_name,
                'weight_keys': self._weight_keys,
                'weight_shapes': self._weight_shapes,
                'weight_dtypes': self._weight_dtypes,
                'weight_total_elements': self._weight_total_elements,
                'train_config': train_config,
            },
            daemon=True,
            name="rl_train_process",
        )
        self._train_process.start()
        logger.info(
            "Started train process PID=%s (spawn, device=%s)",
            self._train_process.pid, device_str,
        )

    def _write_weights_to_shm(self, state_dict: Dict[str, torch.Tensor]) -> None:
        offset = 0
        for key in self._weight_keys:
            flat = state_dict[key].detach().cpu().numpy().astype(np.float32).ravel()
            n = flat.size
            self._weight_flat[offset:offset + n] = flat
            offset += n

    def _read_weights_from_shm(self) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        offset = 0
        for key in self._weight_keys:
            shape = self._weight_shapes[key]
            n = int(np.prod(shape))
            arr = self._weight_flat[offset:offset + n].copy()
            tensor = torch.from_numpy(arr).reshape(shape)
            dtype_str = self._weight_dtypes[key]
            if 'float32' in dtype_str:
                orig_dtype = torch.float32
            elif 'float16' in dtype_str:
                orig_dtype = torch.float16
            elif 'bfloat16' in dtype_str:
                orig_dtype = torch.bfloat16
            else:
                orig_dtype = torch.float32
            if tensor.dtype != orig_dtype:
                tensor = tensor.to(orig_dtype)
            sd[key] = tensor
            offset += n
        return sd

    def _wait_for_training(self) -> None:
        """Block until the in-flight training round (if any) completes."""
        if not self._train_pending:
            return

        from .train_process import TrainResult

        result_msg: Optional[Dict[str, Any]] = None
        while True:
            try:
                result_msg = self._train_result_queue.get(timeout=10.0)
                break
            except _queue.Empty:
                if self._train_process is not None and not self._train_process.is_alive():
                    exit_code = self._train_process.exitcode
                    logger.error(
                        "Train process (PID=%s) died unexpectedly (exit_code=%s). "
                        "Possible causes: OOM killer, CUDA OOM, segfault. "
                        "Cleaning up and will restart on next training trigger.",
                        self._train_process.pid, exit_code,
                    )
                    self._train_pending = False
                    self._cleanup_dead_train_process()
                    return
                logger.warning("Still waiting for train process result (>10 s) ...")

        self._train_pending = False

        if result_msg['result'] == TrainResult.ERROR:
            logger.error("Train process error: %s", result_msg.get('error'))
            logger.error("Traceback:\n%s", result_msg.get('traceback', 'N/A'))
            if self._train_process is not None and not self._train_process.is_alive():
                logger.warning("Train process exited after error, cleaning up for restart.")
                self._cleanup_dead_train_process()
            else:
                logger.warning("Train process still alive after error, will retry on next command.")
            return

        new_weights = self._read_weights_from_shm()
        self.policy.load_state_dict(new_weights, strict=True)

        self._timing_train_total += result_msg.get('time_total', 0)
        self._timing_train_count += 1
        self._timing_buffer_sample_total += result_msg.get('time_sample', 0)
        self._timing_buffer_sample_count += result_msg.get('sample_count', 0)

        if 'n_updates' in result_msg and hasattr(self, '_n_updates'):
            self._n_updates = result_msg['n_updates']

        self._last_train_result = result_msg

    def _cleanup_dead_train_process(self) -> None:
        if self._train_process is not None:
            try:
                self._train_process.join(timeout=2.0)
            except Exception:
                pass
            self._train_process = None

        if self._weight_mm is not None:
            try:
                self._weight_mm.close()
            except Exception:
                pass
            self._weight_mm = None

        if self._weight_fd is not None:
            try:
                os.close(self._weight_fd)
            except Exception:
                pass
            self._weight_fd = None

        if self._weight_shm_name is not None:
            try:
                os.unlink(os.path.join('/dev/shm', self._weight_shm_name))
            except FileNotFoundError:
                pass
            except Exception:
                pass
            self._weight_shm_name = None

        self._weight_flat = None

        for attr in ('_train_cmd_queue', '_train_result_queue'):
            q = getattr(self, attr, None)
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
                setattr(self, attr, None)

        logger.info("Cleaned up dead train process resources; ready for restart.")

    def _shutdown_train_process(self) -> None:
        if self._train_process is None:
            return

        if self._train_pending:
            try:
                self._wait_for_training()
            except Exception as e:
                logger.warning("Error waiting for training during shutdown: %s", e)

        if self._train_process is None:
            logger.info("Train process already cleaned up during shutdown wait")
            return

        from .train_process import TrainCmd
        try:
            if self._train_cmd_queue is not None:
                self._train_cmd_queue.put({'cmd': TrainCmd.SHUTDOWN})
        except Exception:
            pass

        if self._train_process.is_alive():
            self._train_process.join(timeout=10)
            if self._train_process.is_alive():
                logger.warning("Train process did not exit gracefully, terminating")
                self._train_process.terminate()
                self._train_process.join(timeout=5)

        if self._weight_mm is not None:
            try:
                self._weight_mm.close()
            except Exception:
                pass
        if self._weight_fd is not None:
            try:
                os.close(self._weight_fd)
            except Exception:
                pass
        if self._weight_shm_name is not None:
            try:
                os.unlink(os.path.join('/dev/shm', self._weight_shm_name))
            except FileNotFoundError:
                pass
        self._weight_mm = None
        self._weight_fd = None
        self._weight_shm_name = None
        self._weight_flat = None

        self._train_process = None
        logger.info("Train process shut down")

    # Main learning loop

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
        log_interval: int = 4,
        reset_num_timesteps: bool = True,
    ) -> 'OffPolicyAlgorithm':
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "Environment must be set before training"

        obs_dict, info_dict = self.env.reset(min_ready=self.n_envs)
        self._last_obs = obs_dict
        self._worker_episode_counts = {i: 0 for i in range(self.n_envs)}
        self._pending_obs.clear()
        self._pending_actions.clear()
        if self._warmup_source == 'lqr' and info_dict:
            self._cache_expert_actions_from_infos(info_dict)
        if self.l5_visualizer is not None:
            for wid in obs_dict:
                self.l5_visualizer.on_episode_start(wid, episode_id=self._worker_episode_counts.get(wid, 0))
        self._reset_rollout_training_timer()

        self._n_collected_steps = 0
        self._iteration = 0
        last_log_episode = 0
        self._last_train_result = None

        try:
            while self.num_timesteps < total_timesteps:
                collected_steps, collected_episodes, continue_training = self.collect_rollouts(
                    self.env,
                    callback,
                    n_steps=max(self.train_freq, 1),
                )

                if not continue_training:
                    break

                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                if self._episode_num - last_log_episode >= log_interval:
                    self._wait_for_training()
                    last_log_episode = self._episode_num
                    self._iteration += 1
                    self._dump_logs()

        finally:
            # Always shut down the train sub-process before returning so
            # /dev/shm files don't leak on KeyboardInterrupt etc. The replay
            # buffer is intentionally NOT cleaned up here: segmented training
            # may call ``learn()`` multiple times; the caller is responsible
            # for invoking ``replay_buffer.cleanup()`` at the very end.
            self._shutdown_train_process()

        callback.on_training_end()

        return self

    # Logging

    def _dump_logs(self) -> None:
        time_elapsed = self._get_elapsed_sec()
        sim_real_ratio = self._get_sim_real_ratio(time_elapsed)

        ep_rew_mean = float(np.mean(self._ep_reward_buffer)) if self._ep_reward_buffer else None
        ep_len_mean = float(np.mean(self._ep_length_buffer)) if self._ep_length_buffer else None
        replay_size = self.replay_buffer.size() if self.replay_buffer is not None else 0
        logger.info(
            self._format_progress_line(
                total_timesteps=self._total_timesteps,
                sim_real_ratio=sim_real_ratio,
                reward_mean=ep_rew_mean,
                length_mean=ep_len_mean,
                extra={
                    "replay_size": replay_size,
                    "updates": getattr(self, "_n_updates", 0),
                },
            )
        )

        def _avg_ms(total: float, count: int) -> float:
            return (total / count * 1000) if count > 0 else 0.0

        env_step_avg = _avg_ms(self._timing_env_step_total, self._timing_env_step_count)
        buf_add_avg = _avg_ms(self._timing_buffer_add_total, self._timing_buffer_add_count)
        buf_sample_avg = _avg_ms(self._timing_buffer_sample_total, self._timing_buffer_sample_count)
        train_avg = _avg_ms(self._timing_train_total, self._timing_train_count)
        act_sample_avg = _avg_ms(self._timing_action_sample_total, self._timing_action_sample_count)
        obs_stack_avg = _avg_ms(self._timing_obs_stack_total, self._timing_obs_stack_count)

        total_tracked = (
            self._timing_env_step_total + self._timing_buffer_add_total
            + self._timing_buffer_sample_total + self._timing_train_total
            + self._timing_action_sample_total + self._timing_obs_stack_total
        )

        tracked_wall_pct = (total_tracked / time_elapsed * 100) if time_elapsed > 0 else 0.0
        logger.debug(
            "[timing] env_step_ms=%.2f buffer_add_ms=%.2f buffer_sample_ms=%.2f "
            "train_ms=%.2f action_sample_ms=%.2f obs_stack_ms=%.2f tracked_wall_pct=%.1f",
            env_step_avg,
            buf_add_avg,
            buf_sample_avg,
            train_avg,
            act_sample_avg,
            obs_stack_avg,
            tracked_wall_pct,
        )

        if self.logger is None:
            return

        self.logger.record('time/iterations', self._iteration)
        self.logger.record('time/episodes', self._episode_num)
        self.logger.record('time/elapsed_sec', float(time_elapsed))
        self.logger.record('time/sim_real_ratio', float(sim_real_ratio))
        self.logger.record('rollout/replay_buffer_size', self.replay_buffer.size())
        if self._mixed_buffer is not None:
            self.logger.record(
                'rollout/static_buffer_size', self._mixed_buffer.static_total_size()
            )
            self.logger.record('rollout/static_ratio', self._current_static_ratio())
            self.logger.record(
                'rollout/static_admitted_episodes',
                int(sum(self._static_admitted_counts.values())),
            )
        self._record_episode_tensorboard_metrics()

        if self._last_train_result is not None:
            r = self._last_train_result
            if r.get('critic_loss') is not None:
                self.logger.record('train/value_loss', r['critic_loss'])
            if r.get('actor_loss') is not None:
                self.logger.record('train/policy_loss', r['actor_loss'])
            if r.get('learning_rate') is not None:
                self.logger.record('train/learning_rate', r['learning_rate'])
            if r.get('ent_coef') is not None:
                self.logger.record('train/ent_coef', r['ent_coef'])
            if hasattr(self, '_get_exploration_noise_scale') and getattr(self, 'exploration_noise_decay_steps', 0) > 0:
                self.logger.record('train/action_noise_scale', float(self._get_exploration_noise_scale()))
            if r.get('explained_var') is not None:
                self.logger.record('train/explained_variance', r['explained_var'])
            if r.get('n_updates') is not None:
                self.logger.record('train/n_updates', r['n_updates'])
            moe_aux_losses = []
            if r.get('moe_actor_aux_loss') is not None:
                moe_aux_losses.append(float(r['moe_actor_aux_loss']))
            if r.get('moe_critic_aux_loss') is not None:
                moe_aux_losses.append(float(r['moe_critic_aux_loss']))
            if moe_aux_losses:
                self.logger.record('moe/load_balance_loss', float(np.mean(moe_aux_losses)))
            if r.get('moe_actor_gate_entropy') is not None:
                self.logger.record('moe/policy_router_entropy', r['moe_actor_gate_entropy'])
            if r.get('moe_critic_gate_entropy') is not None:
                self.logger.record('moe/value_router_entropy', r['moe_critic_gate_entropy'])
            for i, u in enumerate(r.get('moe_actor_usage') or []):
                self.logger.record(f'moe/policy_expert_usage_rate/{i}', float(u))
            for i, u in enumerate(r.get('moe_critic_usage') or []):
                self.logger.record(f'moe/value_expert_usage_rate/{i}', float(u))

        self.logger.dump(step=self.num_timesteps)

    # Persistence

    def _get_save_data(self) -> Dict[str, Any]:
        return {
            'buffer_size': self.buffer_size,
            'learning_starts': self.learning_starts,
            'batch_size': self.batch_size,
            'tau': self.tau,
            'gamma': self.gamma,
            'train_freq': self.train_freq,
            'gradient_steps': self.gradient_steps,
        }
