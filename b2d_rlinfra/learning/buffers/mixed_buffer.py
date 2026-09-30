"""Replay buffer wrapper that mixes dynamic PER with static scenario buffers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .npz_transition_pool import NpzTransitionPool
from .shared_buffer import (
    PrioritizedReplayBufferSamples,
    SharedPrioritizedReplayBuffer,
)
from .static_buffer import StaticReplayBuffer

logger = logging.getLogger("Policy")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _current_static_ratio(
    num_timesteps: int,
    initial: float,
    final: float,
    anneal_steps: int,
) -> float:
    if anneal_steps <= 0:
        return float(final)
    progress = min(1.0, max(0.0, float(num_timesteps) / float(anneal_steps)))
    return float(initial) + (float(final) - float(initial)) * progress


def _is_scenario_simple(
    scenario_name: str,
    simple_scenarios: Sequence[str],
) -> bool:
    if not scenario_name:
        return False
    for pattern in simple_scenarios:
        if not pattern:
            continue
        if pattern.endswith("*"):
            if scenario_name.startswith(pattern[:-1]):
                return True
        elif scenario_name == pattern:
            return True
    return False


# ---------------------------------------------------------------------------
# Mixed buffer
# ---------------------------------------------------------------------------


class MixedReplayBuffer:
    """Dispatches sampling between a PER dynamic buffer and static buffers."""

    # ---- construction ----------------------------------------------------

    def __init__(
        self,
        dynamic_buffer: SharedPrioritizedReplayBuffer,
        *,
        static_buffers: Dict[str, StaticReplayBuffer],
        simple_scenarios: Sequence[str],
        completion_threshold: float,
        static_ratio_initial: float,
        static_ratio_final: float,
        static_ratio_anneal_steps: int,
        num_timesteps_getter: Optional[callable] = None,
    ):
        self.dynamic = dynamic_buffer
        self.static_buffers: Dict[str, StaticReplayBuffer] = dict(static_buffers)
        self.simple_scenarios = list(simple_scenarios)
        self.completion_threshold = float(completion_threshold)
        self.static_ratio_initial = float(static_ratio_initial)
        self.static_ratio_final = float(static_ratio_final)
        self.static_ratio_anneal_steps = int(static_ratio_anneal_steps)
        self._num_timesteps_getter = num_timesteps_getter or (lambda: 0)

        # Stats (owner-side only).
        self._stats_admitted_success = 0
        self._stats_admitted_high_completion = 0
        self._stats_rejected = 0
        self._stats_last_scenarios: Dict[str, int] = {}
        # Counter of how many reserved-slot refills the view side has
        # performed during sampling (cheap diagnostic).
        self._stats_reserved_refilled = 0

        # View-side only: synced timestep state, if the train process needs it.
        self._view_ratio_override: Optional[float] = None

    # ---- admission -------------------------------------------------------

    def try_add_episode(
        self,
        scenario_name: str,
        transitions: List[Dict[str, Any]],
        *,
        is_success: bool,
        is_high_completion: bool,
        route_completed: float,
    ) -> bool:
        """Write a full episode into the static tier if it qualifies.

        Parameters mirror ``StaticReplayBuffer.try_add_episode``.  Returns
        True iff the episode was admitted.
        """
        if not transitions:
            return False

        if _is_scenario_simple(scenario_name, self.simple_scenarios):
            return False

        if not (is_success or is_high_completion):
            self._stats_rejected += 1
            return False

        sb = self.static_buffers.get(scenario_name)
        if sb is None:
            # Unknown scenario: no pre-allocated buffer for it. Log once
            # per scenario to avoid log spam.
            if scenario_name not in self._stats_last_scenarios:
                logger.warning(
                    "[MixedReplayBuffer] No static buffer for scenario '%s' "
                    "(not in config); skipping static insertion.",
                    scenario_name,
                )
            self._stats_last_scenarios[scenario_name] = \
                self._stats_last_scenarios.get(scenario_name, 0) + 1
            return False

        admitted = sb.try_add_episode(
            transitions,
            is_success=is_success,
            route_completed=route_completed,
        )
        if admitted:
            if is_success:
                self._stats_admitted_success += 1
            else:
                self._stats_admitted_high_completion += 1
        else:
            self._stats_rejected += 1
        return admitted

    # ---- owner-side dynamic add passthrough ------------------------------

    def add_transition(self, *args, **kwargs) -> None:
        """Forward to ``dynamic.add``; kept for callsite clarity."""
        self.dynamic.add(*args, **kwargs)

    def add(self, *args, **kwargs) -> None:
        """Alias for :meth:`add_transition`, matching the standard buffer API."""
        self.dynamic.add(*args, **kwargs)

    # ---- size helpers ----------------------------------------------------

    def can_sample(self, batch_size: int) -> bool:
        return self.dynamic.can_sample(batch_size)

    def size(self) -> int:
        return self.dynamic.size()

    def static_total_size(self) -> int:
        return sum(sb.size() for sb in self.static_buffers.values())

    def static_size_breakdown(self) -> Dict[str, int]:
        return {name: sb.size() for name, sb in self.static_buffers.items()}

    def static_success_breakdown(self) -> Dict[str, int]:
        return {
            name: sb.num_success_entries()
            for name, sb in self.static_buffers.items()
        }

    def static_reserved_breakdown(self) -> Dict[str, int]:
        return {
            name: int(getattr(sb, "reserved_size", 0))
            for name, sb in self.static_buffers.items()
        }

    @property
    def reserved_refilled_total(self) -> int:
        """Total number of reserved-slot refills performed by this view."""
        return int(self._stats_reserved_refilled)

    # ---- PER pass-through ------------------------------------------------

    def update_priorities(self, tree_indices, td_errors) -> None:
        """Update PER priorities, ignoring sentinel -1 indices from static samples."""
        tree_indices = np.asarray(tree_indices).reshape(-1)
        td_errors = np.asarray(td_errors).reshape(-1)
        mask = tree_indices >= 0
        if not mask.any():
            return
        self.dynamic.update_priorities(tree_indices[mask], td_errors[mask])

    def boost_priorities(self, buffer_indices, factors) -> None:
        """Forward tail-boost to the dynamic PER buffer (no-op on static tier)."""
        if hasattr(self.dynamic, "boost_priorities"):
            self.dynamic.boost_priorities(buffer_indices, factors)

    # ---- shared config (for spawning train process) ----------------------

    def get_shared_config(self) -> Dict[str, Any]:
        return {
            "dynamic": self.dynamic.get_shared_config(),
            "static": {
                name: sb.get_shared_config()
                for name, sb in self.static_buffers.items()
            },
            "simple_scenarios": list(self.simple_scenarios),
            "static_ratio_initial": self.static_ratio_initial,
            "static_ratio_final": self.static_ratio_final,
            "static_ratio_anneal_steps": self.static_ratio_anneal_steps,
        }

    @classmethod
    def from_shared(cls, config: Mapping[str, Any], device="cpu") -> "MixedReplayBuffer":
        dynamic = SharedPrioritizedReplayBuffer.from_shared(config["dynamic"], device=device)
        static_buffers: Dict[str, StaticReplayBuffer] = {}
        for name, cfg in config.get("static", {}).items():
            sb = StaticReplayBuffer.from_shared(cfg, device=device)
            # Reconstruct the npz pool on the view side so reserved-slot
            # refills don't need any IPC: each train sub-process opens its own
            # mmap'd handles to the same files the owner used.
            npz_dir = getattr(sb, "_npz_dir", None)
            if int(getattr(sb, "reserved_size", 0)) > 0 and npz_dir:
                try:
                    pool = NpzTransitionPool(npz_dir)
                    sb.attach_view_npz_pool(pool)
                except Exception as exc:  # pragma: no cover (defensive)
                    logger.warning(
                        "[MixedReplayBuffer.from_shared] failed to open npz "
                        "pool for scenario %s at %s: %s — reserved refill "
                        "will be disabled in this view.",
                        name, npz_dir, exc,
                    )
            static_buffers[name] = sb

        obj = cls.__new__(cls)
        obj.dynamic = dynamic
        obj.static_buffers = static_buffers
        obj.simple_scenarios = list(config.get("simple_scenarios", []))
        obj.completion_threshold = 0.0  # not needed on view side
        obj.static_ratio_initial = float(config.get("static_ratio_initial", 0.0))
        obj.static_ratio_final = float(config.get("static_ratio_final", 0.0))
        obj.static_ratio_anneal_steps = int(config.get("static_ratio_anneal_steps", 1))
        obj._num_timesteps_getter = lambda: 0
        obj._stats_admitted_success = 0
        obj._stats_admitted_high_completion = 0
        obj._stats_rejected = 0
        obj._stats_last_scenarios = {}
        obj._stats_reserved_refilled = 0
        obj._view_ratio_override = None
        return obj

    def set_view_ratio_override(self, ratio: Optional[float]) -> None:
        """Set the static-tier ratio on the train-process side.

        The training sub-process doesn't have direct access to the owner's
        ``num_timesteps``; the owner instead sends the current ratio via the
        TRAIN command.  Passing ``None`` falls back to the schedule based on a
        caller-supplied ``num_timesteps`` handed to ``sample``.
        """
        self._view_ratio_override = ratio

    # ---- sampling --------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        *,
        num_timesteps: Optional[int] = None,
    ) -> PrioritizedReplayBufferSamples:
        """Return a combined batch of ``batch_size`` transitions.

        Parameters
        ----------
        batch_size:
            Total samples to return.
        num_timesteps:
            Optional override used to compute ``static_ratio(t)``.  If None
            the internal ``_num_timesteps_getter`` is consulted.

        Implementation detail:
        the number of static samples is clipped to
        ``min(n_requested, total_static_size)``; if the static tier is still
        empty we simply fall back to a pure PER batch.
        """
        # Determine ratio.
        if self._view_ratio_override is not None:
            ratio = float(self._view_ratio_override)
        else:
            t = int(num_timesteps) if num_timesteps is not None else int(self._num_timesteps_getter())
            ratio = _current_static_ratio(
                t,
                self.static_ratio_initial,
                self.static_ratio_final,
                self.static_ratio_anneal_steps,
            )

        total_static = self.static_total_size()
        if total_static <= 0:
            return self.dynamic.sample(batch_size)

        n_static_desired = int(round(batch_size * ratio))
        n_static = min(n_static_desired, total_static, batch_size)
        n_dynamic = batch_size - n_static
        if n_dynamic <= 0:
            n_dynamic = 1
            n_static = batch_size - 1

        dynamic_batch = self.dynamic.sample(n_dynamic)
        if n_static <= 0:
            return dynamic_batch

        static_batch = self._sample_static_batch(n_static)
        if static_batch is None:
            return dynamic_batch

        return self._concatenate_batches(dynamic_batch, static_batch, n_dynamic, n_static)

    # ---- static sampling core -------------------------------------------

    def _sample_static_batch(self, n_total: int) -> Optional[Dict[str, Any]]:
        """Draw ``n_total`` transitions from the static tier, reweighting
        scenarios so that smaller buffers are oversampled.
        """
        # Collect non-empty scenarios with their sizes.
        names: List[str] = []
        sizes: List[int] = []
        for name, sb in self.static_buffers.items():
            s = sb.size()
            if s > 0:
                names.append(name)
                sizes.append(s)
        if not names:
            return None

        sizes_arr = np.asarray(sizes, dtype=np.float64)
        weights = 1.0 / (sizes_arr + 1.0)
        weights = weights / weights.sum()

        # Allocate per-scenario sample counts via multinomial draw.
        counts = np.random.multinomial(n_total, weights)

        indices_by_scenario: Dict[str, np.ndarray] = {}
        for name, cnt in zip(names, counts):
            if cnt <= 0:
                continue
            sb = self.static_buffers[name]
            idx = sb.sample_raw_indices(int(cnt))
            if idx is None:
                continue
            indices_by_scenario[name] = idx

        if not indices_by_scenario:
            return None

        obs_parts: List[Any] = []
        nobs_parts: List[Any] = []
        action_parts: List[np.ndarray] = []
        reward_parts: List[np.ndarray] = []
        term_parts: List[np.ndarray] = []
        trunc_parts: List[np.ndarray] = []
        expert_parts: List[np.ndarray] = []
        has_any_expert = False
        emean_parts: List[Any] = []
        elogstd_parts: List[Any] = []
        emask_parts: List[Any] = []
        has_any_expert_dist = False

        obs_is_dict = self.dynamic._obs_is_dict
        obs_keys = self.dynamic.obs_keys if obs_is_dict else None

        for name, idx in indices_by_scenario.items():
            sb = self.static_buffers[name]
            part = sb.gather(idx)
            obs_parts.append(part["observations"])
            nobs_parts.append(part["next_observations"])
            action_parts.append(np.asarray(part["actions"]))
            reward_parts.append(np.asarray(part["rewards"]))
            term_parts.append(np.asarray(part["terminateds"]))
            trunc_parts.append(np.asarray(part["truncateds"]))
            ea = part.get("expert_actions")
            if ea is not None:
                expert_parts.append(np.asarray(ea))
                has_any_expert = True
            else:
                expert_parts.append(None)  # placeholder

            em = part.get("expert_mean")
            el = part.get("expert_log_std")
            emk = part.get("expert_dist_mask")
            if em is not None and el is not None and emk is not None:
                emean_parts.append(np.asarray(em))
                elogstd_parts.append(np.asarray(el))
                emask_parts.append(np.asarray(emk))
                has_any_expert_dist = True
            else:
                emean_parts.append(None)
                elogstd_parts.append(None)
                emask_parts.append(None)

            # Reserved-slot refill (view-side immediate): after the batch has
            # been gathered, replace every consumed reserved slot with a fresh
            # transition drawn from the scenario's npz pool.  ``gather`` returns
            # copies (numpy fancy indexing or torch.from_numpy().copy()), so
            # subsequent in-place writes do not mutate the batch we just built.
            try:
                refilled = sb.refresh_reserved_in_indices(idx)
                if refilled:
                    self._stats_reserved_refilled += int(refilled)
            except Exception as exc:  # pragma: no cover (defensive)
                logger.warning(
                    "[MixedReplayBuffer] reserved refill failed for scenario %s: %s",
                    name, exc,
                )

        actions_np = np.concatenate(action_parts, axis=0)
        rewards_np = np.concatenate(reward_parts, axis=0).reshape(-1, 1)
        terminateds_np = np.concatenate(term_parts, axis=0).reshape(-1, 1)
        _ = np.concatenate(trunc_parts, axis=0).reshape(-1, 1)  # not returned

        expert_actions_out: Optional[np.ndarray] = None
        if has_any_expert:
            filled: List[np.ndarray] = []
            for ea, acts in zip(expert_parts, action_parts):
                if ea is None:
                    filled.append(np.zeros_like(acts, dtype=np.float32))
                else:
                    filled.append(ea.astype(np.float32, copy=False))
            expert_actions_out = np.concatenate(filled, axis=0)

        expert_mean_out: Optional[np.ndarray] = None
        expert_log_std_out: Optional[np.ndarray] = None
        expert_dist_mask_out: Optional[np.ndarray] = None
        if has_any_expert_dist:
            filled_em: List[np.ndarray] = []
            filled_el: List[np.ndarray] = []
            filled_emk: List[np.ndarray] = []
            for em, el, emk, acts in zip(
                emean_parts, elogstd_parts, emask_parts, action_parts
            ):
                zero_dist = np.zeros_like(acts, dtype=np.float32)
                zero_mask = np.zeros((acts.shape[0],), dtype=np.uint8)
                if em is None or el is None or emk is None:
                    filled_em.append(zero_dist)
                    filled_el.append(zero_dist)
                    filled_emk.append(zero_mask)
                else:
                    filled_em.append(em.astype(np.float32, copy=False))
                    filled_el.append(el.astype(np.float32, copy=False))
                    filled_emk.append(np.asarray(emk, dtype=np.uint8).reshape(-1))
            expert_mean_out = np.concatenate(filled_em, axis=0)
            expert_log_std_out = np.concatenate(filled_el, axis=0)
            expert_dist_mask_out = np.concatenate(filled_emk, axis=0)

        # Stitch observations back.  Depending on bit-pack mode, parts may be
        # torch tensors (already on GPU) or numpy arrays.
        obs_cat = self._concat_obs_parts(obs_parts, obs_is_dict, obs_keys)
        nobs_cat = self._concat_obs_parts(nobs_parts, obs_is_dict, obs_keys)

        return {
            "observations": obs_cat,
            "next_observations": nobs_cat,
            "actions": actions_np,
            "rewards": rewards_np,
            "dones": terminateds_np,
            "expert_actions": expert_actions_out,
            "expert_mean": expert_mean_out,
            "expert_log_std": expert_log_std_out,
            "expert_dist_mask": expert_dist_mask_out,
        }

    @staticmethod
    def _concat_obs_parts(parts: List[Any], is_dict: bool, keys: Optional[List[str]]) -> Any:
        if is_dict:
            out: Dict[str, Any] = {}
            for key in keys:
                values = [p[key] for p in parts]
                if isinstance(values[0], torch.Tensor):
                    out[key] = torch.cat(values, dim=0)
                else:
                    out[key] = np.concatenate(values, axis=0)
            return out
        else:
            if isinstance(parts[0], torch.Tensor):
                return torch.cat(parts, dim=0)
            return np.concatenate(parts, axis=0)

    # ---- batch concatenation --------------------------------------------

    def _concatenate_batches(
        self,
        dyn: PrioritizedReplayBufferSamples,
        stat: Dict[str, Any],
        n_dynamic: int,
        n_static: int,
    ) -> PrioritizedReplayBufferSamples:
        device = self.dynamic.device

        # --- observations ---
        obs_cat = self._join_obs(dyn.observations, stat["observations"], device)
        nobs_cat = self._join_obs(dyn.next_observations, stat["next_observations"], device)

        # --- scalars ---
        actions_dyn = dyn.actions
        actions_stat_t = torch.as_tensor(stat["actions"], dtype=actions_dyn.dtype, device=device)
        actions_cat = torch.cat([actions_dyn, actions_stat_t], dim=0)

        rewards_dyn = dyn.rewards
        rewards_stat_t = torch.as_tensor(stat["rewards"], dtype=rewards_dyn.dtype, device=device)
        rewards_cat = torch.cat([rewards_dyn, rewards_stat_t], dim=0)

        dones_dyn = dyn.dones
        dones_stat_t = torch.as_tensor(stat["dones"], dtype=dones_dyn.dtype, device=device)
        dones_cat = torch.cat([dones_dyn, dones_stat_t], dim=0)

        # --- IS weights: dynamic keeps its PER weights; static = 1.0 ---
        weights_dyn = dyn.weights
        weights_stat = torch.ones((n_static, 1), dtype=weights_dyn.dtype, device=device)
        weights_cat = torch.cat([weights_dyn, weights_stat], dim=0)

        # --- Tree indices: -1 sentinel for static section ---
        idx_dyn = dyn.indices
        idx_dyn_np = np.asarray(idx_dyn)
        idx_stat_np = -np.ones(n_static, dtype=idx_dyn_np.dtype)
        indices_cat = np.concatenate([idx_dyn_np, idx_stat_np], axis=0)

        # --- expert actions: optional ---
        expert_cat: Optional[torch.Tensor]
        if dyn.expert_actions is not None or stat.get("expert_actions") is not None:
            if dyn.expert_actions is None:
                ea_dyn_np = np.zeros((n_dynamic, self.dynamic.action_dim), dtype=np.float32)
                ea_dyn = torch.as_tensor(ea_dyn_np, dtype=torch.float32, device=device)
            else:
                ea_dyn = dyn.expert_actions
            if stat.get("expert_actions") is None:
                ea_stat_np = np.zeros((n_static, self.dynamic.action_dim), dtype=np.float32)
            else:
                ea_stat_np = stat["expert_actions"].astype(np.float32, copy=False)
            ea_stat = torch.as_tensor(ea_stat_np, dtype=ea_dyn.dtype, device=device)
            expert_cat = torch.cat([ea_dyn, ea_stat], dim=0)
        else:
            expert_cat = None

        # --- expert distribution params: optional ---
        expert_mean_cat: Optional[torch.Tensor] = None
        expert_log_std_cat: Optional[torch.Tensor] = None
        expert_dist_mask_cat: Optional[torch.Tensor] = None
        if (
            dyn.expert_mean is not None
            or stat.get("expert_mean") is not None
        ):
            action_dim = self.dynamic.action_dim
            if dyn.expert_mean is None:
                em_dyn = torch.zeros((n_dynamic, action_dim), dtype=torch.float32, device=device)
                el_dyn = torch.zeros((n_dynamic, action_dim), dtype=torch.float32, device=device)
                emk_dyn = torch.zeros((n_dynamic, 1), dtype=torch.float32, device=device)
            else:
                em_dyn = dyn.expert_mean
                el_dyn = dyn.expert_log_std
                emk_dyn = dyn.expert_dist_mask
            if stat.get("expert_mean") is None:
                em_stat = torch.zeros((n_static, action_dim), dtype=em_dyn.dtype, device=device)
                el_stat = torch.zeros((n_static, action_dim), dtype=el_dyn.dtype, device=device)
                emk_stat = torch.zeros((n_static, 1), dtype=emk_dyn.dtype, device=device)
            else:
                em_stat_np = stat["expert_mean"].astype(np.float32, copy=False)
                el_stat_np = stat["expert_log_std"].astype(np.float32, copy=False)
                emk_stat_np = (
                    stat["expert_dist_mask"].astype(np.float32, copy=False).reshape(-1, 1)
                )
                em_stat = torch.as_tensor(em_stat_np, dtype=em_dyn.dtype, device=device)
                el_stat = torch.as_tensor(el_stat_np, dtype=el_dyn.dtype, device=device)
                emk_stat = torch.as_tensor(emk_stat_np, dtype=emk_dyn.dtype, device=device)
            expert_mean_cat = torch.cat([em_dyn, em_stat], dim=0)
            expert_log_std_cat = torch.cat([el_dyn, el_stat], dim=0)
            expert_dist_mask_cat = torch.cat([emk_dyn, emk_stat], dim=0)

        return PrioritizedReplayBufferSamples(
            observations=obs_cat,
            actions=actions_cat,
            next_observations=nobs_cat,
            dones=dones_cat,
            rewards=rewards_cat,
            weights=weights_cat,
            indices=indices_cat,
            expert_actions=expert_cat,
            expert_mean=expert_mean_cat,
            expert_log_std=expert_log_std_cat,
            expert_dist_mask=expert_dist_mask_cat,
        )

    def _join_obs(self, dyn_obs, stat_obs, device) -> Any:
        if isinstance(dyn_obs, dict):
            out = {}
            for key in dyn_obs:
                d = dyn_obs[key]
                s = stat_obs[key] if isinstance(stat_obs, dict) else stat_obs
                if isinstance(d, torch.Tensor):
                    if not isinstance(s, torch.Tensor):
                        s = torch.as_tensor(s, dtype=d.dtype, device=device)
                    else:
                        s = s.to(dtype=d.dtype, device=device)
                    out[key] = torch.cat([d, s], dim=0)
                else:
                    # numpy dyn obs (non-bitpack): convert both to torch.
                    d_t = torch.as_tensor(d, dtype=torch.float32, device=device)
                    if isinstance(s, torch.Tensor):
                        s_t = s.to(dtype=d_t.dtype, device=device)
                    else:
                        s_t = torch.as_tensor(s, dtype=d_t.dtype, device=device)
                    out[key] = torch.cat([d_t, s_t], dim=0)
            return out
        # Non-dict observations.
        if isinstance(dyn_obs, torch.Tensor):
            if not isinstance(stat_obs, torch.Tensor):
                stat_obs = torch.as_tensor(stat_obs, dtype=dyn_obs.dtype, device=device)
            else:
                stat_obs = stat_obs.to(dtype=dyn_obs.dtype, device=device)
            return torch.cat([dyn_obs, stat_obs], dim=0)
        d_t = torch.as_tensor(dyn_obs, dtype=torch.float32, device=device)
        s_t = torch.as_tensor(stat_obs, dtype=d_t.dtype, device=device)
        return torch.cat([d_t, s_t], dim=0)

    # ---- cleanup ---------------------------------------------------------

    def cleanup(self) -> None:
        try:
            self.dynamic.cleanup()
        except Exception:
            pass
        for sb in self.static_buffers.values():
            try:
                sb.cleanup()
            except Exception:
                pass
