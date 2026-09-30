"""Per-worker episode assembly for the mixed dynamic/static replay buffer.

Current off-policy collect loop calls ``replay_buffer.add`` on every step.
For the static buffer we need the **whole episode** in order to decide
whether it should be admitted (B2D success / high-completion) and to record
its scenario type.  This module buffers transitions per worker and flushes
the completed episode to caller-supplied callbacks.

The assembler is algorithm-agnostic and works with any observation
representation (scalar array or Dict).  It stores only lightweight
references to the numpy arrays handed in by the collect loop; no copies are
made until a flush happens.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

logger = logging.getLogger("Policy")


@dataclass
class _PendingTransition:
    obs: Any
    next_obs: Any
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    expert_action: Optional[np.ndarray]
    info: Dict[str, Any]
    # Expert SAC-policy pre-tanh Gaussian params (for KL BC).  Both fields
    # must be set together; ``None`` means "no expert distribution available
    # for this transition" -> static/dynamic buffers store mask=0.
    expert_mean: Optional[np.ndarray] = None
    expert_log_std: Optional[np.ndarray] = None


@dataclass
class EpisodeFlushResult:
    """What ``EpisodeAssembler.flush`` returned for a completed episode."""

    worker_id: int
    transitions: List[_PendingTransition]
    info_sequence: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:  # convenience
        return len(self.transitions)


class EpisodeAssembler:
    """Collects per-worker transitions and flushes on episode boundaries.

    Typical usage in the collect loop::

        assembler.record(worker_id, obs, next_obs, action, reward,
                         terminated, truncated, info, expert_action=...)
        if done:
            ep = assembler.flush(worker_id)
            if ep is not None:
                # caller writes `ep.transitions` to the dynamic buffer and
                # optionally to the per-scenario static buffer.

    On a crashed worker, call ``discard(worker_id)`` so the partially
    collected transitions are dropped without entering any buffer.
    """

    def __init__(self, max_episode_length: int = 20000):
        self.max_episode_length = int(max_episode_length)
        self._pending: Dict[int, List[_PendingTransition]] = {}
        self._pending_infos: Dict[int, List[Dict[str, Any]]] = {}

    # ---- lifecycle -------------------------------------------------------

    def record(
        self,
        worker_id: int,
        obs: Any,
        next_obs: Any,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        expert_action: Optional[np.ndarray] = None,
        expert_mean: Optional[np.ndarray] = None,
        expert_log_std: Optional[np.ndarray] = None,
    ) -> None:
        buf = self._pending.setdefault(worker_id, [])
        info_buf = self._pending_infos.setdefault(worker_id, [])

        if len(buf) >= self.max_episode_length:
            # Pathologically long episode – drop oldest to keep memory bounded.
            # In practice this should never hit.
            logger.warning(
                "[EpisodeAssembler] worker=%d exceeded max_episode_length=%d, "
                "dropping oldest transition",
                worker_id, self.max_episode_length,
            )
            buf.pop(0)
            if info_buf:
                info_buf.pop(0)

        buf.append(
            _PendingTransition(
                obs=obs,
                next_obs=next_obs,
                action=np.asarray(action),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                expert_action=(
                    np.asarray(expert_action, dtype=np.float32)
                    if expert_action is not None else None
                ),
                info=dict(info) if isinstance(info, Mapping) else {},
                expert_mean=(
                    np.asarray(expert_mean, dtype=np.float32)
                    if expert_mean is not None else None
                ),
                expert_log_std=(
                    np.asarray(expert_log_std, dtype=np.float32)
                    if expert_log_std is not None else None
                ),
            )
        )
        # Light-weight snapshot of info (shallow copy: events are already dicts).
        info_buf.append(dict(info) if isinstance(info, Mapping) else {})

    def flush(self, worker_id: int) -> Optional[EpisodeFlushResult]:
        """Return the collected episode for ``worker_id`` and clear its state."""
        trans = self._pending.pop(worker_id, None)
        infos = self._pending_infos.pop(worker_id, None)
        if not trans:
            return None
        return EpisodeFlushResult(
            worker_id=worker_id,
            transitions=trans,
            info_sequence=infos or [],
        )

    def discard(self, worker_id: int) -> None:
        """Drop a partially collected episode without flushing (e.g. crash)."""
        self._pending.pop(worker_id, None)
        self._pending_infos.pop(worker_id, None)

    def reset(self) -> None:
        self._pending.clear()
        self._pending_infos.clear()

    # ---- introspection ---------------------------------------------------

    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())

    def active_workers(self) -> List[int]:
        return list(self._pending.keys())
