"""Lazy-mmap pool of transitions sampled from a directory of ``.npz`` episodes.

Used by :class:`StaticReplayBuffer` to **prefill** a fixed number of "reserved"
slots that bypass the normal episode-level admission/eviction policy.  Each
time a reserved slot is consumed by the train sub-process during sampling, it
is refilled by drawing a fresh transition from this pool.

Expected on-disk layout
=======================

The pool scans (recursively) a directory for ``*.npz`` episode files with the
following arrays::

    actions          : float32  [T, action_dim]
    expert_actions   : float32  [T, action_dim]   (optional dataset field; defaults to 0)
    rewards          : float32  [T]
    terminateds      : bool     [T]
    truncateds       : bool     [T]
    obs_<key>        : <dtype>  [T+1, *shape]     (one entry per Dict obs key)
        OR
    observations     : <dtype>  [T+1, *obs_shape]

A transition at local index ``t`` is reconstructed as::

    obs            = obs[t]
    next_obs       = obs[t+1]
    action         = actions[t]
    expert_action  = expert_actions[t] if present, otherwise zeros_like(action)
    reward         = rewards[t]
    terminated     = terminateds[t]
    truncated      = truncateds[t]

Sampling
========

``sample_one()`` picks a transition uniformly across the *entire* pool
(weighted by the number of transitions per file, so longer episodes contribute
more samples — i.e. every transition is equally likely).
"""

from __future__ import annotations

import bisect
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger("Policy")


class NpzTransitionPool:
    """Lazy, thread-safe pool of transitions backed by a directory of npz files.

    Parameters
    ----------
    npz_dir:
        Directory whose ``*.npz`` files (recursively) form the sampling pool.
    recursive:
        If True (default), traverse subdirectories.
    obs_keys:
        Optional whitelist of dict-obs keys to return.  When ``None`` (default)
        the full set of ``obs_<key>`` arrays found in the first file is used.
        Box-obs files always return a single ``obs``/``next_obs`` array.
    cache_files:
        Whether to cache opened ``np.load`` handles for reuse across draws.
        Defaults to True (recommended for speed).
    """

    def __init__(
        self,
        npz_dir: Union[str, Path],
        *,
        recursive: bool = True,
        obs_keys: Optional[Sequence[str]] = None,
        cache_files: bool = True,
    ):
        self.npz_dir = Path(npz_dir).expanduser().resolve()
        if not self.npz_dir.exists() or not self.npz_dir.is_dir():
            raise FileNotFoundError(
                f"[NpzTransitionPool] directory not found: {self.npz_dir}"
            )

        if recursive:
            raw_files: List[Path] = sorted(self.npz_dir.rglob("*.npz"))
        else:
            raw_files = sorted(self.npz_dir.glob("*.npz"))
        if not raw_files:
            raise FileNotFoundError(
                f"[NpzTransitionPool] no *.npz files under: {self.npz_dir}"
            )

        # Header-only scan: keep files with T > 0, build the cumulative table
        # in lock-step so that index ``i`` of ``self.files``,
        # ``self._t_per_file`` and ``self._cumulative`` all refer to the same
        # episode file.
        self.files: List[Path] = []
        self._t_per_file: List[int] = []
        self._cumulative: List[int] = []  # exclusive suffix-sum offsets
        running = 0
        for fp in raw_files:
            try:
                with np.load(str(fp), allow_pickle=False, mmap_mode="r") as data:
                    if "actions" not in data.files:
                        raise KeyError(
                            f"npz {fp} missing required key 'actions'"
                        )
                    T = int(data["actions"].shape[0])
            except Exception as exc:
                raise RuntimeError(
                    f"[NpzTransitionPool] failed to scan {fp}: {exc}"
                ) from exc
            if T <= 0:
                logger.warning(
                    "[NpzTransitionPool] skipping empty episode npz: %s", fp
                )
                continue
            self.files.append(fp)
            self._t_per_file.append(T)
            running += T
            self._cumulative.append(running)
        if running == 0:
            raise RuntimeError(
                f"[NpzTransitionPool] all npz files under {self.npz_dir} are empty"
            )
        self.total = running

        # Probe the obs schema from the first non-empty file.
        with np.load(str(self.files[0]), allow_pickle=False, mmap_mode="r") as data:
            keys_in_file = set(data.files)
            obs_dict_keys = sorted(
                k[len("obs_"):] for k in keys_in_file if k.startswith("obs_")
            )
            self._obs_is_dict = bool(obs_dict_keys)
            if self._obs_is_dict:
                if obs_keys is None:
                    self.obs_keys: List[str] = obs_dict_keys
                else:
                    missing = [k for k in obs_keys if f"obs_{k}" not in keys_in_file]
                    if missing:
                        raise KeyError(
                            f"[NpzTransitionPool] requested obs keys missing in "
                            f"{self.files[0]}: {missing}"
                        )
                    self.obs_keys = list(obs_keys)
            else:
                if "observations" not in keys_in_file:
                    raise KeyError(
                        f"[NpzTransitionPool] {self.files[0]} has neither "
                        f"obs_<key> nor 'observations'"
                    )
                self.obs_keys = []
            self._has_expert_actions = "expert_actions" in keys_in_file

        self._cache_files = bool(cache_files)
        self._handles: List[Optional[np.lib.npyio.NpzFile]] = [None] * len(self.files)
        self._lock = threading.Lock()

        logger.info(
            "[NpzTransitionPool] dir=%s files=%d total_transitions=%d "
            "obs_is_dict=%s expert_actions=%s",
            self.npz_dir, len(self.files), self.total,
            self._obs_is_dict, self._has_expert_actions,
        )

    def _open(self, file_idx: int) -> np.lib.npyio.NpzFile:
        # ``np.load(..., allow_pickle=False)`` returns a context manager and
        # supports lazy access; we keep the handle alive for the pool lifetime
        # to avoid repeated open/close costs.
        if self._cache_files:
            handle = self._handles[file_idx]
            if handle is not None:
                return handle
            handle = np.load(
                str(self.files[file_idx]),
                allow_pickle=False,
                mmap_mode="r",
            )
            self._handles[file_idx] = handle
            return handle
        # Non-cached path returns a fresh handle every call.
        return np.load(str(self.files[file_idx]), allow_pickle=False, mmap_mode="r")

    def close(self) -> None:
        """Close any cached np.load handles (safe to call multiple times)."""
        with self._lock:
            for i, h in enumerate(self._handles):
                if h is None:
                    continue
                try:
                    h.close()
                except Exception:
                    pass
                self._handles[i] = None

    def sample_one(self) -> Dict[str, Any]:
        """Draw one transition uniformly from the pool."""
        global_t = int(np.random.randint(0, self.total))
        return self._materialize(global_t)

    def sample_many(self, n: int) -> List[Dict[str, Any]]:
        """Draw ``n`` transitions (with replacement) — convenience helper."""
        if n <= 0:
            return []
        global_ts = np.random.randint(0, self.total, size=n)
        return [self._materialize(int(g)) for g in global_ts]

    def _materialize(self, global_t: int) -> Dict[str, Any]:
        """Translate a global transition index into a concrete dict."""
        # ``_cumulative[i]`` holds the exclusive end of file ``i``; bisect_right
        # gives the file containing ``global_t``.
        file_idx = bisect.bisect_right(self._cumulative, global_t)
        if file_idx >= len(self.files):  # pragma: no cover (defensive)
            file_idx = len(self.files) - 1
        prev = self._cumulative[file_idx - 1] if file_idx > 0 else 0
        local_t = global_t - prev

        with self._lock:
            data = self._open(file_idx)

        if self._obs_is_dict:
            obs: Dict[str, np.ndarray] = {}
            next_obs: Dict[str, np.ndarray] = {}
            for key in self.obs_keys:
                arr = data[f"obs_{key}"]
                obs[key] = np.asarray(arr[local_t]).copy()
                next_obs[key] = np.asarray(arr[local_t + 1]).copy()
        else:
            arr = data["observations"]
            obs = np.asarray(arr[local_t]).copy()
            next_obs = np.asarray(arr[local_t + 1]).copy()

        action = np.asarray(data["actions"][local_t]).copy()
        reward = float(data["rewards"][local_t])
        terminated = bool(data["terminateds"][local_t])
        truncated = bool(data["truncateds"][local_t])
        if self._has_expert_actions:
            expert_action = np.asarray(data["expert_actions"][local_t]).copy()
        else:
            # Current LQR recordings omit this optional field.
            expert_action = np.zeros_like(action, dtype=np.float32)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "expert_action": expert_action,
        }

    @property
    def is_dict_obs(self) -> bool:
        return self._obs_is_dict

    def __len__(self) -> int:
        return self.total

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NpzTransitionPool(dir={self.npz_dir}, files={len(self.files)}, "
            f"total={self.total}, obs_is_dict={self._obs_is_dict})"
        )
