"""Per-scenario *static* replay buffer backed by ``/dev/shm + mmap``.

Design summary
==============

* **Episode-level admission / eviction, transition-level sampling.**
  Trajectories are flushed in whole by :class:`EpisodeAssembler`; once a
  trajectory passes the B2D-style admission filter it is written into one of
  the per-scenario buffers.  Sampling (performed by the spawned train
  sub-process) is done *per transition* just like the standard replay buffer.

* **Uniform sampling, no PER.**  Static buffers store "golden" samples whose
  value should not be further reweighted by TD error.

* **Eviction policy.**  When the buffer is full:

  - A new ``SUCCESS`` episode evicts the oldest non-``SUCCESS`` episode(s);
    if that is still insufficient it starts evicting the oldest ``SUCCESS``
    episodes (FIFO).
  - A new ``high-completion`` (non-``SUCCESS``) episode can only evict other
    non-``SUCCESS`` episodes; if there is no non-``SUCCESS`` space available
    the new episode is rejected.

* **Shared memory.**  Data arrays live in ``/dev/shm`` and are attached by
  the spawned train process read-only.  A compact ``valid_indices[size]``
  array is maintained in shared memory so that the train process can sample
  without needing to know the owner-side episode metadata.

Only the *owner* process (main/collect) writes to the buffer.  Training
sub-processes call :meth:`StaticReplayBuffer.from_shared` and use the read
only ``sample`` API.
"""

from __future__ import annotations

import logging
import mmap
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

logger = logging.getLogger("Policy")

SHM_DIR = "/dev/shm"


# Kept local to avoid coupling to SharedPrioritizedReplayBuffer internals.
def _shm_create(name: str, nbytes: int) -> Tuple[int, mmap.mmap, str]:
    path = os.path.join(SHM_DIR, name)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, nbytes)
    mm = mmap.mmap(fd, nbytes)
    return fd, mm, path


def _shm_attach(name: str, nbytes: int) -> Tuple[int, mmap.mmap]:
    path = os.path.join(SHM_DIR, name)
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, nbytes)
    return fd, mm


def _create_shared_array(
    name: str, shape: Tuple[int, ...], dtype
) -> Tuple[str, np.ndarray, int, mmap.mmap]:
    dtype = np.dtype(dtype)
    total = int(np.prod(shape))
    nbytes = max(total * dtype.itemsize, 1)
    fd, mm, _ = _shm_create(name, nbytes)
    arr = np.frombuffer(mm, dtype=dtype).reshape(shape)
    arr[:] = 0
    return name, arr, fd, mm


def _view_shared_array(
    name: str, shape: Tuple[int, ...], dtype
) -> Tuple[np.ndarray, int, mmap.mmap]:
    dtype = np.dtype(dtype)
    total = int(np.prod(shape))
    nbytes = max(total * dtype.itemsize, 1)
    fd, mm = _shm_attach(name, nbytes)
    arr = np.frombuffer(mm, dtype=dtype).reshape(shape)
    return arr, fd, mm


def _gpu_unpackbits(packed_tensor: torch.Tensor, original_last_dim: int) -> torch.Tensor:
    """Mirror of ``shared_buffer._gpu_unpackbits`` so we don't introduce a cycle."""
    bits = torch.tensor(
        [128, 64, 32, 16, 8, 4, 2, 1],
        dtype=torch.uint8, device=packed_tensor.device,
    )
    unpacked = (packed_tensor.unsqueeze(-1) & bits).ne(0)
    shape = list(unpacked.shape[:-2]) + [-1]
    unpacked = unpacked.reshape(shape)[..., :original_last_dim]
    return unpacked.float() * 255.0



@dataclass
class _EpisodeMeta:
    episode_id: int
    slot_indices: List[int]
    is_success: bool
    insert_order: int
    route_completed: float = 0.0



class StaticReplayBuffer:
    """Per-scenario shared-memory static replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored for this scenario.
    observation_space / action_space:
        Gym spaces – same format that ``SharedPrioritizedReplayBuffer`` uses.
    scenario_name:
        Used only for logging / shm-file naming.
    use_bitpack:
        Same heuristic as the dynamic buffer – Dict Box(uint8) BEV channels
        get bit-packed to 1/8 the size.
    store_expert_actions:
        Whether to allocate the ``expert_actions`` array.
    """

    # ---- construction ----------------------------------------------------

    def __init__(
        self,
        capacity: int,
        observation_space,
        action_space,
        *,
        scenario_name: str,
        shm_prefix: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        use_bitpack: bool = True,
        store_expert_actions: bool = False,
        store_expert_dist: bool = False,
        reserved_size: int = 0,
        npz_pool: Optional[Any] = None,
        npz_dir: Optional[str] = None,
    ):
        self._is_owner = True
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.observation_space = observation_space
        self.action_space = action_space
        self.scenario_name = scenario_name
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.use_bitpack = use_bitpack
        self.store_expert_actions = store_expert_actions
        self.store_expert_dist = store_expert_dist
        self.reserved_size = int(reserved_size)
        if self.reserved_size < 0:
            raise ValueError("reserved_size must be >= 0")
        if self.reserved_size > self.capacity:
            raise ValueError(
                f"reserved_size ({self.reserved_size}) > capacity ({self.capacity})"
            )
        if self.reserved_size > 0 and npz_pool is None:
            raise ValueError(
                "reserved_size > 0 but no npz_pool was provided"
            )
        self._npz_pool = npz_pool
        # Stash the source dir so the train-process view can build its own
        # ``NpzTransitionPool`` to perform refills without IPC.
        self._npz_dir: Optional[str] = str(npz_dir) if npz_dir else None

        # A deterministic-ish hash of the scenario name keeps /dev/shm file
        # names stable within a run while still being unique across scenarios.
        self._shm_prefix = shm_prefix or "rl_sbuf_{}_{}".format(
            _safe_tag(scenario_name), uuid.uuid4().hex[:8],
        )

        self._obs_is_dict = isinstance(observation_space, spaces.Dict)
        self.obs_keys: List[str] = []
        self.obs_shapes: Dict[str, tuple] = {}
        self._packed_keys: set = set()
        self._packed_shapes: Dict[str, tuple] = {}
        self.obs_shape = self._get_obs_shape()
        self.action_dim = self._get_action_dim()

        self._shm_meta: Dict[str, Tuple[str, tuple, str]] = {}
        self._fds: List[int] = []
        self._mms: List[mmap.mmap] = []

        self._allocate_arrays()

        # Owner-only episode metadata table.
        self._episode_table: "OrderedDict[int, _EpisodeMeta]" = OrderedDict()
        self._next_episode_id = 1
        self._next_insert_order = 1
        # Reserved slots [0, reserved_size) are *never* free for episode
        # admission — they are pre-filled from the npz pool by the owner and
        # subsequently refreshed in-place by the view (train sub-process).
        # The free-slot stack therefore only contains the non-reserved range
        # [reserved_size, capacity).
        self._free_slots: List[int] = list(
            range(self.capacity - 1, self.reserved_size - 1, -1)
        )

        # Owner-side: prefill the reserved area immediately.
        if self.reserved_size > 0:
            self._prefill_reserved_slots()

        logger.info(
            "[StaticReplayBuffer] scenario=%s capacity=%d reserved=%d "
            "prefix=%s bitpack=%s",
            scenario_name, self.capacity, self.reserved_size,
            self._shm_prefix, bool(self._packed_keys),
        )

    # ---- observation-space helpers (self-contained to avoid coupling) ----

    def _get_obs_shape(self):
        if self._obs_is_dict:
            self.obs_keys = [
                key for key, space in self.observation_space.spaces.items()
                if isinstance(space, spaces.Box)
            ]
            if not self.obs_keys:
                raise ValueError("No Box space found in Dict observation space")
            self.obs_shapes = {
                key: self.observation_space.spaces[key].shape for key in self.obs_keys
            }
            return self.obs_shapes[self.obs_keys[0]]
        return self.observation_space.shape

    def _get_action_dim(self) -> int:
        if isinstance(self.action_space, spaces.Discrete):
            return 1
        return int(np.prod(self.action_space.shape))

    def _get_action_dtype(self):
        if hasattr(self.action_space, "dtype"):
            dt = self.action_space.dtype
            return np.float32 if dt == np.float64 else dt
        return np.float32

    def _should_use_bitpack(self, space, key) -> bool:
        if not self.use_bitpack:
            return False
        if not isinstance(space, spaces.Box):
            return False
        if not hasattr(space, "dtype") or space.dtype != np.uint8:
            return False
        if len(space.shape) != 3:
            return False
        if key is not None:
            key_lower = key.lower()
            if any(hint in key_lower for hint in ("vector", "bev", "mask", "binary")):
                return True
        if space.shape[1] == space.shape[2]:
            return True
        return False

    @staticmethod
    def _get_packed_shape(shape):
        leading = list(shape[:-1])
        last = shape[-1]
        packed_last = (last + 7) // 8
        return tuple(leading) + (packed_last,)

    # ---- shared-memory allocation ----------------------------------------

    def _allocate_arrays(self) -> None:
        prefix = self._shm_prefix
        cap = self.capacity

        if self._obs_is_dict:
            self.observations: Dict[str, np.ndarray] = {}
            self.next_observations: Dict[str, np.ndarray] = {}
            for key in self.obs_keys:
                space = self.observation_space.spaces[key]
                obs_dtype = space.dtype if hasattr(space, "dtype") else np.float32
                shape = self.obs_shapes[key]
                if self._should_use_bitpack(space, key):
                    self._packed_keys.add(key)
                    self._packed_shapes[key] = shape
                    shape = self._get_packed_shape(shape)
                    obs_dtype = np.uint8
                full_shape = (cap,) + tuple(shape)
                obs_name = "{}_{}_obs".format(prefix, _safe_tag(key))
                nobs_name = "{}_{}_nobs".format(prefix, _safe_tag(key))
                _, self.observations[key], fd1, mm1 = _create_shared_array(obs_name, full_shape, obs_dtype)
                _, self.next_observations[key], fd2, mm2 = _create_shared_array(nobs_name, full_shape, obs_dtype)
                self._fds.extend([fd1, fd2])
                self._mms.extend([mm1, mm2])
                self._shm_meta["obs_" + key] = (obs_name, full_shape, np.dtype(obs_dtype).str)
                self._shm_meta["nobs_" + key] = (nobs_name, full_shape, np.dtype(obs_dtype).str)
        else:
            obs_dtype = self.observation_space.dtype if hasattr(self.observation_space, "dtype") else np.float32
            shape = self.obs_shape
            if self._should_use_bitpack(self.observation_space, None):
                self._packed_keys.add("__single__")
                self._packed_shapes["__single__"] = shape
                shape = self._get_packed_shape(shape)
                obs_dtype = np.uint8
            full_shape = (cap,) + tuple(shape)
            obs_name = "{}_obs".format(prefix)
            nobs_name = "{}_nobs".format(prefix)
            _, self.observations, fd1, mm1 = _create_shared_array(obs_name, full_shape, obs_dtype)
            _, self.next_observations, fd2, mm2 = _create_shared_array(nobs_name, full_shape, obs_dtype)
            self._fds.extend([fd1, fd2])
            self._mms.extend([mm1, mm2])
            self._shm_meta["obs"] = (obs_name, full_shape, np.dtype(obs_dtype).str)
            self._shm_meta["nobs"] = (nobs_name, full_shape, np.dtype(obs_dtype).str)

        action_dtype = self._get_action_dtype()
        act_name = "{}_act".format(prefix)
        rew_name = "{}_rew".format(prefix)
        term_name = "{}_term".format(prefix)
        trunc_name = "{}_trunc".format(prefix)
        _, self.actions, fd_a, mm_a = _create_shared_array(act_name, (cap, self.action_dim), action_dtype)
        _, self.rewards, fd_r, mm_r = _create_shared_array(rew_name, (cap,), np.float32)
        _, self.terminateds, fd_t, mm_t = _create_shared_array(term_name, (cap,), np.float32)
        _, self.truncateds, fd_tr, mm_tr = _create_shared_array(trunc_name, (cap,), np.float32)
        self._fds.extend([fd_a, fd_r, fd_t, fd_tr])
        self._mms.extend([mm_a, mm_r, mm_t, mm_tr])
        self._shm_meta["act"] = (act_name, (cap, self.action_dim), np.dtype(action_dtype).str)
        self._shm_meta["rew"] = (rew_name, (cap,), np.dtype(np.float32).str)
        self._shm_meta["term"] = (term_name, (cap,), np.dtype(np.float32).str)
        self._shm_meta["trunc"] = (trunc_name, (cap,), np.dtype(np.float32).str)

        if self.store_expert_actions:
            eact_name = "{}_eact".format(prefix)
            _, self.expert_actions, fd_ea, mm_ea = _create_shared_array(
                eact_name, (cap, self.action_dim), np.float32,
            )
            self._fds.append(fd_ea)
            self._mms.append(mm_ea)
            self._shm_meta["eact"] = (eact_name, (cap, self.action_dim), np.dtype(np.float32).str)
        else:
            self.expert_actions = None

        if self.store_expert_dist:
            emean_name = "{}_emean".format(prefix)
            elogstd_name = "{}_elogstd".format(prefix)
            emask_name = "{}_emask".format(prefix)
            _, self.expert_mean, fd_em, mm_em = _create_shared_array(
                emean_name, (cap, self.action_dim), np.float32,
            )
            _, self.expert_log_std, fd_el, mm_el = _create_shared_array(
                elogstd_name, (cap, self.action_dim), np.float32,
            )
            _, self.expert_dist_mask, fd_emk, mm_emk = _create_shared_array(
                emask_name, (cap,), np.uint8,
            )
            self._fds.extend([fd_em, fd_el, fd_emk])
            self._mms.extend([mm_em, mm_el, mm_emk])
            self._shm_meta["emean"] = (
                emean_name, (cap, self.action_dim), np.dtype(np.float32).str,
            )
            self._shm_meta["elogstd"] = (
                elogstd_name, (cap, self.action_dim), np.dtype(np.float32).str,
            )
            self._shm_meta["emask"] = (
                emask_name, (cap,), np.dtype(np.uint8).str,
            )
        else:
            self.expert_mean = None
            self.expert_log_std = None
            self.expert_dist_mask = None

        # Index / size table (the train process only needs these + the data
        # arrays to sample).
        idx_name = "{}_idx".format(prefix)
        size_name = "{}_size".format(prefix)
        is_succ_name = "{}_issucc".format(prefix)
        is_resv_name = "{}_isresv".format(prefix)
        _, self._valid_indices_arr, fd_idx, mm_idx = _create_shared_array(idx_name, (cap,), np.int32)
        _, self._size_arr, fd_sz, mm_sz = _create_shared_array(size_name, (1,), np.int32)
        _, self._slot_is_success_arr, fd_s, mm_s = _create_shared_array(is_succ_name, (cap,), np.uint8)
        # ``_slot_is_reserved_arr[i] == 1`` iff slot i belongs to the npz-backed
        # reserved area.  Used by the view (train sub-process) to identify
        # which slots in a sampled batch must be refilled in-place.
        _, self._slot_is_reserved_arr, fd_r, mm_r = _create_shared_array(
            is_resv_name, (cap,), np.uint8,
        )
        self._fds.extend([fd_idx, fd_sz, fd_s, fd_r])
        self._mms.extend([mm_idx, mm_sz, mm_s, mm_r])
        self._shm_meta["idx"] = (idx_name, (cap,), np.dtype(np.int32).str)
        self._shm_meta["size"] = (size_name, (1,), np.dtype(np.int32).str)
        self._shm_meta["issucc"] = (is_succ_name, (cap,), np.dtype(np.uint8).str)
        self._shm_meta["isresv"] = (is_resv_name, (cap,), np.dtype(np.uint8).str)

    # ---- view (train process) --------------------------------------------

    def get_shared_config(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "scenario_name": self.scenario_name,
            "use_bitpack": self.use_bitpack,
            "store_expert_actions": self.store_expert_actions,
            "store_expert_dist": self.store_expert_dist,
            "obs_is_dict": self._obs_is_dict,
            "obs_keys": list(self.obs_keys),
            "obs_shapes": dict(self.obs_shapes),
            "obs_shape": self.obs_shape,
            "action_dim": self.action_dim,
            "packed_keys": list(self._packed_keys),
            "packed_shapes": dict(self._packed_shapes),
            "shm_meta": self._shm_meta,
            "shm_prefix": self._shm_prefix,
            "reserved_size": int(self.reserved_size),
            "npz_dir": self._npz_dir,
        }

    @classmethod
    def from_shared(cls, config: Dict[str, Any], device="cpu") -> "StaticReplayBuffer":
        obj = cls.__new__(cls)
        obj._is_owner = False
        obj.capacity = config["capacity"]
        obj.observation_space = config["observation_space"]
        obj.action_space = config["action_space"]
        obj.scenario_name = config["scenario_name"]
        obj.device = device if isinstance(device, torch.device) else torch.device(device)
        obj.use_bitpack = config["use_bitpack"]
        obj.store_expert_actions = config.get("store_expert_actions", False)
        obj.store_expert_dist = config.get("store_expert_dist", False)
        obj._obs_is_dict = config["obs_is_dict"]
        obj.obs_keys = list(config["obs_keys"])
        obj.obs_shapes = dict(config["obs_shapes"])
        obj.obs_shape = config["obs_shape"]
        obj.action_dim = config["action_dim"]
        obj._packed_keys = set(config["packed_keys"])
        obj._packed_shapes = dict(config["packed_shapes"])
        obj._shm_prefix = config["shm_prefix"]
        obj.reserved_size = int(config.get("reserved_size", 0))
        obj._npz_dir = config.get("npz_dir")
        obj._npz_pool = None  # the caller may attach via attach_view_npz_pool
        shm_meta = config["shm_meta"]
        obj._shm_meta = dict(shm_meta)
        obj._fds = []
        obj._mms = []

        if obj._obs_is_dict:
            obj.observations = {}
            obj.next_observations = {}
            for key in obj.obs_keys:
                obs_name, shape, dtype_str = shm_meta["obs_" + key]
                nobs_name, nshape, ndtype_str = shm_meta["nobs_" + key]
                obs_arr, fd1, mm1 = _view_shared_array(obs_name, tuple(shape), dtype_str)
                nobs_arr, fd2, mm2 = _view_shared_array(nobs_name, tuple(nshape), ndtype_str)
                obj.observations[key] = obs_arr
                obj.next_observations[key] = nobs_arr
                obj._fds.extend([fd1, fd2])
                obj._mms.extend([mm1, mm2])
        else:
            obs_name, shape, dtype_str = shm_meta["obs"]
            nobs_name, nshape, ndtype_str = shm_meta["nobs"]
            obj.observations, fd1, mm1 = _view_shared_array(obs_name, tuple(shape), dtype_str)
            obj.next_observations, fd2, mm2 = _view_shared_array(nobs_name, tuple(nshape), ndtype_str)
            obj._fds.extend([fd1, fd2])
            obj._mms.extend([mm1, mm2])

        act_name, act_shape, act_dtype = shm_meta["act"]
        rew_name, rew_shape, rew_dtype = shm_meta["rew"]
        term_name, term_shape, term_dtype = shm_meta["term"]
        trunc_name, trunc_shape, trunc_dtype = shm_meta["trunc"]
        obj.actions, fd_a, mm_a = _view_shared_array(act_name, tuple(act_shape), act_dtype)
        obj.rewards, fd_r, mm_r = _view_shared_array(rew_name, tuple(rew_shape), rew_dtype)
        obj.terminateds, fd_t, mm_t = _view_shared_array(term_name, tuple(term_shape), term_dtype)
        obj.truncateds, fd_tr, mm_tr = _view_shared_array(trunc_name, tuple(trunc_shape), trunc_dtype)
        obj._fds.extend([fd_a, fd_r, fd_t, fd_tr])
        obj._mms.extend([mm_a, mm_r, mm_t, mm_tr])

        if obj.store_expert_actions and "eact" in shm_meta:
            eact_name, eact_shape, eact_dtype = shm_meta["eact"]
            obj.expert_actions, fd_ea, mm_ea = _view_shared_array(
                eact_name, tuple(eact_shape), eact_dtype,
            )
            obj._fds.append(fd_ea)
            obj._mms.append(mm_ea)
        else:
            obj.expert_actions = None

        if (
            obj.store_expert_dist
            and "emean" in shm_meta
            and "elogstd" in shm_meta
            and "emask" in shm_meta
        ):
            emean_name, emean_shape, emean_dtype = shm_meta["emean"]
            elogstd_name, elogstd_shape, elogstd_dtype = shm_meta["elogstd"]
            emask_name, emask_shape, emask_dtype = shm_meta["emask"]
            obj.expert_mean, fd_em, mm_em = _view_shared_array(
                emean_name, tuple(emean_shape), emean_dtype,
            )
            obj.expert_log_std, fd_el, mm_el = _view_shared_array(
                elogstd_name, tuple(elogstd_shape), elogstd_dtype,
            )
            obj.expert_dist_mask, fd_emk, mm_emk = _view_shared_array(
                emask_name, tuple(emask_shape), emask_dtype,
            )
            obj._fds.extend([fd_em, fd_el, fd_emk])
            obj._mms.extend([mm_em, mm_el, mm_emk])
        else:
            obj.expert_mean = None
            obj.expert_log_std = None
            obj.expert_dist_mask = None

        idx_name, idx_shape, idx_dtype = shm_meta["idx"]
        size_name, size_shape, size_dtype = shm_meta["size"]
        is_succ_name, is_succ_shape, is_succ_dtype = shm_meta["issucc"]
        obj._valid_indices_arr, fd_idx, mm_idx = _view_shared_array(idx_name, tuple(idx_shape), idx_dtype)
        obj._size_arr, fd_sz, mm_sz = _view_shared_array(size_name, tuple(size_shape), size_dtype)
        obj._slot_is_success_arr, fd_s, mm_s = _view_shared_array(is_succ_name, tuple(is_succ_shape), is_succ_dtype)
        obj._fds.extend([fd_idx, fd_sz, fd_s])
        obj._mms.extend([mm_idx, mm_sz, mm_s])

        # ``isresv`` is optional for backward compatibility — if absent the
        # buffer was created before the reserved-area feature.
        if "isresv" in shm_meta:
            is_resv_name, is_resv_shape, is_resv_dtype = shm_meta["isresv"]
            obj._slot_is_reserved_arr, fd_r, mm_r = _view_shared_array(
                is_resv_name, tuple(is_resv_shape), is_resv_dtype,
            )
            obj._fds.append(fd_r)
            obj._mms.append(mm_r)
        else:
            obj._slot_is_reserved_arr = np.zeros((obj.capacity,), dtype=np.uint8)

        return obj

    # ---- introspection ---------------------------------------------------

    def size(self) -> int:
        return int(self._size_arr[0])

    def num_success_entries(self) -> int:
        # Count only slots that are currently valid AND belong to a success
        # episode.  We use the compact valid-index array to keep this cheap.
        size = self.size()
        if size == 0:
            return 0
        valid_idx = self._valid_indices_arr[:size]
        return int(self._slot_is_success_arr[valid_idx].sum())

    def num_non_success_entries(self) -> int:
        return self.size() - self.num_success_entries()

    def num_episodes(self) -> int:
        return len(self._episode_table) if self._is_owner else -1

    # ---- eviction + insertion (owner-only) -------------------------------

    def _evict_episode(self, episode_id: int) -> None:
        meta = self._episode_table.pop(episode_id, None)
        if meta is None:
            return
        size = int(self._size_arr[0])
        valid = self._valid_indices_arr
        slot_is_success = self._slot_is_success_arr

        # Pre-build a fast set for O(1) membership test.
        evict_set = set(meta.slot_indices)

        # Two-pointer compaction: keep indices that are not in evict_set.
        write = 0
        for read in range(size):
            idx = int(valid[read])
            if idx in evict_set:
                # Mark slot free.
                slot_is_success[idx] = 0
                self._free_slots.append(idx)
            else:
                if write != read:
                    valid[write] = idx
                write += 1
        self._size_arr[0] = write

    def _plan_evictions(
        self, need: int, *, allow_success_eviction: bool,
    ) -> Optional[List[int]]:
        """Pick episode ids whose removal frees at least ``need`` slots.

        Returns ``None`` if insufficient capacity can be reclaimed without
        breaking the caller's policy.
        """
        freed = 0
        plan: List[int] = []

        # Iterate in ascending insert_order (OrderedDict preserves insertion
        # order which equals insert_order for our use).
        for eid, meta in list(self._episode_table.items()):
            if meta.is_success:
                continue
            plan.append(eid)
            freed += len(meta.slot_indices)
            if freed >= need:
                return plan

        if not allow_success_eviction:
            return None

        for eid, meta in list(self._episode_table.items()):
            if not meta.is_success:
                continue
            plan.append(eid)
            freed += len(meta.slot_indices)
            if freed >= need:
                return plan

        return None if freed < need else plan

    def try_add_episode(
        self,
        episode_payload: List[Dict[str, Any]],
        *,
        is_success: bool,
        route_completed: float = 0.0,
    ) -> bool:
        """Admit a full episode into the buffer.

        Parameters
        ----------
        episode_payload:
            Each element is a dict with keys ``obs``, ``next_obs``, ``action``,
            ``reward``, ``terminated``, ``truncated``, and optionally
            ``expert_action``.  Observations follow the same convention as
            :class:`SharedPrioritizedReplayBuffer.add` (raw, not bit-packed).
        is_success:
            Whether the trajectory passes the strict B2D-success filter.
        route_completed:
            Completion percentage (0..100), stored for diagnostics.

        Returns
        -------
        bool
            True if the episode was accepted; False otherwise.
        """
        if not episode_payload:
            return False
        T = len(episode_payload)
        if T > self.capacity:
            return False

        free_now = self.capacity - self.size()
        need = T - free_now
        if need > 0:
            plan = self._plan_evictions(
                need=need,
                allow_success_eviction=is_success,
            )
            if plan is None:
                return False
            for eid in plan:
                self._evict_episode(eid)

        # Allocate slots.
        slot_indices: List[int] = []
        for tr in episode_payload:
            if not self._free_slots:
                # Should not happen because we ensured free_now >= T after
                # eviction.  Guard against programming errors.
                raise RuntimeError(
                    "[StaticReplayBuffer] out of free slots during insert"
                )
            idx = self._free_slots.pop()
            self._write_slot(idx, tr)
            self._slot_is_success_arr[idx] = 1 if is_success else 0
            slot_indices.append(idx)

        # Publish new valid indices into the compact table.
        cur = int(self._size_arr[0])
        for offset, idx in enumerate(slot_indices):
            self._valid_indices_arr[cur + offset] = idx
        self._size_arr[0] = cur + len(slot_indices)

        # Record episode metadata.
        episode_id = self._next_episode_id
        self._next_episode_id += 1
        insert_order = self._next_insert_order
        self._next_insert_order += 1
        self._episode_table[episode_id] = _EpisodeMeta(
            episode_id=episode_id,
            slot_indices=slot_indices,
            is_success=bool(is_success),
            insert_order=insert_order,
            route_completed=float(route_completed),
        )
        return True

    # ---- reserved (npz-backed) area --------------------------------------

    def _prefill_reserved_slots(self) -> None:
        """Owner-side: fill ``[0, reserved_size)`` from the npz pool.

        Each reserved slot becomes immediately visible to samplers via
        ``valid_indices`` and is marked in ``_slot_is_reserved_arr`` so the
        view side can refresh it after consumption.  Reserved slots are
        deliberately excluded from ``_free_slots`` and ``_episode_table`` so
        the existing eviction logic never touches them.
        """
        if self.reserved_size <= 0 or self._npz_pool is None:
            return
        for slot in range(self.reserved_size):
            tr = self._npz_pool.sample_one()
            self._write_slot(slot, tr)
            # Reserved slots are explicitly NOT considered ``success`` so that
            # eviction's success/non-success accounting is unaffected (they
            # cannot be evicted regardless because they are not in
            # ``_episode_table``, but we keep the bookkeeping clean).
            self._slot_is_success_arr[slot] = 0
            self._slot_is_reserved_arr[slot] = 1
            self._valid_indices_arr[slot] = slot
        self._size_arr[0] = self.reserved_size
        logger.info(
            "[StaticReplayBuffer] scenario=%s prefilled %d reserved slots "
            "from npz pool (%d transitions available)",
            self.scenario_name, self.reserved_size, len(self._npz_pool),
        )

    def attach_view_npz_pool(self, npz_pool) -> None:
        """View-side hook: register a ``NpzTransitionPool`` so this buffer can
        refill its reserved slots after they are consumed by ``sample()``.

        Called by ``MixedReplayBuffer.from_shared`` (or any equivalent caller
        on the view side) once shared memory has been attached.  Owner-side
        instances also keep their pool around for diagnostics; calling this
        on the owner is a no-op when a pool is already set.
        """
        if npz_pool is None:
            return
        if self._npz_pool is None:
            self._npz_pool = npz_pool

    def refresh_reserved_slot(self, slot: int) -> bool:
        """Replace the contents of a reserved slot with a fresh sample drawn
        uniformly from the attached npz pool.

        Returns True iff a refresh happened.  Safe to call on slots that are
        not reserved (no-op).
        """
        if self._npz_pool is None:
            return False
        if not (0 <= slot < self.capacity):
            return False
        if self._slot_is_reserved_arr[slot] == 0:
            return False
        tr = self._npz_pool.sample_one()
        self._write_slot(int(slot), tr)
        return True

    def refresh_reserved_in_indices(self, indices) -> int:
        """Refresh every reserved slot present in ``indices`` (view-side).

        ``indices`` is the array returned by ``sample_raw_indices`` for this
        buffer.  Returns the number of slots that were actually refreshed.
        """
        if self._npz_pool is None or self.reserved_size <= 0:
            return 0
        idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx_arr.size == 0:
            return 0
        # Vectorised reserved-mask check.
        is_resv = self._slot_is_reserved_arr[idx_arr] != 0
        if not np.any(is_resv):
            return 0
        n = 0
        for slot in idx_arr[is_resv].tolist():
            if self.refresh_reserved_slot(int(slot)):
                n += 1
        return n

    # ---- single-slot write ------------------------------------------------

    def _write_slot(self, slot: int, tr: Dict[str, Any]) -> None:
        obs = tr["obs"]
        next_obs = tr["next_obs"]
        action = np.asarray(tr["action"])
        reward = float(tr["reward"])
        terminated = bool(tr["terminated"])
        truncated = bool(tr.get("truncated", False))
        expert_action = tr.get("expert_action")

        if self._obs_is_dict:
            for key in self.obs_keys:
                obs_data = np.asarray(obs[key])
                next_obs_data = np.asarray(next_obs[key])
                if key in self._packed_keys:
                    obs_data = _pack_obs(obs_data)
                    next_obs_data = _pack_obs(next_obs_data)
                self.observations[key][slot] = obs_data
                self.next_observations[key][slot] = next_obs_data
        else:
            obs_arr = np.asarray(obs)
            next_obs_arr = np.asarray(next_obs)
            if "__single__" in self._packed_keys:
                obs_arr = _pack_obs(obs_arr)
                next_obs_arr = _pack_obs(next_obs_arr)
            self.observations[slot] = obs_arr
            self.next_observations[slot] = next_obs_arr

        if action.ndim == 1 and self.action_dim > 1:
            action = action.reshape(self.action_dim)
        elif action.ndim == 0:
            action = action.reshape(1)
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.terminateds[slot] = float(terminated)
        self.truncateds[slot] = float(truncated)
        if self.store_expert_actions and self.expert_actions is not None:
            if expert_action is not None:
                ea = np.asarray(expert_action, dtype=np.float32)
                self.expert_actions[slot] = ea.flatten()
            else:
                self.expert_actions[slot] = 0.0

        if self.store_expert_dist and self.expert_mean is not None:
            expert_mean = tr.get("expert_mean")
            expert_log_std = tr.get("expert_log_std")
            if expert_mean is not None and expert_log_std is not None:
                em = np.asarray(expert_mean, dtype=np.float32).flatten()
                el = np.asarray(expert_log_std, dtype=np.float32).flatten()
                self.expert_mean[slot] = em
                self.expert_log_std[slot] = el
                self.expert_dist_mask[slot] = 1
            else:
                self.expert_mean[slot] = 0.0
                self.expert_log_std[slot] = 0.0
                # Reserved (npz-prefilled) and unmapped-scenario rows fall
                # through here -> mask=0 -> KL term is 0 for these samples.
                self.expert_dist_mask[slot] = 0

    # ---- sampling (view process) -----------------------------------------

    def sample_raw_indices(self, batch_size: int) -> Optional[np.ndarray]:
        """Return ``batch_size`` slot indices sampled uniformly w/o replacement
        when possible.  Returns ``None`` if the buffer is empty.
        """
        size = int(self._size_arr[0])
        if size <= 0:
            return None
        valid = self._valid_indices_arr[:size]
        if batch_size >= size:
            # Sample with replacement when batch_size exceeds size.
            return np.random.choice(valid, size=batch_size, replace=True)
        return np.random.choice(valid, size=batch_size, replace=False)

    def gather(self, indices: np.ndarray) -> Dict[str, Any]:
        """Collect a batch of transitions at the given slot indices.

        Observations are returned as **GPU tensors** if they were bit-packed
        (matching ``SharedPrioritizedReplayBuffer`` behaviour).  Otherwise the
        caller is responsible for casting.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if self._obs_is_dict:
            obs: Dict[str, Any] = {}
            next_obs: Dict[str, Any] = {}
            for key in self.obs_keys:
                obs_data = self.observations[key][indices]
                next_obs_data = self.next_observations[key][indices]
                if key in self._packed_keys:
                    obs[key] = self._packed_to_gpu(obs_data, key)
                    next_obs[key] = self._packed_to_gpu(next_obs_data, key)
                else:
                    obs[key] = obs_data
                    next_obs[key] = next_obs_data
        else:
            obs = self.observations[indices]
            next_obs = self.next_observations[indices]
            if "__single__" in self._packed_keys:
                obs = self._packed_to_gpu(obs, "__single__")
                next_obs = self._packed_to_gpu(next_obs, "__single__")

        payload = {
            "observations": obs,
            "next_observations": next_obs,
            "actions": self.actions[indices],
            "rewards": self.rewards[indices].reshape(-1, 1),
            "terminateds": self.terminateds[indices].reshape(-1, 1),
            "truncateds": self.truncateds[indices].reshape(-1, 1),
        }
        if self.store_expert_actions and self.expert_actions is not None:
            payload["expert_actions"] = self.expert_actions[indices]
        else:
            payload["expert_actions"] = None

        if self.store_expert_dist and self.expert_mean is not None:
            payload["expert_mean"] = self.expert_mean[indices]
            payload["expert_log_std"] = self.expert_log_std[indices]
            payload["expert_dist_mask"] = self.expert_dist_mask[indices]
        else:
            payload["expert_mean"] = None
            payload["expert_log_std"] = None
            payload["expert_dist_mask"] = None
        return payload

    def _packed_to_gpu(self, packed_arr: np.ndarray, key: str) -> torch.Tensor:
        original_shape = self._packed_shapes.get(key)
        if original_shape is None:
            raise ValueError("No packed shape for key: {}".format(key))
        original_last = original_shape[-1]
        packed_np = np.ascontiguousarray(packed_arr)
        t = torch.from_numpy(packed_np.copy()).to(self.device, non_blocking=True)
        return _gpu_unpackbits(t, original_last)

    # ---- cleanup ----------------------------------------------------------

    def cleanup(self) -> None:
        if not getattr(self, "_is_owner", False):
            for mm in getattr(self, "_mms", []):
                try:
                    mm.close()
                except Exception:
                    pass
            for fd in getattr(self, "_fds", []):
                try:
                    os.close(fd)
                except Exception:
                    pass
            return

        for mm in getattr(self, "_mms", []):
            try:
                mm.close()
            except Exception:
                pass
        for fd in getattr(self, "_fds", []):
            try:
                os.close(fd)
            except Exception:
                pass

        for _key, (name, _shape, _dtype) in getattr(self, "_shm_meta", {}).items():
            path = os.path.join(SHM_DIR, name)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass



def _pack_obs(obs: np.ndarray) -> np.ndarray:
    if obs.max() > 1:
        obs = (obs > 127).astype(np.uint8)
    else:
        obs = obs.astype(np.uint8)
    return np.packbits(obs, axis=-1)


def _safe_tag(name: str) -> str:
    # Keep /dev/shm file names path-safe.
    import re
    s = re.sub(r"[^0-9A-Za-z_-]+", "_", str(name).strip())
    s = s.strip("_-")
    return s or "unknown"
