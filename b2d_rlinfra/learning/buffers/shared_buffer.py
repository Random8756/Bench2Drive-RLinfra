"""Shared-memory prioritized replay buffer for off-policy training.

Stores numpy arrays in ``/dev/shm``-backed ``mmap`` files so spawned training
processes can attach to the same replay data.
"""

import logging
import os
import mmap
import uuid
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

logger = logging.getLogger("Policy")

__layer__ = (4, "Algorithm")


class PrioritizedReplayBufferSamples(NamedTuple):
    """Named tuple for PER samples (with IS weights and tree indices).

    The trailing expert-distribution fields are only populated by buffers
    that store per-sample expert policy parameters (e.g. the static tier of
    :class:`~b2d_rlinfra.learning.buffers.mixed_buffer.MixedReplayBuffer`); they default
    to ``None`` for the plain shared PER buffer.
    """
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    next_observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray
    expert_actions: Optional[torch.Tensor] = None
    expert_mean: Optional[torch.Tensor] = None
    expert_log_std: Optional[torch.Tensor] = None
    expert_dist_mask: Optional[torch.Tensor] = None

SHM_DIR = '/dev/shm'


# GPU unpack avoids transferring expanded observations from the CPU.
def _gpu_unpackbits(packed_tensor, original_last_dim):
    """Unpack bitpacked uint8 tensor on GPU.

    Replaces the CPU path with a single small PCIe transfer and fast GPU bit
    ops.

    Args:
        packed_tensor: (..., packed_dim) uint8 tensor **already on GPU**.
                       packed_dim = ceil(original_last_dim / 8).
        original_last_dim: original size of last dimension before packing.

    Returns:
        (..., original_last_dim) float32 tensor with values 0.0 or 255.0.
    """
    # Big-endian bit masks, matching np.packbits default bit-order
    bits = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                        dtype=torch.uint8, device=packed_tensor.device)
    # (..., P, 1) & (8,) -> (..., P, 8): broadcast bit extraction
    unpacked = (packed_tensor.unsqueeze(-1) & bits).ne(0)
    # Flatten last two dims: (..., P*8), then trim to original size
    shape = list(unpacked.shape[:-2]) + [-1]
    unpacked = unpacked.reshape(shape)[..., :original_last_dim]
    # Return float32 0/255 (matching the old CPU path output)
    return unpacked.float() * 255.0


def _shm_create(name, nbytes):
    """Create a /dev/shm/<name> file of the given size and return (fd, mm)."""
    path = os.path.join(SHM_DIR, name)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, nbytes)
    mm = mmap.mmap(fd, nbytes)
    return fd, mm, path


def _shm_attach(name, nbytes):
    """Open an existing /dev/shm/<name> file and return (fd, mm)."""
    path = os.path.join(SHM_DIR, name)
    fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, nbytes)
    return fd, mm


def _create_shared_array(name, shape, dtype):
    """
    Create a numpy array backed by a /dev/shm file.

    Returns (shm_name, numpy_array, fd, mm).
    The caller must keep fd/mm alive for the lifetime of the array.
    """
    dtype = np.dtype(dtype)
    total = int(np.prod(shape))
    nbytes = total * dtype.itemsize
    fd, mm, path = _shm_create(name, nbytes)
    arr = np.frombuffer(mm, dtype=dtype).reshape(shape)
    arr[:] = 0
    return name, arr, fd, mm


def _view_shared_array(name, shape, dtype):
    """
    Attach a numpy view to an existing /dev/shm file (in child process).

    Returns (numpy_array, fd, mm).
    """
    dtype = np.dtype(dtype)
    total = int(np.prod(shape))
    nbytes = total * dtype.itemsize
    fd, mm = _shm_attach(name, nbytes)
    arr = np.frombuffer(mm, dtype=dtype).reshape(shape)
    return arr, fd, mm


class SharedSumTree:
    """SumTree with tree array in shared memory (/dev/shm + mmap)."""

    def __init__(self, capacity, *, shm_name=None, create=True):
        """
        Args:
            capacity: Number of leaf nodes.
            shm_name: Name of /dev/shm file. If create=True and shm_name is
                      None, a random name is generated.
            create: If True, create the shm file; if False, attach to existing.
        """
        self.capacity = capacity
        self._depth = int(np.ceil(np.log2(max(capacity, 2)))) + 1
        tree_size = 2 * capacity - 1

        if create:
            self._shm_name = shm_name or 'rl_sumtree_{}'.format(uuid.uuid4().hex[:12])
            self._shm_name, self.tree, self._fd, self._mm = _create_shared_array(
                self._shm_name, (tree_size,), np.float64
            )
        else:
            assert shm_name is not None
            self._shm_name = shm_name
            self.tree, self._fd, self._mm = _view_shared_array(
                shm_name, (tree_size,), np.float64
            )

        self.write_pos = 0
        self.n_entries = 0

    # ---- single-element ops (used by add) ----
    def _propagate_single(self, idx, change):
        while idx > 0:
            parent = (idx - 1) // 2
            self.tree[parent] += change
            idx = parent

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority):
        tree_idx = self.write_pos + self.capacity - 1
        self.update_single(tree_idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update_single(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate_single(tree_idx, change)

    # ---- vectorized batch ops ----
    def update_batch(self, tree_indices, priorities):
        tree_indices = np.asarray(tree_indices, dtype=np.intp)
        priorities = np.asarray(priorities, dtype=np.float64)
        changes = priorities - self.tree[tree_indices]
        self.tree[tree_indices] = priorities
        current_indices = tree_indices.copy()
        current_changes = changes.copy()
        while True:
            parent_indices = (current_indices - 1) // 2
            mask = current_indices > 0
            if not mask.any():
                break
            parent_indices = parent_indices[mask]
            current_changes = current_changes[mask]
            np.add.at(self.tree, parent_indices, current_changes)
            current_indices = parent_indices

    def retrieve_batch(self, values):
        values = np.asarray(values, dtype=np.float64)
        batch_size = len(values)
        indices = np.zeros(batch_size, dtype=np.intp)
        remaining = values.copy()
        tree = self.tree
        tree_len = len(tree)
        for _ in range(self._depth):
            left = 2 * indices + 1
            is_leaf = left >= tree_len
            if is_leaf.all():
                break
            left_values = np.where(is_leaf, 0.0, tree[np.minimum(left, tree_len - 1)])
            go_right = remaining > left_values
            right = left + 1
            new_indices = np.where(is_leaf, indices, np.where(go_right, right, left))
            new_remaining = np.where(is_leaf, remaining, np.where(go_right, remaining - left_values, remaining))
            indices = new_indices
            remaining = new_remaining
        indices = np.clip(indices, 0, tree_len - 1)
        priorities = tree[indices]
        data_indices = indices - (self.capacity - 1)
        return indices, priorities, data_indices

    def cleanup(self):
        """Close mmap/fd and unlink the /dev/shm file."""
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        path = os.path.join(SHM_DIR, self._shm_name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class SharedMinTree:
    """MinTree with tree array in shared memory (/dev/shm + mmap)."""

    def __init__(self, capacity, *, shm_name=None, create=True, init_inf=True):
        self.capacity = capacity
        tree_size = 2 * capacity - 1

        if create:
            self._shm_name = shm_name or 'rl_mintree_{}'.format(uuid.uuid4().hex[:12])
            self._shm_name, self.tree, self._fd, self._mm = _create_shared_array(
                self._shm_name, (tree_size,), np.float64
            )
            if init_inf:
                self.tree[:] = np.inf
        else:
            assert shm_name is not None
            self._shm_name = shm_name
            self.tree, self._fd, self._mm = _view_shared_array(
                shm_name, (tree_size,), np.float64
            )

    def update_single(self, tree_idx, priority):
        self.tree[tree_idx] = priority
        idx = tree_idx
        while idx > 0:
            parent = (idx - 1) // 2
            left = 2 * parent + 1
            right = left + 1
            self.tree[parent] = min(self.tree[left], self.tree[right])
            idx = parent

    def update_batch(self, tree_indices, priorities):
        tree_indices = np.asarray(tree_indices, dtype=np.intp)
        priorities = np.asarray(priorities, dtype=np.float64)
        self.tree[tree_indices] = priorities
        current_indices = tree_indices.copy()
        while True:
            parent_indices = (current_indices - 1) // 2
            mask = current_indices > 0
            if not mask.any():
                break
            parent_indices = parent_indices[mask]
            unique_parents = np.unique(parent_indices)
            left = 2 * unique_parents + 1
            right = left + 1
            right_clamped = np.minimum(right, len(self.tree) - 1)
            self.tree[unique_parents] = np.minimum(
                self.tree[left], self.tree[right_clamped]
            )
            current_indices = unique_parents

    @property
    def min(self):
        return float(self.tree[0])

    def cleanup(self):
        """Close mmap/fd and unlink the /dev/shm file."""
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        path = os.path.join(SHM_DIR, self._shm_name)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class SharedPrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with all storage in /dev/shm + mmap.

    Can be constructed in *owner* mode (__init__, allocates shm files)
    or *view* mode (from_shared, attaches numpy views to existing shm files
    by name). The view is used in the training sub-process (spawned).

    The owner process calls add(); the view process calls sample() and
    update_priorities(). They share the same underlying numpy arrays
    with zero copy.

    Compatible with Python 3.7+ and multiprocessing 'spawn' start method.
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device='cpu',
        alpha=0.6,
        beta=0.4,
        beta_annealing_steps=100000,
        min_priority=1e-6,
        use_bitpack=True,
        store_expert_actions=False,
    ):
        """Create buffer in *owner* mode (allocates /dev/shm files)."""
        self._is_owner = True
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.use_bitpack = use_bitpack

        # PER params
        self.alpha = alpha
        self.beta = beta
        self.beta_initial = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.min_priority = min_priority
        self._max_priority = 1.0
        self._sample_step = 0

        # Unique prefix for all shm files of this buffer instance
        self._shm_prefix = 'rl_buf_{}'.format(uuid.uuid4().hex[:12])

        # ---- Observation space analysis ----
        self._obs_is_dict = isinstance(observation_space, spaces.Dict)
        self.obs_keys = []         # type: List[str]
        self.obs_shapes = {}       # type: Dict[str, tuple]
        self._packed_keys = set()  # type: set
        self._packed_shapes = {}   # type: Dict[str, tuple]
        self.obs_shape = self._get_obs_shape()
        self.action_dim = self._get_action_dim()

        # ---- Allocate shared-memory arrays (/dev/shm + mmap) ----
        # Store (shm_name, shape, dtype_str) for each array so we can
        # pass them to child processes via get_shared_config() (picklable!).
        self._shm_meta = {}  # type: Dict[str, Tuple[str, tuple, str]]
        self._fds = []       # keep fds alive
        self._mms = []       # keep mmaps alive

        # Observations
        if self._obs_is_dict:
            self.observations = {}   # type: Dict[str, np.ndarray]
            self.next_observations = {}
            for key in self.obs_keys:
                space = observation_space.spaces[key]
                obs_dtype = space.dtype if hasattr(space, 'dtype') else np.float32
                shape = self.obs_shapes[key]
                if self._should_use_bitpack(space, key):
                    self._packed_keys.add(key)
                    self._packed_shapes[key] = shape
                    shape = self._get_packed_shape(shape)
                    obs_dtype = np.uint8
                full_shape = (buffer_size,) + tuple(shape)
                obs_name = '{}_{}_obs'.format(self._shm_prefix, key)
                nobs_name = '{}_{}_nobs'.format(self._shm_prefix, key)
                _, self.observations[key], fd1, mm1 = _create_shared_array(obs_name, full_shape, obs_dtype)
                _, self.next_observations[key], fd2, mm2 = _create_shared_array(nobs_name, full_shape, obs_dtype)
                self._fds.extend([fd1, fd2])
                self._mms.extend([mm1, mm2])
                self._shm_meta['obs_' + key] = (obs_name, full_shape, np.dtype(obs_dtype).str)
                self._shm_meta['nobs_' + key] = (nobs_name, full_shape, np.dtype(obs_dtype).str)
        else:
            obs_dtype = observation_space.dtype if hasattr(observation_space, 'dtype') else np.float32
            shape = self.obs_shape
            if self._should_use_bitpack(observation_space, None):
                self._packed_keys.add('__single__')
                self._packed_shapes['__single__'] = shape
                shape = self._get_packed_shape(shape)
                obs_dtype = np.uint8
            full_shape = (buffer_size,) + tuple(shape)
            obs_name = '{}_obs'.format(self._shm_prefix)
            nobs_name = '{}_nobs'.format(self._shm_prefix)
            _, self.observations, fd1, mm1 = _create_shared_array(obs_name, full_shape, obs_dtype)
            _, self.next_observations, fd2, mm2 = _create_shared_array(nobs_name, full_shape, obs_dtype)
            self._fds.extend([fd1, fd2])
            self._mms.extend([mm1, mm2])
            self._shm_meta['obs'] = (obs_name, full_shape, np.dtype(obs_dtype).str)
            self._shm_meta['nobs'] = (nobs_name, full_shape, np.dtype(obs_dtype).str)

        # Actions, rewards, terminateds, truncateds
        action_dtype = self._get_action_dtype()
        act_name = '{}_act'.format(self._shm_prefix)
        rew_name = '{}_rew'.format(self._shm_prefix)
        term_name = '{}_term'.format(self._shm_prefix)
        trunc_name = '{}_trunc'.format(self._shm_prefix)
        _, self.actions, fd_a, mm_a = _create_shared_array(act_name, (buffer_size, self.action_dim), action_dtype)
        _, self.rewards, fd_r, mm_r = _create_shared_array(rew_name, (buffer_size,), np.float32)
        _, self.terminateds, fd_t, mm_t = _create_shared_array(term_name, (buffer_size,), np.float32)
        _, self.truncateds, fd_tr, mm_tr = _create_shared_array(trunc_name, (buffer_size,), np.float32)
        self._fds.extend([fd_a, fd_r, fd_t, fd_tr])
        self._mms.extend([mm_a, mm_r, mm_t, mm_tr])
        self._shm_meta['act'] = (act_name, (buffer_size, self.action_dim), np.dtype(action_dtype).str)
        self._shm_meta['rew'] = (rew_name, (buffer_size,), np.dtype(np.float32).str)
        self._shm_meta['term'] = (term_name, (buffer_size,), np.dtype(np.float32).str)
        self._shm_meta['trunc'] = (trunc_name, (buffer_size,), np.dtype(np.float32).str)

        # Expert actions (optional, for BC with rule-based agent as source)
        self.store_expert_actions = store_expert_actions
        if store_expert_actions:
            eact_name = '{}_eact'.format(self._shm_prefix)
            _, self.expert_actions, fd_ea, mm_ea = _create_shared_array(
                eact_name, (buffer_size, self.action_dim), np.float32
            )
            self._fds.append(fd_ea)
            self._mms.append(mm_ea)
            self._shm_meta['eact'] = (eact_name, (buffer_size, self.action_dim), np.dtype(np.float32).str)
        else:
            self.expert_actions = None

        # ---- Shared pos/full via /dev/shm ----
        # Use a small shm file to hold two int32 values: [pos, full]
        self._pos_shm_name = '{}_pos'.format(self._shm_prefix)
        _, self._pos_arr, fd_p, mm_p = _create_shared_array(
            self._pos_shm_name, (2,), np.int32
        )
        self._fds.append(fd_p)
        self._mms.append(mm_p)
        # _pos_arr[0] = pos, _pos_arr[1] = full (0 or 1)

        # ---- SumTree / MinTree in shared memory ----
        self.tree = SharedSumTree(buffer_size, shm_name='{}_sumtree'.format(self._shm_prefix))
        self.min_tree = SharedMinTree(buffer_size, shm_name='{}_mintree'.format(self._shm_prefix))

        logger.debug(
            "[SharedPrioritizedReplayBuffer] Created (/dev/shm mmap backend) "
            "prefix=%s alpha=%s beta=%s->1.0 over %s sample calls",
            self._shm_prefix,
            alpha,
            beta,
            beta_annealing_steps,
        )
        if self._packed_keys:
            logger.debug("[SharedPrioritizedReplayBuffer] Bitpack keys: %s", self._packed_keys)

    # ---- Properties wrapping shared pos/full ----
    @property
    def pos(self):
        return int(self._pos_arr[0])

    @pos.setter
    def pos(self, val):
        self._pos_arr[0] = val

    @property
    def full(self):
        return bool(self._pos_arr[1])

    @full.setter
    def full(self, val):
        self._pos_arr[1] = int(val)

    # ---- Construct a view in another process ----

    def get_shared_config(self):
        """
        Return a fully-picklable dict with everything needed to reconstruct
        a *view* of this buffer in a spawned child process.

        Only contains strings, ints, floats, tuples, lists, dicts - no
        mp.RawArray / mp.Value / non-picklable objects.
        """
        return {
            'buffer_size': self.buffer_size,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_annealing_steps': self.beta_annealing_steps,
            'min_priority': self.min_priority,
            'use_bitpack': self.use_bitpack,
            'store_expert_actions': self.store_expert_actions,
            'obs_is_dict': self._obs_is_dict,
            'obs_keys': list(self.obs_keys),
            'obs_shapes': dict(self.obs_shapes),
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
            'packed_keys': list(self._packed_keys),
            'packed_shapes': dict(self._packed_shapes),
            # /dev/shm file names + shapes + dtypes (all picklable)
            'shm_meta': self._shm_meta,
            'pos_shm_name': self._pos_shm_name,
            'sumtree_shm_name': self.tree._shm_name,
            'sumtree_capacity': self.tree.capacity,
            'mintree_shm_name': self.min_tree._shm_name,
            'mintree_capacity': self.min_tree.capacity,
        }

    @classmethod
    def from_shared(cls, config, device='cpu'):
        """
        Attach to existing /dev/shm files created by the owner process.
        Used in the training sub-process (spawned).
        """
        obj = cls.__new__(cls)
        obj._is_owner = False
        obj.buffer_size = config['buffer_size']
        obj.observation_space = config['observation_space']
        obj.action_space = config['action_space']
        obj.device = device if isinstance(device, torch.device) else torch.device(device)
        obj.use_bitpack = config['use_bitpack']

        obj.alpha = config['alpha']
        obj.beta = config['beta']
        obj.beta_initial = config['beta']
        obj.beta_annealing_steps = config['beta_annealing_steps']
        obj.min_priority = config['min_priority']
        obj._max_priority = 1.0
        obj._sample_step = 0

        obj._obs_is_dict = config['obs_is_dict']
        obj.obs_keys = config['obs_keys']
        obj.obs_shapes = config['obs_shapes']
        obj.obs_shape = config['obs_shape']
        obj.action_dim = config['action_dim']
        obj._packed_keys = set(config['packed_keys'])
        obj._packed_shapes = config['packed_shapes']

        shm_meta = config['shm_meta']
        obj._fds = []
        obj._mms = []

        # Attach observation arrays
        if obj._obs_is_dict:
            obj.observations = {}
            obj.next_observations = {}
            for key in obj.obs_keys:
                obs_name, shape, dtype_str = shm_meta['obs_' + key]
                nobs_name, nshape, ndtype_str = shm_meta['nobs_' + key]
                obs_arr, fd1, mm1 = _view_shared_array(obs_name, tuple(shape), dtype_str)
                nobs_arr, fd2, mm2 = _view_shared_array(nobs_name, tuple(nshape), ndtype_str)
                obj.observations[key] = obs_arr
                obj.next_observations[key] = nobs_arr
                obj._fds.extend([fd1, fd2])
                obj._mms.extend([mm1, mm2])
        else:
            obs_name, shape, dtype_str = shm_meta['obs']
            nobs_name, nshape, ndtype_str = shm_meta['nobs']
            obj.observations, fd1, mm1 = _view_shared_array(obs_name, tuple(shape), dtype_str)
            obj.next_observations, fd2, mm2 = _view_shared_array(nobs_name, tuple(nshape), ndtype_str)
            obj._fds.extend([fd1, fd2])
            obj._mms.extend([mm1, mm2])

        # Attach scalar arrays
        act_name, act_shape, act_dtype = shm_meta['act']
        rew_name, rew_shape, rew_dtype = shm_meta['rew']
        term_name, term_shape, term_dtype = shm_meta['term']
        trunc_name, trunc_shape, trunc_dtype = shm_meta['trunc']
        obj.actions, fd_a, mm_a = _view_shared_array(act_name, tuple(act_shape), act_dtype)
        obj.rewards, fd_r, mm_r = _view_shared_array(rew_name, tuple(rew_shape), rew_dtype)
        obj.terminateds, fd_t, mm_t = _view_shared_array(term_name, tuple(term_shape), term_dtype)
        obj.truncateds, fd_tr, mm_tr = _view_shared_array(trunc_name, tuple(trunc_shape), trunc_dtype)
        obj._fds.extend([fd_a, fd_r, fd_t, fd_tr])
        obj._mms.extend([mm_a, mm_r, mm_t, mm_tr])

        # Attach expert actions (optional)
        obj.store_expert_actions = config.get('store_expert_actions', False)
        if obj.store_expert_actions and 'eact' in shm_meta:
            eact_name, eact_shape, eact_dtype = shm_meta['eact']
            obj.expert_actions, fd_ea, mm_ea = _view_shared_array(
                eact_name, tuple(eact_shape), eact_dtype
            )
            obj._fds.append(fd_ea)
            obj._mms.append(mm_ea)
        else:
            obj.expert_actions = None

        # Attach shared pos/full
        pos_name = config['pos_shm_name']
        obj._pos_arr, fd_p, mm_p = _view_shared_array(pos_name, (2,), np.int32)
        obj._fds.append(fd_p)
        obj._mms.append(mm_p)

        # Attach SumTree / MinTree
        obj.tree = SharedSumTree(
            config['sumtree_capacity'],
            shm_name=config['sumtree_shm_name'],
            create=False,
        )
        obj.min_tree = SharedMinTree(
            config['mintree_capacity'],
            shm_name=config['mintree_shm_name'],
            create=False,
            init_inf=False,
        )

        logger.debug("[SharedPrioritizedReplayBuffer] Attached view (/dev/shm mmap)")
        return obj

    # ---- Buffer interface ----

    def size(self):
        return self.buffer_size if self.full else self.pos

    def __len__(self):
        return self.size()

    def can_sample(self, batch_size):
        return self.size() >= batch_size

    # ---- add() : called by main (collect) process ----

    def add(self, obs, next_obs, action, reward, terminated, truncated=None, expert_action=None):
        action = np.asarray(action)
        reward = np.asarray(reward)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated) if truncated is not None else np.zeros_like(terminated)
        if expert_action is not None:
            expert_action = np.asarray(expert_action, dtype=np.float32)

        # Determine batch vs single
        if self._obs_is_dict:
            first_key = self.obs_keys[0]
            obs_arr = np.asarray(obs[first_key])
            is_batch = obs_arr.ndim == len(self.obs_shapes[first_key]) + 1
            batch_size = obs_arr.shape[0] if is_batch else 1
        else:
            obs = np.asarray(obs)
            next_obs = np.asarray(next_obs)
            is_batch = obs.ndim == len(self.obs_shape) + 1
            batch_size = obs.shape[0] if is_batch else 1

        current_pos = self.pos

        if is_batch:
            if action.ndim == 1 and self.action_dim > 1:
                action = action.reshape(-1, self.action_dim)
            elif action.ndim == 1 and self.action_dim == 1:
                action = action.reshape(-1, 1)

            if current_pos + batch_size <= self.buffer_size:
                indices = np.arange(current_pos, current_pos + batch_size)
            else:
                first_part = self.buffer_size - current_pos
                indices = np.concatenate([
                    np.arange(current_pos, self.buffer_size),
                    np.arange(0, batch_size - first_part)
                ])

            if self._obs_is_dict:
                for key in self.obs_keys:
                    obs_data = np.asarray(obs[key])
                    next_obs_data = np.asarray(next_obs[key])
                    if key in self._packed_keys:
                        obs_data = self._pack_obs(obs_data, key)
                        next_obs_data = self._pack_obs(next_obs_data, key)
                    self.observations[key][indices] = obs_data
                    self.next_observations[key][indices] = next_obs_data
            else:
                if '__single__' in self._packed_keys:
                    obs = self._pack_obs(obs)
                    next_obs = self._pack_obs(next_obs)
                self.observations[indices] = obs
                self.next_observations[indices] = next_obs

            self.actions[indices] = action
            self.rewards[indices] = reward.flatten()
            self.terminateds[indices] = terminated.flatten().astype(np.float32)
            self.truncateds[indices] = truncated.flatten().astype(np.float32)
            if self.store_expert_actions and self.expert_actions is not None:
                if expert_action is not None:
                    if expert_action.ndim == 1 and self.action_dim > 1:
                        expert_action = expert_action.reshape(-1, self.action_dim)
                    elif expert_action.ndim == 1 and self.action_dim == 1:
                        expert_action = expert_action.reshape(-1, 1)
                    self.expert_actions[indices] = expert_action
                else:
                    self.expert_actions[indices] = 0.0

            new_pos = (current_pos + batch_size) % self.buffer_size
            if current_pos + batch_size >= self.buffer_size:
                self.full = True
            self.pos = new_pos
        else:
            if action.ndim == 0:
                action = np.array([action])
            elif action.ndim == 1 and len(action) != self.action_dim:
                action = action.reshape(self.action_dim)

            if self._obs_is_dict:
                for key in self.obs_keys:
                    obs_data = np.asarray(obs[key])
                    next_obs_data = np.asarray(next_obs[key])
                    if key in self._packed_keys:
                        obs_data = self._pack_obs(obs_data, key)
                        next_obs_data = self._pack_obs(next_obs_data, key)
                    self.observations[key][current_pos] = obs_data
                    self.next_observations[key][current_pos] = next_obs_data
            else:
                if '__single__' in self._packed_keys:
                    obs = self._pack_obs(obs)
                    next_obs = self._pack_obs(next_obs)
                self.observations[current_pos] = obs
                self.next_observations[current_pos] = next_obs

            self.actions[current_pos] = action
            self.rewards[current_pos] = float(reward)
            self.terminateds[current_pos] = float(terminated)
            self.truncateds[current_pos] = float(truncated)
            if self.store_expert_actions and self.expert_actions is not None:
                if expert_action is not None:
                    self.expert_actions[current_pos] = expert_action.flatten()
                else:
                    self.expert_actions[current_pos] = 0.0

            new_pos = (current_pos + 1) % self.buffer_size
            if new_pos == 0:
                self.full = True
            self.pos = new_pos

        # Update SumTree / MinTree priorities for new transitions
        priority = self._max_priority ** self.alpha
        data_indices = np.arange(batch_size) + current_pos
        data_indices = data_indices % self.buffer_size
        tree_indices = data_indices + self.tree.capacity - 1
        priorities = np.full(batch_size, priority, dtype=np.float64)
        self.tree.update_batch(tree_indices, priorities)
        self.min_tree.update_batch(tree_indices, priorities)
        self.tree.n_entries = self.size()

    # ---- sample() : called by train process ----

    def sample(self, batch_size):
        self._sample_step += 1
        fraction = min(1.0, self._sample_step / max(1, self.beta_annealing_steps))
        current_beta = self.beta_initial + fraction * (1.0 - self.beta_initial)

        total = self.tree.total
        if total <= 0 or np.isnan(total) or np.isinf(total):
            upper_bound = self.buffer_size if self.full else self.pos
            upper_bound = max(upper_bound, 1)
            data_indices = np.random.randint(0, upper_bound, size=batch_size)
            tree_indices = data_indices + self.tree.capacity - 1
            weights = torch.ones(batch_size, 1, device=self.device)
            return self._build_samples(data_indices, weights, tree_indices)

        segment = total / batch_size
        segment_starts = np.arange(batch_size, dtype=np.float64) * segment
        segment_ends = segment_starts + segment

        # Clamp to avoid OverflowError in np.random.uniform
        segment_ends = np.minimum(segment_ends, total)
        segment_starts = np.minimum(segment_starts, segment_ends - 1e-10)
        segment_starts = np.maximum(segment_starts, 0.0)

        rand_values = np.random.uniform(segment_starts, segment_ends)
        tree_indices, priorities, data_indices = self.tree.retrieve_batch(rand_values)

        max_idx = self.buffer_size if self.full else max(self.pos, 1)
        data_indices = data_indices % max_idx

        priorities = np.maximum(priorities, self.min_priority)

        n = max(self.size(), 1)
        probabilities = priorities / max(total, 1e-10)
        weights = (n * probabilities) ** (-current_beta)
        max_w = weights.max()
        if max_w > 0:
            weights = weights / max_w
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return self._build_samples(data_indices, weights, tree_indices)

    # ---- update_priorities() : called by train process ----

    def update_priorities(self, tree_indices, td_errors):
        td_errors = np.abs(td_errors).flatten()
        tree_indices = np.asarray(tree_indices, dtype=np.intp)
        priorities = (td_errors + self.min_priority) ** self.alpha

        unique_indices, inverse = np.unique(tree_indices, return_inverse=True)
        if unique_indices.size != tree_indices.size:
            unique_priorities = np.full(unique_indices.shape, -np.inf, dtype=np.float64)
            np.maximum.at(unique_priorities, inverse, priorities)
            tree_indices = unique_indices
            priorities = unique_priorities

        self.tree.update_batch(tree_indices, priorities)
        self.min_tree.update_batch(tree_indices, priorities)
        max_raw = (td_errors + self.min_priority).max()
        self._max_priority = max(self._max_priority, float(max_raw))

    # ---- Internal helpers ----

    def _build_samples(self, indices, weights, tree_indices):
        dones = self.terminateds[indices]

        if self._obs_is_dict:
            obs = {}
            next_obs = {}
            for key in self.obs_keys:
                obs_data = self.observations[key][indices]
                next_obs_data = self.next_observations[key][indices]
                if key in self._packed_keys:
                    # GPU-side unpack: send packed uint8 to GPU, then unpack bits
                    # Skips CPU unpackbits/float32/copy (saves ~280ms per call)
                    obs[key] = self._packed_to_gpu(obs_data, key)
                    next_obs[key] = self._packed_to_gpu(next_obs_data, key)
                else:
                    obs[key] = obs_data
                    next_obs[key] = next_obs_data
        else:
            obs = self.observations[indices]
            next_obs = self.next_observations[indices]
            if '__single__' in self._packed_keys:
                obs = self._packed_to_gpu(obs)
                next_obs = self._packed_to_gpu(next_obs)

        expert_act = None
        if self.store_expert_actions and self.expert_actions is not None:
            expert_act = self._to_torch(self.expert_actions[indices])

        return PrioritizedReplayBufferSamples(
            observations=self._to_torch(obs),
            actions=self._to_torch(self.actions[indices]),
            next_observations=self._to_torch(next_obs),
            dones=self._to_torch(dones.reshape(-1, 1)),
            rewards=self._to_torch(self.rewards[indices].reshape(-1, 1)),
            weights=weights,
            indices=tree_indices,
            expert_actions=expert_act,
        )

    def _packed_to_gpu(self, packed_arr, key=None):
        """Send packed uint8 array to GPU and unpack there.

        This replaces the old CPU path:
            np.unpackbits (96ms) -> *255 -> astype(float32) (84ms)
            -> .copy() + from_numpy (100ms) -> .to(device) (53ms)
        with:
            .copy() (3ms) -> .to(device) (1.5ms) -> GPU bit-unpack (0.5ms)

        Total: ~340ms -> ~5ms (68x speedup per call)
        """
        original_shape = self._packed_shapes.get(key or '__single__')
        if original_shape is None:
            raise ValueError("No packed shape for key: {}".format(key))
        original_last = original_shape[-1]
        # Small contiguous copy (18 MB packed vs 576 MB unpacked float32)
        packed_np = np.ascontiguousarray(packed_arr)
        t = torch.from_numpy(packed_np.copy()).to(self.device, non_blocking=True)
        # GPU-side unpack: bit extraction + float conversion all on GPU
        return _gpu_unpackbits(t, original_last)

    def _to_torch(self, arr):
        if isinstance(arr, dict):
            result = {}
            for key, value in arr.items():
                if isinstance(value, torch.Tensor):
                    # Already a GPU tensor (from _packed_to_gpu)
                    result[key] = value
                    continue
                if value.dtype != np.float32:
                    value = value.astype(np.float32, copy=False)
                if not value.flags['C_CONTIGUOUS']:
                    value = np.ascontiguousarray(value)
                # Copy so the tensor owns its memory (safe across processes)
                result[key] = torch.from_numpy(value.copy()).to(self.device, non_blocking=True)
            return result
        if isinstance(arr, torch.Tensor):
            return arr  # Already a GPU tensor
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        return torch.from_numpy(arr.copy()).to(self.device, non_blocking=True)

    # ---- Obs space helpers (same as ReplayBuffer) ----

    def _get_obs_shape(self):
        if isinstance(self.observation_space, spaces.Dict):
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

    def _get_action_dim(self):
        if isinstance(self.action_space, spaces.Discrete):
            return 1
        return int(np.prod(self.action_space.shape))

    def _get_action_dtype(self):
        if hasattr(self.action_space, 'dtype'):
            dtype = self.action_space.dtype
            return np.float32 if dtype == np.float64 else dtype
        return np.float32

    def _should_use_bitpack(self, space, key):
        if not self.use_bitpack:
            return False
        if not isinstance(space, spaces.Box):
            return False
        if not hasattr(space, 'dtype') or space.dtype != np.uint8:
            return False
        if len(space.shape) != 3:
            return False
        if key is not None:
            key_lower = key.lower()
            if any(hint in key_lower for hint in ['vector', 'bev', 'mask', 'binary']):
                return True
        if space.shape[1] == space.shape[2]:
            return True
        return False

    def _get_packed_shape(self, shape):
        leading = list(shape[:-1])
        last = shape[-1]
        packed_last = (last + 7) // 8
        return tuple(leading) + (packed_last,)

    def _pack_obs(self, obs, key=None):
        if obs.max() > 1:
            obs = (obs > 127).astype(np.uint8)
        else:
            obs = obs.astype(np.uint8)
        return np.packbits(obs, axis=-1)

    def _unpack_obs(self, packed, key=None):
        if key is None:
            original_shape = self._packed_shapes.get('__single__')
        else:
            original_shape = self._packed_shapes.get(key)
        if original_shape is None:
            raise ValueError("No packed shape found for key: {}".format(key))
        original_last = original_shape[-1]
        unpacked = np.unpackbits(packed, axis=-1)
        unpacked = unpacked[..., :original_last]
        return (unpacked * 255).astype(np.uint8)

    def get_stats(self):
        current_size = self.size()
        return {
            'size': current_size,
            'buffer_size': self.buffer_size,
            'pos': self.pos,
            'full': self.full,
            'alpha': self.alpha,
            'max_priority': self._max_priority,
            'total_tree_sum': self.tree.total,
            'min_tree_min': self.min_tree.min,
            'sample_steps': self._sample_step,
        }

    def cleanup(self):
        """
        Close mmap/fd handles and unlink /dev/shm files.
        Only the owner process should call this.
        """
        if not getattr(self, '_is_owner', False):
            # View process: just close handles, don't unlink
            for mm in getattr(self, '_mms', []):
                try:
                    mm.close()
                except Exception:
                    pass
            for fd in getattr(self, '_fds', []):
                try:
                    os.close(fd)
                except Exception:
                    pass
            return

        # Owner: close handles and unlink files
        for mm in getattr(self, '_mms', []):
            try:
                mm.close()
            except Exception:
                pass
        for fd in getattr(self, '_fds', []):
            try:
                os.close(fd)
            except Exception:
                pass

        # Unlink all shm files
        for key, (name, _, _) in getattr(self, '_shm_meta', {}).items():
            path = os.path.join(SHM_DIR, name)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        # pos shm
        pos_name = getattr(self, '_pos_shm_name', None)
        if pos_name:
            try:
                os.unlink(os.path.join(SHM_DIR, pos_name))
            except FileNotFoundError:
                pass

        # Trees
        if hasattr(self, 'tree'):
            self.tree.cleanup()
        if hasattr(self, 'min_tree'):
            self.min_tree.cleanup()

        logger.info("[SharedPrioritizedReplayBuffer] Cleaned up /dev/shm files")
