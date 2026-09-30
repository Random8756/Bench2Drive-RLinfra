"""Per-worker rollout storage that can discard crashed-worker episodes."""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, NamedTuple, Optional, Union
import numpy as np
import torch
from gymnasium import spaces

from ..utils.obs_utils import ObsUtils

__layer__ = (4, "Algorithm")


def _split_minibatch_indices(
    indices: np.ndarray,
    batch_size: int,
) -> List[np.ndarray]:
    """Split shuffled indices without leaving a disproportionately small tail."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batches = [
        indices[start_idx:start_idx + batch_size]
        for start_idx in range(0, len(indices), batch_size)
    ]

    # Async rollout collection can overshoot n_steps by a few transitions.
    # Merge that small remainder so it does not receive a full optimizer step
    # with the same weight as a normal minibatch.
    if batch_size > 1 and len(batches) > 1:
        min_batch_size = max(2, batch_size // 2)
        if len(batches[-1]) < min_batch_size:
            batches[-2] = np.concatenate((batches[-2], batches[-1]))
            batches.pop()

    return batches


class RolloutBufferSamples(NamedTuple):
    """Named tuple for rollout buffer samples."""
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_action_dist_params: Optional[torch.Tensor]


@dataclass
class WorkerEpisodeData:
    """Data for a single worker's current episode."""
    observations: List[Union[np.ndarray, Dict[str, np.ndarray]]] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    action_dist_params: List[Optional[np.ndarray]] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)  # episode_start flags
    
    def clear(self):
        """Clear all data."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.action_dist_params.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.observations)


class PerWorkerRolloutBuffer:
    """
    Per-Worker Rollout Buffer for on-policy algorithms.
    
    Unlike timestep-major rollout buffers that store all workers together,
    this buffer stores data per worker, enabling:
    - Flexible async environment interaction
    - Per-worker episode discard (for crashed episodes)
    - Variable episode lengths per worker
    
    Storage layout:
    - Each worker has its own list of transitions
    - On compute_returns_and_advantage(), data is merged and GAE is computed
    - Batches are sampled from the merged data
    
    Truncated vs terminated transitions:
        - ``terminated``: real episode end; the obs is the terminal
          observation and the worker resets.
        - ``truncated``: reached ``max_episode_steps``; the worker also
          resets, and the caller must bootstrap reward with
          ``gamma * V(terminal_obs)``.

    Algorithm-internal slicing (``truncation_steps``):
        - When set, the buffer inserts artificial episode boundaries every
          ``truncation_steps`` steps before GAE, bootstrapping reward at
          each boundary. This is purely algorithm-side and does not affect
          the environment.
    """
    
    def __init__(
        self,
        max_steps: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_workers: int,
        device: Union[str, torch.device] = 'cpu',
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        truncation_steps: Optional[int] = None,
    ):
        """
        Initialize per-worker buffer.
        
        Args:
            max_steps: Maximum total steps to collect before training.
            observation_space: Observation space.
            action_space: Action space.
            num_workers: Number of parallel workers.
            device: Device for tensors.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            truncation_steps: Algorithm-internal episode slicing step count (None = disabled).
                If set, episodes longer than truncation_steps will be split into sub-segments
                for GAE computation, with bootstrap at each boundary.
        """
        self.max_steps = max_steps
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_workers = num_workers
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.truncation_steps = truncation_steps
        
        # Get shapes
        self._obs_is_dict = isinstance(self.observation_space, spaces.Dict)
        self.obs_keys: List[str] = []
        self.obs_shapes: Dict[str, tuple] = {}
        self.obs_shape = self._get_obs_shape()
        self.action_dim = self._get_action_dim()
        
        # Per-worker storage
        self._worker_data: Dict[int, WorkerEpisodeData] = {
            i: WorkerEpisodeData() for i in range(num_workers)
        }
        
        # Merged data for training (populated by compute_returns_and_advantage)
        self._merged_obs: Optional[np.ndarray] = None
        self._merged_actions: Optional[np.ndarray] = None
        self._merged_values: Optional[np.ndarray] = None
        self._merged_log_probs: Optional[np.ndarray] = None
        self._merged_action_dist_params: Optional[np.ndarray] = None
        self._merged_advantages: Optional[np.ndarray] = None
        self._merged_returns: Optional[np.ndarray] = None
        
        # State
        self._total_steps = 0
        self._ready_for_training = False
    
    def _get_obs_shape(self) -> tuple:
        """Get observation shape from observation space."""
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
    
    def _get_action_dim(self) -> int:
        """Get action dimension from action space."""
        if isinstance(self.action_space, spaces.Discrete):
            return 1
        return int(np.prod(self.action_space.shape))
    
    def reset(self) -> None:
        """Reset buffer for new rollout collection."""
        for data in self._worker_data.values():
            data.clear()
        
        self._merged_obs = None
        self._merged_actions = None
        self._merged_values = None
        self._merged_log_probs = None
        self._merged_action_dist_params = None
        self._merged_advantages = None
        self._merged_returns = None
        
        self._total_steps = 0
        self._ready_for_training = False
    
    def add(
        self,
        worker_id: int,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        action: Union[int, np.ndarray],
        reward: float,
        done: bool,  # episode_start (from previous done)
        value: float,
        log_prob: float,
        action_dist_params: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a transition for a specific worker.
        
        Args:
            worker_id: Worker ID.
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            done: Whether this is episode start (previous was done).
            value: Value estimate.
            log_prob: Log probability of action.
            action_dist_params: Optional policy distribution parameters for exact KL.
        """
        if worker_id not in self._worker_data:
            self._worker_data[worker_id] = WorkerEpisodeData()
        
        data = self._worker_data[worker_id]
        
        # Ensure correct shapes
        if self._obs_is_dict and isinstance(obs, dict):
            obs = {key: np.asarray(obs[key]) for key in self.obs_keys}
        else:
            obs = np.asarray(obs)
        action = np.asarray(action)
        if action.ndim == 0:
            action = action.reshape(1)
        elif self.action_dim == 1 and action.shape == ():
            action = np.array([action])
        
        data.observations.append(obs)
        data.actions.append(action)
        data.rewards.append(float(reward))
        data.dones.append(bool(done))
        data.values.append(float(value))
        data.log_probs.append(float(log_prob))
        data.action_dist_params.append(
            None
            if action_dist_params is None
            else np.asarray(action_dist_params, dtype=np.float32).copy()
        )
        
        self._total_steps += 1
        self._ready_for_training = False
    
    def discard_episode(self, worker_id: int) -> int:
        """
        Discard current episode data for a crashed worker.
        
        This clears all pending data for the worker, as the episode
        was not completed normally (crashed).
        
        Args:
            worker_id: Worker ID whose episode to discard.
            
        Returns:
            Number of steps discarded.
        """
        if worker_id not in self._worker_data:
            return 0
        
        data = self._worker_data[worker_id]
        discarded = len(data)
        self._total_steps -= discarded
        data.clear()
        
        return discarded
    
    def compute_returns_and_advantage(
        self,
        last_values: Dict[int, float],
        last_dones: Dict[int, bool],
    ) -> None:
        """
        Compute returns and advantages using GAE for all workers.
        
        Merges all worker data and computes GAE across the combined dataset.
        
        Args:
            last_values: {worker_id: value} for last observations.
            last_dones: {worker_id: done} for last observations.
        """
        if self._total_steps == 0:
            self._ready_for_training = True
            return
        
        # Collect all data and compute GAE per worker
        all_obs = []
        all_actions = []
        all_values = []
        all_log_probs = []
        all_action_dist_params = []
        all_advantages = []
        all_returns = []
        workers_with_dist_params = 0
        workers_without_dist_params = 0
        
        for worker_id, data in self._worker_data.items():
            if len(data) == 0:
                continue
            
            # Convert to arrays
            obs = ObsUtils.stack_obs(data.observations, keys=self.obs_keys)
            actions = np.array(data.actions)
            rewards = np.array(data.rewards)
            values = np.array(data.values)
            log_probs = np.array(data.log_probs)
            if data.action_dist_params and len(data.action_dist_params) != len(data):
                raise RuntimeError(
                    f"Worker {worker_id} has {len(data.action_dist_params)} action "
                    f"distribution parameter entries for {len(data)} transitions"
                )
            has_dist_params = [params is not None for params in data.action_dist_params]
            if has_dist_params and any(has_dist_params) and not all(has_dist_params):
                raise RuntimeError(
                    f"Worker {worker_id} rollout mixes transitions with and without "
                    "action distribution parameters"
                )
            if has_dist_params and all(has_dist_params):
                action_dist_params = np.stack(data.action_dist_params, axis=0)
                workers_with_dist_params += 1
            else:
                action_dist_params = None
                workers_without_dist_params += 1
            episode_starts = np.array(data.dones)
            
            # Get last value for this worker
            last_value = last_values.get(worker_id, 0.0)
            last_done = last_dones.get(worker_id, False)
            
            # Algorithm-internal truncation: insert artificial boundaries for long episodes
            if self.truncation_steps is not None and self.truncation_steps > 0:
                steps_in_segment = 0
                for t in range(len(rewards)):
                    if episode_starts[t]:
                        steps_in_segment = 0
                    steps_in_segment += 1
                    if steps_in_segment >= self.truncation_steps and t + 1 < len(rewards):
                        if not episode_starts[t + 1]:
                            episode_starts[t + 1] = True
                            # Bootstrap: add gamma * V(s_{t+1}) to reward at truncation point
                            rewards[t] += self.gamma * values[t + 1]
                        steps_in_segment = 0
            
            # Compute GAE
            advantages = np.zeros_like(rewards)
            last_gae_lam = 0.0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - float(last_done)
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - float(episode_starts[t + 1])
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
            
            returns = advantages + values
            
            # Append to merged arrays
            all_obs.append(obs)
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_probs)
            if action_dist_params is not None:
                all_action_dist_params.append(action_dist_params)
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # Merge all data
        if all_obs:
            if workers_with_dist_params and workers_without_dist_params:
                raise RuntimeError(
                    "Rollout mixes workers with and without action distribution parameters"
                )
            self._merged_obs = ObsUtils.concat_obs(all_obs, keys=self.obs_keys)
            self._merged_actions = np.concatenate(all_actions, axis=0)
            self._merged_values = np.concatenate(all_values, axis=0)
            self._merged_log_probs = np.concatenate(all_log_probs, axis=0)
            self._merged_action_dist_params = (
                np.concatenate(all_action_dist_params, axis=0)
                if all_action_dist_params
                else None
            )
            self._merged_advantages = np.concatenate(all_advantages, axis=0)
            self._merged_returns = np.concatenate(all_returns, axis=0)
        else:
            # No data, create empty arrays
            empty_shape = self.obs_shapes if self._obs_is_dict else self.obs_shape
            self._merged_obs = ObsUtils.empty_obs(empty_shape, keys=self.obs_keys)
            self._merged_actions = np.zeros((0, self.action_dim))
            self._merged_values = np.zeros(0)
            self._merged_log_probs = np.zeros(0)
            self._merged_action_dist_params = None
            self._merged_advantages = np.zeros(0)
            self._merged_returns = np.zeros(0)
        
        self._ready_for_training = True
    
    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Get batches of data for training.
        
        Args:
            batch_size: Size of each batch. If None, return all data at once.
            
        Yields:
            RolloutBufferSamples namedtuples containing tensors.
        """
        if not self._ready_for_training:
            raise RuntimeError("Must call compute_returns_and_advantage() before get()")
        
        if self._merged_obs is None or ObsUtils.get_obs_length(self._merged_obs, keys=self.obs_keys) == 0:
            return
        
        total_samples = ObsUtils.get_obs_length(self._merged_obs, keys=self.obs_keys)
        
        if batch_size is None:
            batch_size = total_samples
        
        # Random permutation for shuffling
        indices = np.random.permutation(total_samples)
        
        for batch_indices in _split_minibatch_indices(indices, batch_size):
            batch_obs = ObsUtils.slice_obs(self._merged_obs, batch_indices, keys=self.obs_keys)
            yield RolloutBufferSamples(
                observations=self._to_torch(batch_obs),
                actions=self._to_torch(self._merged_actions[batch_indices]),
                old_values=self._to_torch(self._merged_values[batch_indices]),
                old_log_prob=self._to_torch(self._merged_log_probs[batch_indices]),
                advantages=self._to_torch(self._merged_advantages[batch_indices]),
                returns=self._to_torch(self._merged_returns[batch_indices]),
                old_action_dist_params=(
                    self._to_torch(self._merged_action_dist_params[batch_indices])
                    if self._merged_action_dist_params is not None
                    else None
                ),
            )
    
    def _to_torch(self, arr: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert numpy array to torch tensor."""
        if isinstance(arr, dict):
            return {
                key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                for key, value in arr.items()
            }
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    
    @property
    def total_steps(self) -> int:
        """Total steps collected across all workers."""
        return self._total_steps
    
    def is_full(self) -> bool:
        """Check if buffer has collected enough steps."""
        return self._total_steps >= self.max_steps
    
    def __len__(self) -> int:
        """Return total steps in buffer."""
        return self._total_steps
    
    def get_worker_steps(self, worker_id: int) -> int:
        """Get number of steps for a specific worker."""
        if worker_id not in self._worker_data:
            return 0
        return len(self._worker_data[worker_id])
    
    def get_stats(self) -> Dict[str, any]:
        """Get buffer statistics."""
        worker_steps = {wid: len(data) for wid, data in self._worker_data.items()}
        return {
            'total_steps': self._total_steps,
            'max_steps': self.max_steps,
            'num_workers': self.num_workers,
            'worker_steps': worker_steps,
            'ready_for_training': self._ready_for_training,
        }
    
    @property
    def values(self) -> Optional[np.ndarray]:
        """
        Get merged values array (available after compute_returns_and_advantage).
        
        Used for computing explained_variance.
        """
        return self._merged_values

    @property
    def actions(self) -> Optional[np.ndarray]:
        """Get merged environment actions for rollout diagnostics."""
        return self._merged_actions

    @property
    def action_dist_params(self) -> Optional[np.ndarray]:
        """Get merged rollout-time policy distribution parameters."""
        return self._merged_action_dist_params
    
    @property
    def returns(self) -> Optional[np.ndarray]:
        """
        Get merged returns array (available after compute_returns_and_advantage).
        
        Used for computing explained_variance.
        """
        return self._merged_returns
    
    @property
    def advantages(self) -> Optional[np.ndarray]:
        """
        Get merged advantages array (available after compute_returns_and_advantage).
        
        Used for computing diagnostic metrics.
        """
        return self._merged_advantages
