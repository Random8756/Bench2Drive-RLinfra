"""
Action noise for off-policy algorithms with deterministic policies (TD3).

Provides Gaussian action noise plus a vectorised wrapper that maintains an
independent copy of the noise generator for each parallel environment.
"""

import copy
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np


class ActionNoise(ABC):
    """Base class for action noise."""

    def reset(self) -> None:
        """Reset noise state (called at episode end)."""
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        """Generate noise."""
        raise NotImplementedError


class NormalActionNoise(ActionNoise):
    """
    Gaussian action noise.

    Args:
        mean: Mean of the noise.
        sigma: Standard deviation of the noise.
    """

    def __init__(self, mean: np.ndarray, sigma: np.ndarray):
        self._mu = mean
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma).astype(np.float32)

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"


class EpsilonGreedyExploration:
    """
    Epsilon-greedy exploration for continuous action spaces.

    With probability exploit_prob, use policy output (deterministic).
    With probability (1 - exploit_prob), randomly choose from predefined actions.

    exploit_prob linearly interpolates from initial to final over decay_steps.

    Args:
        explore_actions: Predefined actions in env space, shape (N, action_dim).
        exploit_prob_initial: Initial probability of using model output.
        exploit_prob_final: Final probability after decay.
        decay_steps: Steps over which exploit_prob linearly changes.
    """

    def __init__(
        self,
        explore_actions: List[List[float]],
        exploit_prob_initial: float = 0.7,
        exploit_prob_final: float = 0.7,
        decay_steps: int = 1000000,
    ):
        self.explore_actions = np.array(explore_actions, dtype=np.float32)
        self.exploit_prob_initial = exploit_prob_initial
        self.exploit_prob_final = exploit_prob_final
        self.decay_steps = max(1, decay_steps)

    def get_exploit_prob(self, timestep: int) -> float:
        progress = min(1.0, timestep / self.decay_steps)
        return self.exploit_prob_initial + (self.exploit_prob_final - self.exploit_prob_initial) * progress

    def should_exploit(self, timestep: int) -> bool:
        return np.random.random() < self.get_exploit_prob(timestep)

    def sample_explore_action(self) -> np.ndarray:
        """Sample a random action from predefined exploration actions (env space)."""
        idx = np.random.randint(len(self.explore_actions))
        return self.explore_actions[idx].copy()

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyExploration("
            f"n_actions={len(self.explore_actions)}, "
            f"exploit_prob={self.exploit_prob_initial}->{self.exploit_prob_final} "
            f"over {self.decay_steps} steps)"
        )


class VectorizedActionNoise(ActionNoise):
    """
    Vectorised action noise for parallel environments.

    Creates independent noise generators for each environment.

    Args:
        base_noise: Base noise generator to copy.
        n_envs: Number of parallel environments.
    """

    def __init__(self, base_noise: ActionNoise, n_envs: int):
        if n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")

        self._n_envs = n_envs
        self._base_noise = base_noise
        self._noises: List[ActionNoise] = [
            copy.deepcopy(base_noise) for _ in range(n_envs)
        ]

    def __call__(self, indices: Optional[Iterable[int]] = None) -> np.ndarray:
        """
        Generate and stack noise for environments.

        Args:
            indices: Environment indices to generate noise for. None generates for all.

        Returns:
            Noise array of shape (len(indices), action_dim) or (n_envs, action_dim).
        """
        if indices is None:
            return np.stack([noise() for noise in self._noises])
        return np.stack([self._noises[idx]() for idx in indices])

    def reset(self, indices: Optional[Iterable[int]] = None) -> None:
        """
        Reset noise for specified environments.

        Args:
            indices: Environment indices to reset. None resets all.
        """
        if indices is None:
            indices = range(self._n_envs)

        for idx in indices:
            self._noises[idx].reset()

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def __repr__(self) -> str:
        return f"VectorizedActionNoise(base={self._base_noise!r}, n_envs={self._n_envs})"
