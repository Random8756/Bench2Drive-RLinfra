"""Bench2Drive reinforcement-learning infrastructure for CARLA."""

__layer__ = (4, "Algorithm")

from .learning.adapters import StandardEnvAdapter
from .learning.algorithms import OffPolicyAlgorithm, PPO, TD3
from .learning.buffers import PerWorkerRolloutBuffer, SharedPrioritizedReplayBuffer
from .learning.policies import ActorCriticPolicyV2, CNNFeatureExtractor, TD3Policy
from .learning.utils import NormalActionNoise, VectorizedActionNoise

__version__ = "1.0.0"
__all__ = [
    "PPO",
    "TD3",
    "OffPolicyAlgorithm",
    "ActorCriticPolicyV2",
    "TD3Policy",
    "CNNFeatureExtractor",
    "StandardEnvAdapter",
    "PerWorkerRolloutBuffer",
    "SharedPrioritizedReplayBuffer",
    "NormalActionNoise",
    "VectorizedActionNoise",
]
