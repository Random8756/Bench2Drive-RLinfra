"""Layer 4 — Algorithm.

Algorithm abstraction. Groups learner implementations, buffers, policies, and
training utilities. Algorithms interact with the parallel environment pool
through an environment adapter.

Layer entries and implementation modules
----------------------------------------
Algorithms (Learner)
    L4_PPO / L4_A2C                ← b2d_rlinfra.learning.algorithms.{ppo, a2c}
    L4_SAC / L4_TD3                ← b2d_rlinfra.learning.algorithms.{sac, td3}
    L4_OnPolicyAlgorithm           ← b2d_rlinfra.learning.algorithms.on_policy_algorithm
    L4_OffPolicyAlgorithm          ← b2d_rlinfra.learning.algorithms.off_policy_algorithm

Buffers
    L4_RolloutBuffer               ← b2d_rlinfra.learning.buffers.per_worker_buffer
    L4_ReplayBuffer                ← b2d_rlinfra.learning.buffers.shared_buffer

Environment Adapter
    L4_EnvAdapter                  ← b2d_rlinfra.learning.adapters.standard_adapter

Policies & Feature Extractors
    L4_ActorCriticPolicy           ← b2d_rlinfra.learning.policies
    L4_CNNFeatureExtractor         ← b2d_rlinfra.learning.policies

Configuration
    L4_load_config                 ← b2d_rlinfra.learning.utils.config
"""

from __future__ import annotations

__layer__ = (4, "Algorithm")

from b2d_rlinfra.learning.algorithms import (
    BaseAlgorithm as L4_BaseAlgorithm,
    OnPolicyAlgorithm as L4_OnPolicyAlgorithm,
    OffPolicyAlgorithm as L4_OffPolicyAlgorithm,
    PPO as L4_PPO,
    A2C as L4_A2C,
    SAC as L4_SAC,
    TD3 as L4_TD3,
)

from b2d_rlinfra.learning.buffers import (
    PerWorkerRolloutBuffer as L4_RolloutBuffer,
    SharedPrioritizedReplayBuffer as L4_ReplayBuffer,
)

from b2d_rlinfra.learning.adapters import StandardEnvAdapter as L4_EnvAdapter

from b2d_rlinfra.learning.policies import (
    ActorCriticPolicyV2 as L4_ActorCriticPolicy,
    TD3Policy as L4_TD3Policy,
    SACPolicy as L4_SACPolicy,
    CNNFeatureExtractor as L4_CNNFeatureExtractor,
    BaseFeaturesExtractor as L4_BaseFeaturesExtractor,
)

from b2d_rlinfra.learning.utils.noise import (
    NormalActionNoise as L4_NormalActionNoise,
    VectorizedActionNoise as L4_VectorizedActionNoise,
)

from b2d_rlinfra.learning.utils.config import (
    load_config as L4_load_config,
    load_env_config as L4_load_env_config,
    Config as L4_Config,
)

from b2d_rlinfra.learning.utils.callbacks import (
    BaseCallback as L4_BaseCallback,
    CallbackList as L4_CallbackList,
    CheckpointCallback as L4_CheckpointCallback,
)


__all__ = [
    "L4_BaseAlgorithm",
    "L4_OnPolicyAlgorithm",
    "L4_OffPolicyAlgorithm",
    "L4_PPO",
    "L4_A2C",
    "L4_SAC",
    "L4_TD3",
    "L4_RolloutBuffer",
    "L4_ReplayBuffer",
    "L4_EnvAdapter",
    "L4_ActorCriticPolicy",
    "L4_TD3Policy",
    "L4_SACPolicy",
    "L4_CNNFeatureExtractor",
    "L4_BaseFeaturesExtractor",
    "L4_NormalActionNoise",
    "L4_VectorizedActionNoise",
    "L4_load_config",
    "L4_load_env_config",
    "L4_Config",
    "L4_BaseCallback",
    "L4_CallbackList",
    "L4_CheckpointCallback",
]
