from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import torch

from b2d_rlinfra.learning.algorithms import A2C, PPO, SAC, TD3
from b2d_rlinfra.learning.policies.actor_critic_policy_v2 import ActorCriticPolicyV2
from b2d_rlinfra.learning.policies.feature_extractor import CNNFeatureExtractor, CombinedExtractor
from b2d_rlinfra.learning.policies.feature_extractor_v2 import CombinedExtractorV2
from b2d_rlinfra.learning.policies.sac_policy import SACPolicy
from b2d_rlinfra.learning.policies.td3_policy import TD3Policy
from b2d_rlinfra.learning.utils.config import Config, load_config
from b2d_rlinfra.learning.utils.noise import NormalActionNoise
from b2d_rlinfra.learning.utils.schedules import linear_schedule

from .space_builder import build_action_space, build_observation_space, normalize_env_config_paths


@dataclass(frozen=True)
class LoadedModelBundle:
    config: Config
    env_config: Dict[str, Any]
    observation_space: Any
    action_space: Any
    model: Any


class _DummyAdapter:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = 1

    def seed(self, seed: int) -> None:
        return None


def _get_feature_extractor(config: Config):
    fe_type = config.policy.features_extractor_type.lower()

    if fe_type == "combined_v2":
        fe_class = CombinedExtractorV2
        fe_kwargs = {
            "features_dim": config.policy.features_dim,
            "use_layer_norm": config.policy.use_layer_norm,
            "cnn_channels": config.policy.cnn_channels,
            "state_neurons": config.policy.state_neurons,
            "fusion_dims": config.policy.fusion_dims,
        }
    elif fe_type == "combined":
        fe_class = CombinedExtractor
        fe_kwargs = {"features_dim": config.policy.features_dim}
    else:
        fe_class = CNNFeatureExtractor
        fe_kwargs = {"features_dim": config.policy.features_dim}

    policy_kwargs = {
        "features_extractor_class": fe_class,
        "features_extractor_kwargs": fe_kwargs,
    }

    algo_name = config.algorithm.name.lower()
    if algo_name in ("ppo", "a2c"):
        policy_kwargs["net_arch"] = {
            "pi": config.policy.net_arch_pi,
            "vf": config.policy.net_arch_vf,
        }
        if fe_type == "combined_v2":
            policy_kwargs["use_layer_norm_policy_head"] = config.policy.use_layer_norm_policy_head
            policy_kwargs["value_head_type"] = config.policy.value_head_type
            policy_kwargs["value_num_bins"] = config.policy.value_num_bins
            policy_kwargs["value_support_min"] = config.policy.value_support_min
            policy_kwargs["value_support_max"] = config.policy.value_support_max
            policy_kwargs["value_transform"] = config.policy.value_transform
            policy_kwargs["action_distribution"] = config.policy.action_distribution
            policy_kwargs["beta_min_a_b_value"] = config.policy.beta_min_a_b_value
            policy_kwargs["beta_epsilon"] = config.policy.beta_epsilon
            policy_kwargs["beta_deterministic_action"] = config.policy.beta_deterministic_action
    else:
        policy_kwargs["net_arch"] = {
            "pi": config.policy.net_arch_pi,
            "qf": config.policy.net_arch_qf,
        }
        if algo_name == "sac":
            policy_kwargs["log_std_init"] = getattr(config.policy, "log_std_init", -3.0)
        if algo_name in ("td3", "sac"):
            # Must mirror runner_utils.get_feature_extractor / training so MoE
            # checkpoints load (latent_net / q_networks use MoEMLP, not plain MLP).
            policy_kwargs["moe_actor"] = {
                "enabled": getattr(config.policy, "moe_actor_enabled", False),
                "num_experts": getattr(config.policy, "moe_actor_num_experts", 4),
                "top_k": getattr(config.policy, "moe_actor_top_k", 2),
                "noisy_gating": getattr(config.policy, "moe_actor_noisy_gating", True),
            }
            policy_kwargs["moe_critic"] = {
                "enabled": getattr(config.policy, "moe_critic_enabled", False),
                "num_experts": getattr(config.policy, "moe_critic_num_experts", 4),
                "top_k": getattr(config.policy, "moe_critic_top_k", 2),
                "noisy_gating": getattr(config.policy, "moe_critic_noisy_gating", True),
            }
        if algo_name in ("td3", "sac") and getattr(config.policy, "use_distributional", False):
            policy_kwargs["use_distributional"] = True
            policy_kwargs["num_bins"] = getattr(config.policy, "num_bins", 255)
            policy_kwargs["v_min"] = getattr(config.policy, "v_min", -300.0)
            policy_kwargs["v_max"] = getattr(config.policy, "v_max", 800.0)
            policy_kwargs["use_symlog"] = getattr(config.policy, "use_symlog", True)

    return fe_type, policy_kwargs


def _get_algorithm_components(config: Config, adapter: _DummyAdapter):
    algo = config.algorithm
    name = algo.name.lower()
    _, policy_kwargs = _get_feature_extractor(config)

    common = dict(
        env=adapter,
        learning_rate=algo.learning_rate,
        gamma=algo.gamma,
        min_ready=1,
        visualizer=None,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=config.training.verbose,
        config=config,
    )

    if name in ("ppo", "a2c"):
        if getattr(algo, "learning_rate_final", None) is not None:
            common["learning_rate"] = linear_schedule(
                float(algo.learning_rate),
                float(algo.learning_rate_final),
            )

        if config.policy.features_extractor_type.lower() != "combined_v2":
            raise ValueError("PPO/A2C checkpoints require policy.feature_extractor.type=combined_v2")
        common["policy"] = ActorCriticPolicyV2

        on_policy = dict(
            n_steps=algo.n_steps,
            gae_lambda=algo.gae_lambda,
            ent_coef=algo.ent_coef,
            ent_coef_final=getattr(algo, "ent_coef_final", None),
            vf_coef=algo.vf_coef,
            max_grad_norm=algo.max_grad_norm,
            normalize_advantage=getattr(algo, "normalize_advantage", True),
            truncation_steps=getattr(algo, "truncation", None),
        )
        if name == "ppo":
            return PPO, {
                **common,
                **on_policy,
                "batch_size": algo.batch_size,
                "n_epochs": algo.n_epochs,
                "clip_range": algo.clip_range,
                "target_kl": algo.target_kl,
            }

        return A2C, {
            **common,
            **on_policy,
        }

    # Evaluation never adds transitions, but off-policy setup still allocates
    # a shared replay buffer. Keep it minimal to avoid exhausting /dev/shm.
    off_policy = dict(
        buffer_size=1,
        learning_starts=algo.learning_starts,
        batch_size=algo.batch_size,
        tau=algo.tau,
        train_freq=algo.train_freq,
        gradient_steps=algo.gradient_steps,
        per_alpha=getattr(algo, "per_alpha", 0.6),
        per_beta=getattr(algo, "per_beta", 0.4),
        per_beta_annealing_steps=1,
        per_min_priority=getattr(algo, "per_min_priority", 1e-6),
        warmup_source=getattr(algo, "warmup_source", "random"),
    )

    bc_params = dict(
        bc_enabled=getattr(algo, "bc_enabled", False),
        bc_lambda_initial=getattr(algo, "bc_lambda_initial", 1.0),
        bc_lambda_decay_steps=getattr(algo, "bc_lambda_decay_steps", 200000),
        bc_lambda_final=getattr(algo, "bc_lambda_final", 0.0),
        bc_prior_action=getattr(algo, "bc_prior_action", [0.6, 0.0]),
        bc_source=getattr(algo, "bc_source", "fixed"),
    )

    if name == "td3":
        action_dim = adapter.action_space.shape[0]
        sigma = algo.exploration_noise
        sigma = sigma * np.ones(action_dim) if isinstance(sigma, (int, float)) else np.array(sigma, dtype=np.float32)
        action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=sigma)
        return TD3, {
            **common,
            **off_policy,
            **bc_params,
            "policy": TD3Policy,
            "policy_delay": algo.policy_delay,
            "target_policy_noise": algo.target_policy_noise,
            "target_noise_clip": algo.target_noise_clip,
            "action_noise": action_noise,
        }

    if name == "sac":
        return SAC, {
            **common,
            **off_policy,
            **bc_params,
            "policy": SACPolicy,
            "ent_coef": algo.ent_coef,
            "target_update_interval": algo.target_update_interval,
            "target_entropy": algo.target_entropy,
        }

    raise ValueError(f"Unknown algorithm: {name}")


def load_model_bundle(
    rl_config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    *,
    device: str = "auto",
) -> LoadedModelBundle:
    rl_config_path = Path(rl_config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    config = load_config(rl_config_path)
    if config.env_config is None:
        raise ValueError(f"RL config does not contain env config: {rl_config_path}")

    env_config = normalize_env_config_paths(config.env_config, base_dir=rl_config_path.parent)
    observation_space = build_observation_space(env_config)
    action_space = build_action_space(env_config)
    adapter = _DummyAdapter(observation_space, action_space)

    algo_class, algo_kwargs = _get_algorithm_components(config, adapter)
    model = algo_class.load(path=checkpoint_path, device=device, **algo_kwargs)
    model.policy.set_training_mode(False)

    return LoadedModelBundle(
        config=config,
        env_config=env_config,
        observation_space=observation_space,
        action_space=action_space,
        model=model,
    )
