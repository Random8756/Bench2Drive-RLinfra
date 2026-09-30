"""Actor-Critic policy V2 used by PPO / A2C learners.

Deeper actor/critic heads with optional LayerNorm.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from .base_policy import BasePolicy
from .feature_extractor import BaseFeaturesExtractor
from .feature_extractor_v2 import CombinedExtractorV2
from ..utils.distributions import (
    Distribution,
    BetaDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    make_proba_distribution,
)
from ..utils.value_repr import build_value_support, logits_to_scalar

__layer__ = (4, "Algorithm")


def create_mlp_v2(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    use_layer_norm: bool = False,
    squash_output: bool = False,
) -> nn.Sequential:
    """Build an MLP with optional LayerNorm after each hidden layer."""
    layers: list = []
    prev_dim = input_dim

    for hidden_dim in net_arch:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    if squash_output:
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


class ActorCriticPolicyV2(BasePolicy):
    """
    Actor-Critic policy V2 with deeper heads and LayerNorm support.

    Uses configurable LayerNorm policy/value heads and CombinedExtractorV2
    for BEV + scalar feature extraction.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: callable,
        net_arch: Optional[Dict[str, List[int]]] = None,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractorV2,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        value_head_type: str = 'scalar',
        value_num_bins: int = 129,
        value_support_min: float = -7.0,
        value_support_max: float = 7.0,
        value_transform: str = 'identity',
        action_distribution: str = 'auto',
        beta_min_a_b_value: float = 1.0,
        beta_epsilon: float = 1e-6,
        beta_deterministic_action: str = 'mean',
        use_layer_norm_policy_head: bool = True,
        ortho_init: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lr_schedule = lr_schedule
        self.net_arch = net_arch or {'pi': [256, 256], 'vf': [256, 256]}
        self.value_head_type = value_head_type
        self.value_num_bins = int(value_num_bins)
        self.value_support_min = float(value_support_min)
        self.value_support_max = float(value_support_max)
        self.value_transform = value_transform
        self.action_distribution = str(action_distribution).lower()
        self.beta_min_a_b_value = float(beta_min_a_b_value)
        self.beta_epsilon = float(beta_epsilon)
        self.beta_deterministic_action = str(beta_deterministic_action).lower()
        self.use_layer_norm_policy_head = use_layer_norm_policy_head
        self.ortho_init = ortho_init
        if self.value_head_type not in ('scalar', 'categorical'):
            raise ValueError(
                f"Unsupported value_head_type: {self.value_head_type}"
            )
        if self.value_transform not in ('identity', 'symlog'):
            raise ValueError(
                f"Unsupported value_transform: {self.value_transform}"
            )
        if self.action_distribution not in ('auto', 'categorical', 'gaussian', 'beta'):
            raise ValueError(
                f"Unsupported action_distribution: {self.action_distribution}"
            )
        if self.beta_min_a_b_value < 0.0:
            raise ValueError(
                f"beta_min_a_b_value must be >= 0, got {self.beta_min_a_b_value}"
            )
        if not 0.0 < self.beta_epsilon < 0.5:
            raise ValueError(
                f"beta_epsilon must be in (0, 0.5), got {self.beta_epsilon}"
            )
        if self.beta_deterministic_action not in ('mean', 'mode'):
            raise ValueError(
                "beta_deterministic_action must be 'mean' or 'mode', got "
                f"{self.beta_deterministic_action!r}"
            )
        if self.uses_categorical_value_head and self.value_num_bins < 2:
            raise ValueError(
                f"value_num_bins must be >= 2 for categorical value heads, got {self.value_num_bins}"
            )

        self._build()

    def _build(self) -> None:
        self.features_extractor = self.features_extractor_class(
            self.observation_space,
            **self.features_extractor_kwargs,
        )
        features_dim = self.features_extractor.features_dim

        self.action_dist = make_proba_distribution(
            self.action_space,
            self.action_distribution,
        )

        if isinstance(self.action_space, spaces.Discrete):
            self.action_dim = self.action_space.n
        else:
            self.action_dim = int(np.prod(self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            action_low = torch.as_tensor(
                self.action_space.low, dtype=torch.float32,
            ).reshape(-1)
            action_high = torch.as_tensor(
                self.action_space.high, dtype=torch.float32,
            ).reshape(-1)
            action_scale = action_high - action_low
            if not torch.all(torch.isfinite(action_low)) or not torch.all(torch.isfinite(action_high)):
                if isinstance(self.action_dist, BetaDistribution):
                    raise ValueError("Beta policy requires finite Box action bounds")
            if isinstance(self.action_dist, BetaDistribution) and not torch.all(action_scale > 0):
                raise ValueError("Beta policy requires strictly increasing Box action bounds")
        else:
            action_low = torch.empty(0, dtype=torch.float32)
            action_high = torch.empty(0, dtype=torch.float32)
            action_scale = torch.empty(0, dtype=torch.float32)
        self.register_buffer("action_low", action_low, persistent=False)
        self.register_buffer("action_high", action_high, persistent=False)
        self.register_buffer("action_scale", action_scale, persistent=False)

        pi_arch = self.net_arch.get('pi', [256, 256])
        self.policy_net = create_mlp_v2(
            features_dim,
            pi_arch[-1],
            pi_arch[:-1],
            use_layer_norm=self.use_layer_norm_policy_head,
        )
        self.policy_net_out_dim = pi_arch[-1]

        vf_arch = self.net_arch.get('vf', [256, 256])
        value_output_dim = 1
        if self.uses_categorical_value_head:
            value_output_dim = self.value_num_bins
        self.value_output_dim = int(value_output_dim)
        self.value_net = self._make_value_net(
            features_dim,
            self.value_output_dim,
            vf_arch,
        )
        if self.uses_categorical_value_head:
            support = build_value_support(
                self.value_num_bins,
                self.value_support_min,
                self.value_support_max,
            )
        else:
            support = torch.empty(0, dtype=torch.float32)
        self.register_buffer("value_support", support, persistent=False)

        if isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(
                self.policy_net_out_dim
            )
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                self.policy_net_out_dim
            )
        elif isinstance(self.action_dist, BetaDistribution):
            self.alpha_net, self.beta_net = self.action_dist.proba_distribution_net(
                self.policy_net_out_dim
            )
        else:
            raise NotImplementedError(f"Unknown distribution: {type(self.action_dist)}")

        if self.ortho_init:
            self._init_weights()

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.lr_schedule(1.0),
            **self.optimizer_kwargs,
        )

    def _make_value_net(
        self,
        features_dim: int,
        value_output_dim: int,
        vf_arch: List[int],
    ) -> nn.Sequential:
        return create_mlp_v2(
            features_dim,
            value_output_dim,
            vf_arch,
            use_layer_norm=self.use_layer_norm_policy_head,
        )

    def _init_weights(self) -> None:
        def ortho_init(module: nn.Module, gain: float = 1.0):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.policy_net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        self.value_net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))

        if hasattr(self, 'action_net'):
            ortho_init(self.action_net, gain=0.01)
        if hasattr(self, 'alpha_net'):
            ortho_init(self.alpha_net, gain=0.01)
        if hasattr(self, 'beta_net'):
            ortho_init(self.beta_net, gain=0.01)

    @property
    def uses_categorical_value_head(self) -> bool:
        """Whether the critic predicts a categorical distribution on support bins."""
        return self.value_head_type == 'categorical'

    @property
    def uses_beta_distribution(self) -> bool:
        return isinstance(self.action_dist, BetaDistribution)

    def _decode_value_outputs(
        self,
        value_outputs: torch.Tensor,
        *,
        keepdim: bool,
    ) -> torch.Tensor:
        """Decode raw critic head outputs into raw scalar value space."""
        if self.uses_categorical_value_head:
            values = logits_to_scalar(
                value_outputs,
                self.value_support,
                self.value_transform,
            )
            return values.unsqueeze(-1) if keepdim else values
        return value_outputs if keepdim else value_outputs.flatten()

    def _forward_value_outputs(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def decode_value_logits(
        self,
        logits: torch.Tensor,
        *,
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Decode categorical value logits into raw scalar value space."""
        if not self.uses_categorical_value_head:
            raise RuntimeError("decode_value_logits() is only available for categorical value heads")
        return self._decode_value_outputs(logits, keepdim=keepdim)

    def _get_action_dist_and_params(
        self,
        latent_pi: torch.Tensor,
    ) -> Tuple[Distribution, Optional[torch.Tensor]]:
        if isinstance(self.action_dist, CategoricalDistribution):
            logits = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(logits), None
        if isinstance(self.action_dist, DiagGaussianDistribution):
            mean = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(mean, self.log_std), None
        if isinstance(self.action_dist, BetaDistribution):
            alpha = F.softplus(self.alpha_net(latent_pi)) + self.beta_min_a_b_value
            beta = F.softplus(self.beta_net(latent_pi)) + self.beta_min_a_b_value
            params = torch.stack((alpha, beta), dim=1)
            return self.action_dist.proba_distribution(alpha, beta), params
        raise NotImplementedError(f"Unknown distribution: {type(self.action_dist)}")

    def _get_action_dist(self, latent_pi: torch.Tensor) -> Distribution:
        distribution, _ = self._get_action_dist_and_params(latent_pi)
        return distribution

    def _unit_to_env_action(self, unit_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + unit_action * self.action_scale

    def _env_to_unit_action(self, env_action: torch.Tensor) -> torch.Tensor:
        flat_action = env_action.reshape(-1, self.action_dim)
        unit_action = (flat_action - self.action_low) / self.action_scale
        return unit_action.clamp(self.beta_epsilon, 1.0 - self.beta_epsilon)

    def _distribution_log_prob(
        self,
        distribution: Distribution,
        actions: torch.Tensor,
        *,
        actions_are_unit: bool = False,
    ) -> torch.Tensor:
        if self.uses_beta_distribution:
            unit_actions = actions if actions_are_unit else self._env_to_unit_action(actions)
            unit_actions = unit_actions.clamp(self.beta_epsilon, 1.0 - self.beta_epsilon)
            return distribution.log_prob(unit_actions) - torch.log(self.action_scale).sum()
        return distribution.log_prob(actions)

    def _distribution_entropy(self, distribution: Distribution) -> torch.Tensor:
        entropy = distribution.entropy()
        if self.uses_beta_distribution:
            entropy = entropy + torch.log(self.action_scale).sum()
        return entropy

    def _actions_from_distribution(
        self,
        distribution: Distribution,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution_actions = self._select_distribution_actions(
            distribution, deterministic,
        )
        log_prob = self._distribution_log_prob(
            distribution,
            distribution_actions,
            actions_are_unit=self.uses_beta_distribution,
        )
        if self.uses_beta_distribution:
            env_actions = self._unit_to_env_action(distribution_actions)
        else:
            env_actions = distribution_actions
        if isinstance(self.action_space, spaces.Box):
            env_actions = env_actions.reshape((-1,) + self.action_space.shape)
        return env_actions, log_prob

    def _select_distribution_actions(
        self,
        distribution: Distribution,
        deterministic: bool,
    ) -> torch.Tensor:
        if not deterministic:
            return distribution.sample()
        if self.uses_beta_distribution and self.beta_deterministic_action == 'mean':
            return self.action_dist.mean()
        return distribution.mode()

    def forward_with_action_dist_params(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(obs)

        value_outputs = self._forward_value_outputs(features)
        values = self._decode_value_outputs(value_outputs, keepdim=True)

        latent_pi = self.policy_net(features)
        distribution, action_dist_params = self._get_action_dist_and_params(latent_pi)
        actions, log_prob = self._actions_from_distribution(distribution, deterministic)
        return actions, values, log_prob, action_dist_params

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, values, log_prob, _ = self.forward_with_action_dist_params(
            obs, deterministic=deterministic,
        )
        return actions, values, log_prob

    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        features = self.extract_features(observation)
        latent_pi = self.policy_net(features)
        distribution = self._get_action_dist(latent_pi)
        distribution_actions = self._select_distribution_actions(
            distribution, deterministic,
        )
        actions = (
            self._unit_to_env_action(distribution_actions)
            if self.uses_beta_distribution
            else distribution_actions
        )
        if isinstance(self.action_space, spaces.Box):
            actions = actions.reshape((-1,) + self.action_space.shape)
        return actions

    def _evaluate_actions_impl(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(obs)
        value_outputs = self._forward_value_outputs(features)
        values = self._decode_value_outputs(value_outputs, keepdim=False)
        latent_pi = self.policy_net(features)
        distribution, action_dist_params = self._get_action_dist_and_params(latent_pi)
        log_prob = self._distribution_log_prob(distribution, actions)
        entropy = self._distribution_entropy(distribution)
        return values.flatten(), value_outputs, log_prob, entropy, action_dist_params

    def evaluate_actions(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, _, log_prob, entropy, _ = self._evaluate_actions_impl(obs, actions)
        return values, log_prob, entropy

    def evaluate_actions_with_dist_params(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        values, _, log_prob, entropy, action_dist_params = self._evaluate_actions_impl(
            obs, actions,
        )
        return values, log_prob, entropy, action_dist_params

    def evaluate_actions_with_value_logits(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and return both raw scalar values and categorical value logits."""
        if not self.uses_categorical_value_head:
            raise RuntimeError(
                "evaluate_actions_with_value_logits() requires value_head_type='categorical'"
            )

        values, value_logits, log_prob, entropy, _ = self._evaluate_actions_impl(
            obs, actions,
        )
        return values, value_logits, log_prob, entropy

    def evaluate_actions_with_value_logits_and_dist_params(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        if not self.uses_categorical_value_head:
            raise RuntimeError(
                "evaluate_actions_with_value_logits_and_dist_params() requires "
                "value_head_type='categorical'"
            )
        return self._evaluate_actions_impl(obs, actions)

    def predict_values(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        features = self.extract_features(obs)
        value_outputs = self._forward_value_outputs(features)
        return self._decode_value_outputs(value_outputs, keepdim=True)

    def predict_value_logits(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Predict categorical value logits for observations."""
        if not self.uses_categorical_value_head:
            raise RuntimeError(
                "predict_value_logits() requires value_head_type='categorical'"
        )
        features = self.extract_features(obs)
        return self._forward_value_outputs(features)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({
            'net_arch': self.net_arch,
            'value_head_type': self.value_head_type,
            'value_num_bins': self.value_num_bins,
            'value_support_min': self.value_support_min,
            'value_support_max': self.value_support_max,
            'value_transform': self.value_transform,
            'action_distribution': self.action_distribution,
            'beta_min_a_b_value': self.beta_min_a_b_value,
            'beta_epsilon': self.beta_epsilon,
            'beta_deterministic_action': self.beta_deterministic_action,
            'use_layer_norm_policy_head': self.use_layer_norm_policy_head,
        })
        return data

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=strict)
