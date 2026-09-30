"""TD3 policy: deterministic actor with twin critics.

Architecture:
- Deterministic actor: ``obs -> features -> action`` (tanh squashed to
  ``[-1, 1]``).
- Twin critics: ``(obs, action) -> (Q1, Q2)``.
- Target networks for both actor and critic.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch

__layer__ = (4, "Algorithm")
import torch.nn as nn
from gymnasium import spaces

from .base_policy import BasePolicy, create_mlp
from .feature_extractor import BaseFeaturesExtractor, CNNFeatureExtractor
from .moe import MoEMLP


class Actor(nn.Module):
    """
    Deterministic actor network for TD3.
    
    Outputs actions in [-1, 1] range via tanh activation.
    """
    
    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        # Full-trunk MoE
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_noisy_gating: bool = True,
    ):
        super().__init__()

        self.use_moe = use_moe

        if use_moe:
            last_layer_dim = net_arch[-1] if net_arch else features_dim
            self.latent_net = MoEMLP(
                input_dim=features_dim,
                output_dim=last_layer_dim,
                hidden=net_arch[:-1] if len(net_arch) > 1 else [],
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                activation_fn=activation_fn,
                noisy_gating=moe_noisy_gating,
                output_activation=activation_fn,
            )
            self.action_head = nn.Sequential(
                nn.Linear(last_layer_dim, action_dim),
                nn.Tanh(),
            )
        else:
            # Keep the legacy module name so existing non-MoE TD3 checkpoints
            # continue to load with actor.net.* / actor_target.net.* keys.
            self.net = create_mlp(
                features_dim,
                action_dim,
                net_arch,
                activation_fn,
                squash_output=True,  # tanh for bounded actions
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Output deterministic action in [-1, 1]."""
        if self.use_moe:
            return self.action_head(self.latent_net(features))
        return self.net(features)


class ContinuousCritic(nn.Module):
    """
    Twin Q-networks for TD3.
    
    Takes (state features, action) and outputs Q-values.
    Uses multiple critics to mitigate overestimation.
    
    Supports optional **distributional** mode (two-hot encoding):
    when ``use_distributional=True``, each Q-network outputs ``num_bins``
    logits instead of a single scalar.  Use :meth:`q_values` or
    :meth:`q1_forward` to get decoded scalar Q-values.
    """
    
    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        # Two-hot value representation
        use_distributional: bool = False,
        num_bins: int = 255,
        v_min: float = -300.0,
        v_max: float = 800.0,
        use_symlog: bool = True,
        # Full-trunk MoE for every Q-network
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_noisy_gating: bool = True,
    ):
        super().__init__()
        
        self.n_critics = n_critics
        self.use_distributional = use_distributional
        self.use_moe = use_moe
        
        # Output dimension: scalar Q or bin logits
        output_dim = num_bins if use_distributional else 1
        
        # Create multiple Q-networks
        if use_moe:
            self.q_networks = nn.ModuleList([
                MoEMLP(
                    input_dim=features_dim + action_dim,  # concat state and action
                    output_dim=output_dim,
                    hidden=net_arch,
                    num_experts=moe_num_experts,
                    top_k=moe_top_k,
                    activation_fn=activation_fn,
                    noisy_gating=moe_noisy_gating,
                    output_activation=None,
                )
                for _ in range(n_critics)
            ])
        else:
            self.q_networks = nn.ModuleList([
                create_mlp(
                    features_dim + action_dim,  # concat state and action
                    output_dim,
                    net_arch,
                    activation_fn,
                    squash_output=False,
                )
                for _ in range(n_critics)
            ])
        
        # Distributional helper (two-hot encode / decode / loss)
        self.distributional = None
        if use_distributional:
            from ..utils.distributional import TwoHotDistributional
            self.distributional = TwoHotDistributional(
                num_bins=num_bins,
                v_min=v_min,
                v_max=v_max,
                use_symlog=use_symlog,
            )
    
    def _apply(self, fn):
        """Override to move distributional bin_centers along with model parameters."""
        super()._apply(fn)
        if self.distributional is not None:
            self.distributional.bin_centers = fn(self.distributional.bin_centers)
        return self
    
    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return raw Q-network outputs for all critics.
        
        * Non-distributional: each tensor has shape ``(batch, 1)``.
        * Distributional: each tensor has shape ``(batch, num_bins)`` (logits).
        """
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)
    
    def q_values(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        for_actor: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Return **scalar** Q-values ``(batch, 1)`` for all critics.
        
        In distributional mode the logits are decoded via expected value.

        Args:
            for_actor: If True and distributional mode is active, return
                expected values in **transformed (symlog) space** without
                applying ``symexp``.  This keeps actor gradients stable
                (only ``softmax → weighted-sum``, no exponential).
        """
        raw = self.forward(features, actions)
        if self.use_distributional:
            decode_fn = self.distributional.decode_for_actor if for_actor else self.distributional.decode
            return tuple(decode_fn(q) for q in raw)
        return raw
    
    def q1_forward(self, features: torch.Tensor, actions: torch.Tensor, for_actor: bool = False) -> torch.Tensor:
        """Scalar Q-value from the first critic only (for actor update).

        Args:
            for_actor: If True and distributional mode is active, return
                expected value in transformed (symlog) space without
                ``symexp``, for stable actor gradients.
        """
        qvalue_input = torch.cat([features, actions], dim=1)
        raw = self.q_networks[0](qvalue_input)
        if self.use_distributional:
            if for_actor:
                return self.distributional.decode_for_actor(raw)
            return self.distributional.decode(raw)
        return raw


class TD3Policy(BasePolicy):
    """
    TD3 Policy with deterministic actor and twin critics.
    
    Components:
    - actor: Deterministic policy network
    - critic: Twin Q-networks (Q1, Q2)
    - actor_target: Target actor network
    - critic_target: Target critic network
    
    The target networks are soft-updated during training.
    """
    
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: callable,
        net_arch: Optional[Dict[str, List[int]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CNNFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # Two-hot value representation
        use_distributional: bool = False,
        num_bins: int = 255,
        v_min: float = -300.0,
        v_max: float = 800.0,
        use_symlog: bool = True,
        # Independently configurable actor and critic MoE
        moe_actor: Optional[Dict[str, Any]] = None,
        moe_critic: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TD3 policy.
        
        Args:
            observation_space: Observation space.
            action_space: Action space (must be Box for continuous control).
            lr_schedule: Learning rate schedule function.
            net_arch: Network architecture {'pi': [256, 256], 'qf': [256, 256]}.
            activation_fn: Activation function class.
            features_extractor_class: Feature extractor class.
            features_extractor_kwargs: Feature extractor arguments.
            n_critics: Number of critic networks (default 2 for TD3).
            optimizer_class: Optimizer class.
            optimizer_kwargs: Optimizer arguments.
            use_distributional: Use two-hot distributional Q-values instead of scalar.
            num_bins: Number of bins for distributional mode.
            v_min: Minimum Q-value (raw, before symlog).
            v_max: Maximum Q-value (raw, before symlog).
            use_symlog: Apply symlog transform before binning.
        """
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class,
            optimizer_kwargs,
            squash_output=True,
        )
        
        # TD3 only supports continuous actions
        assert isinstance(action_space, spaces.Box), "TD3 only supports Box action spaces"
        
        self.lr_schedule = lr_schedule
        self.net_arch = net_arch or {'pi': [256, 256], 'qf': [256, 256]}
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        
        # Distributional parameters
        self.use_distributional = use_distributional
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.use_symlog = use_symlog

        # Mixture-of-Experts config (normalised to dicts for clean save/load).
        self.moe_actor = self._normalise_moe_config(moe_actor)
        self.moe_critic = self._normalise_moe_config(moe_critic)
        
        self._build()

    @staticmethod
    def _normalise_moe_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Canonicalise a MoE sub-config; missing keys fall back to safe defaults."""
        cfg = dict(cfg) if cfg else {}
        return {
            'enabled': bool(cfg.get('enabled', False)),
            'num_experts': int(cfg.get('num_experts', 4)),
            'top_k': int(cfg.get('top_k', 2)),
            'noisy_gating': bool(cfg.get('noisy_gating', True)),
        }
    
    def _build(self) -> None:
        """Build actor, critic, and target networks."""
        # Create feature extractor
        self.features_extractor = self.features_extractor_class(
            self.observation_space,
            **self.features_extractor_kwargs,
        )
        features_dim = self.features_extractor.features_dim
        
        # Action dimension
        action_dim = int(np.prod(self.action_space.shape))
        
        # Actor network
        self.actor = Actor(
            features_dim=features_dim,
            action_dim=action_dim,
            net_arch=self.net_arch.get('pi', [256, 256]),
            activation_fn=self.activation_fn,
            use_moe=self.moe_actor['enabled'],
            moe_num_experts=self.moe_actor['num_experts'],
            moe_top_k=self.moe_actor['top_k'],
            moe_noisy_gating=self.moe_actor['noisy_gating'],
        )
        
        # Twin critics
        self.critic = ContinuousCritic(
            features_dim=features_dim,
            action_dim=action_dim,
            net_arch=self.net_arch.get('qf', [256, 256]),
            activation_fn=self.activation_fn,
            n_critics=self.n_critics,
            use_distributional=self.use_distributional,
            num_bins=self.num_bins,
            v_min=self.v_min,
            v_max=self.v_max,
            use_symlog=self.use_symlog,
            use_moe=self.moe_critic['enabled'],
            moe_num_experts=self.moe_critic['num_experts'],
            moe_top_k=self.moe_critic['top_k'],
            moe_noisy_gating=self.moe_critic['noisy_gating'],
        )
        
        # Initialize weights with orthogonal initialization
        self._init_weights()
        
        # Target networks (deep copy)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        
        # Freeze target networks (no gradient computation)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=self.lr_schedule(1),
            **self.optimizer_kwargs,
        )
        self.critic_optimizer = self.optimizer_class(
            list(self.critic.parameters()) + list(self.features_extractor.parameters()),
            lr=self.lr_schedule(1),
            **self.optimizer_kwargs,
        )
    
    def _init_weights(self) -> None:
        """
        Initialize weights with orthogonal initialization.
        
        Orthogonal initialization helps:
        - Prevent vanishing/exploding gradients
        - Speed up convergence
        - Improve training stability
        """
        def ortho_init(module: nn.Module, gain: float = 1.0):
            """Orthogonal initialization for linear layers."""
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize actor network with sqrt(2) gain
        if self.moe_actor['enabled']:
            self.actor.latent_net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
            self.actor.action_head.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        else:
            self.actor.net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        
        # Initialize critic networks with sqrt(2) gain
        for q_net in self.critic.q_networks:
            q_net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        
        # Initialize feature extractor CNN layers
        for module in self.features_extractor.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action for observation.
        
        TD3 is always deterministic. The deterministic parameter is ignored.
        
        Args:
            obs: Observation tensor.
            deterministic: Ignored (TD3 is deterministic).
            
        Returns:
            Action tensor in [-1, 1].
        """
        features = self.extract_features(obs)
        return self.actor(features)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Internal prediction (required by BasePolicy)."""
        return self.forward(observation, deterministic)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate Q-values for given observation-action pairs.
        
        Args:
            obs: Observations.
            actions: Actions.
            
        Returns:
            Tuple of (Q1, Q2) values.
        """
        features = self.extract_features(obs)
        return self.critic(features, actions)
    
    def set_training_mode(self, mode: bool) -> None:
        """Set training mode for all networks."""
        self.actor.train(mode)
        self.critic.train(mode)
        self.features_extractor.train(mode)
        # Target networks always in eval mode
        self.actor_target.eval()
        self.critic_target.eval()
        self.training = mode
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get parameters needed to recreate the policy."""
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'net_arch': self.net_arch,
            'activation_fn': self.activation_fn,
            'n_critics': self.n_critics,
            'use_distributional': self.use_distributional,
            'num_bins': self.num_bins,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'use_symlog': self.use_symlog,
            'moe_actor': dict(self.moe_actor),
            'moe_critic': dict(self.moe_critic),
        }

    # MoE diagnostics
    def actor_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Return the actor's most recent MoE load-balancing loss (or ``None``)."""
        if self.moe_actor['enabled'] and isinstance(self.actor.latent_net, MoEMLP):
            return self.actor.latent_net.last_aux_loss
        return None

    def critic_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Return the summed MoE load-balancing loss across all Q-networks."""
        if not self.moe_critic['enabled']:
            return None
        total: Optional[torch.Tensor] = None
        for q_net in self.critic.q_networks:
            if isinstance(q_net, MoEMLP):
                total = q_net.last_aux_loss if total is None else total + q_net.last_aux_loss
        return total

    def actor_moe_stats(self) -> Optional[Dict[str, Any]]:
        """Return usage / entropy stats dict for the actor MoE, or ``None``."""
        if self.moe_actor['enabled'] and isinstance(self.actor.latent_net, MoEMLP):
            return dict(self.actor.latent_net.last_gate_stats)
        return None

    def critic_moe_stats(self) -> Optional[Dict[str, Any]]:
        """Return averaged usage / entropy stats across the twin Q-nets."""
        if not self.moe_critic['enabled']:
            return None
        usages: List[List[float]] = []
        entropies: List[float] = []
        for q_net in self.critic.q_networks:
            if isinstance(q_net, MoEMLP):
                usages.append(list(q_net.last_gate_stats.get('usage', [])))
                entropies.append(float(q_net.last_gate_stats.get('gate_entropy', 0.0)))
        if not usages:
            return None
        n_exp = len(usages[0])
        avg_usage = [float(np.mean([u[i] for u in usages])) for i in range(n_exp)]
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0
        return {'usage': avg_usage, 'gate_entropy': avg_entropy}
    
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale action from [-1, 1] to action space bounds.
        
        Args:
            action: Action in [-1, 1].
            
        Returns:
            Scaled action in [low, high].
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)
    
    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Unscale action from action space bounds to [-1, 1].
        
        Args:
            scaled_action: Action in [low, high].
            
        Returns:
            Action in [-1, 1].
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((scaled_action - low) / (high - low)) - 1.0
