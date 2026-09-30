"""SAC policy: stochastic actor with twin critics.

Architecture:
- Stochastic actor: ``obs -> features -> (mean, log_std)`` -> squashed
  Gaussian.
- Twin critics: ``(obs, action) -> (Q1, Q2)``.
- Target critic networks for the Bellman update.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

__layer__ = (4, "Algorithm")
import torch.nn as nn
from gymnasium import spaces

from .base_policy import BasePolicy, create_mlp
from .feature_extractor import BaseFeaturesExtractor, CNNFeatureExtractor
from .moe import MoEMLP
from ..utils.distributions import SquashedDiagGaussianDistribution

# Clip log_std to prevent numerical instability
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACActorNetwork(nn.Module):
    """
    Stochastic actor network for SAC.
    
    Outputs mean and log_std for a squashed Gaussian distribution.
    Actions are squashed to [-1, 1] via tanh.
    """
    
    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        log_std_init: float = -3.0,
        # Full-trunk MoE
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_noisy_gating: bool = True,
    ):
        super().__init__()

        self.features_dim = features_dim
        self.action_dim = action_dim
        self.log_std_init = log_std_init
        self.use_moe = use_moe

        last_layer_dim = net_arch[-1] if net_arch else features_dim

        if use_moe:
            # Full-trunk MoE: every expert is an independent MLP with the
            # full ``net_arch`` topology; the MoE output is then activated
            # to mirror the original ``create_mlp + outer activation``
            # behaviour before feeding into mu / log_std heads.
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
        else:
            # Shared latent network (original behaviour)
            self.latent_net = create_mlp(
                features_dim,
                last_layer_dim,
                net_arch[:-1] if len(net_arch) > 1 else [],
                activation_fn,
                squash_output=False,
            )
            if net_arch:
                self.latent_net = nn.Sequential(
                    self.latent_net,
                    activation_fn()
                )

        # Mean and log_std heads
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)

        # Action distribution
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
    
    def get_action_dist_params(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get distribution parameters from features.
        
        Args:
            features: Feature tensor from feature extractor.
            
        Returns:
            Tuple of (mean, log_std).
        """
        latent = self.latent_net(features)
        mean = self.mu(latent)
        log_std = self.log_std(latent)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def forward(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action from features.
        
        Args:
            features: Feature tensor.
            deterministic: Whether to use deterministic action (mean).
            
        Returns:
            Action tensor in [-1, 1].
        """
        mean, log_std = self.get_action_dist_params(features)
        return self.action_dist.actions_from_params(mean, log_std, deterministic)
    
    def action_log_prob(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            features: Feature tensor.
            
        Returns:
            Tuple of (action, log_prob).
        """
        mean, log_std = self.get_action_dist_params(features)
        return self.action_dist.log_prob_from_params(mean, log_std)


class SACCritic(nn.Module):
    """
    Twin Q-networks for SAC.
    
    Takes (state features, action) and outputs Q-values.
    Uses multiple critics to mitigate overestimation.
    
    Supports optional **distributional** mode (two-hot encoding):
    when ``use_distributional=True``, each Q-network outputs ``num_bins``
    logits instead of a single scalar.
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

        # Create multiple Q-networks (shared MLP or MoE per critic)
        if use_moe:
            self.q_networks = nn.ModuleList([
                MoEMLP(
                    input_dim=features_dim + action_dim,
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
                    features_dim + action_dim,
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
    ) -> List[torch.Tensor]:
        """Return raw Q-network outputs for all critics.
        
        * Non-distributional: each tensor has shape ``(batch, 1)``.
        * Distributional: each tensor has shape ``(batch, num_bins)`` (logits).
        """
        qvalue_input = torch.cat([features, actions], dim=1)
        return [q_net(qvalue_input) for q_net in self.q_networks]
    
    def q_values(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        for_actor: bool = False,
    ) -> List[torch.Tensor]:
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
            return [decode_fn(q) for q in raw]
        return raw
    
    def q1_forward(self, features: torch.Tensor, actions: torch.Tensor, for_actor: bool = False) -> torch.Tensor:
        """Scalar Q-value from the first critic only.

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


class SACPolicy(BasePolicy):
    """
    SAC Policy with stochastic actor and twin critics.
    
    Components:
    - actor: Stochastic policy network (squashed Gaussian)
    - critic: Twin Q-networks (Q1, Q2)
    - critic_target: Target critic networks
    
    The target networks are soft-updated during training.
    """
    
    actor: SACActorNetwork
    critic: SACCritic
    critic_target: SACCritic
    
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
        log_std_init: float = -3.0,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
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
        Initialize SAC policy.
        
        Args:
            observation_space: Observation space.
            action_space: Action space (must be Box for continuous control).
            lr_schedule: Learning rate schedule function.
            net_arch: Network architecture {'pi': [256, 256], 'qf': [256, 256]}.
            activation_fn: Activation function class.
            features_extractor_class: Feature extractor class.
            features_extractor_kwargs: Feature extractor arguments.
            n_critics: Number of critic networks (default 2 for SAC).
            log_std_init: Initial value for log standard deviation.
            optimizer_class: Optimizer class.
            optimizer_kwargs: Optimizer arguments.
            share_features_extractor: Whether actor and critic share features extractor.
            use_distributional: Use two-hot distributional Q-values instead of scalar.
            num_bins: Number of bins for distributional mode.
            v_min: Minimum Q-value (raw, before symlog).
            v_max: Maximum Q-value (raw, before symlog).
            use_symlog: Apply symlog transform before binning.
            moe_actor: Optional dict enabling Mixture-of-Experts on the actor
                trunk.  Keys: ``enabled`` (bool), ``num_experts`` (int),
                ``top_k`` (int), ``noisy_gating`` (bool).  When ``enabled``
                is falsy the actor keeps its original shared MLP trunk.
            moe_critic: Same schema as ``moe_actor`` but applied independently
                to every Q-network inside the twin critic.
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
        
        # SAC only supports continuous actions
        assert isinstance(action_space, spaces.Box), "SAC only supports Box action spaces"
        
        self.lr_schedule = lr_schedule
        self.net_arch = net_arch or {'pi': [256, 256], 'qf': [256, 256]}
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.log_std_init = log_std_init
        self.share_features_extractor = share_features_extractor
        
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
        # Create feature extractor for actor
        self.features_extractor = self.features_extractor_class(
            self.observation_space,
            **self.features_extractor_kwargs,
        )
        features_dim = self.features_extractor.features_dim
        
        # Action dimension
        action_dim = int(np.prod(self.action_space.shape))
        
        # Actor network (stochastic, optionally with MoE trunk)
        self.actor = SACActorNetwork(
            features_dim=features_dim,
            action_dim=action_dim,
            net_arch=self.net_arch.get('pi', [256, 256]),
            activation_fn=self.activation_fn,
            log_std_init=self.log_std_init,
            use_moe=self.moe_actor['enabled'],
            moe_num_experts=self.moe_actor['num_experts'],
            moe_top_k=self.moe_actor['top_k'],
            moe_noisy_gating=self.moe_actor['noisy_gating'],
        )

        # Twin critics (optionally with MoE Q-networks)
        self.critic = SACCritic(
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
        
        # Target critic (deep copy)
        self.critic_target = deepcopy(self.critic)
        
        # Freeze target networks
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=self.lr_schedule(1),
            **self.optimizer_kwargs,
        )
        
        # Critic optimizer includes feature extractor parameters
        critic_params = list(self.critic.parameters())
        if not self.share_features_extractor:
            critic_params += list(self.features_extractor.parameters())
        
        self.critic_optimizer = self.optimizer_class(
            critic_params,
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
        self.actor.latent_net.apply(lambda m: ortho_init(m, gain=np.sqrt(2)))
        ortho_init(self.actor.mu, gain=np.sqrt(2))
        ortho_init(self.actor.log_std, gain=np.sqrt(2))
        
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
        
        Args:
            obs: Observation tensor.
            deterministic: Whether to use deterministic action (mean).
            
        Returns:
            Action tensor in [-1, 1].
        """
        features = self.extract_features(obs)
        return self.actor(features, deterministic)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Internal prediction (required by BasePolicy)."""
        return self.forward(observation, deterministic)
    
    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            obs: Observation tensor.
            
        Returns:
            Tuple of (action, log_prob).
        """
        features = self.extract_features(obs)
        return self.actor.action_log_prob(features)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Evaluate Q-values for given observation-action pairs.
        
        Args:
            obs: Observations.
            actions: Actions.
            
        Returns:
            List of Q-values from each critic.
        """
        features = self.extract_features(obs)
        return self.critic(features, actions)
    
    def set_training_mode(self, mode: bool) -> None:
        """Set training mode for all networks."""
        self.actor.train(mode)
        self.critic.train(mode)
        self.features_extractor.train(mode)
        # Target networks always in eval mode
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
            'log_std_init': self.log_std_init,
            'share_features_extractor': self.share_features_extractor,
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
