"""
Action Distributions for RL Policies.

Provides probability distributions for action sampling.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical, Normal

from gymnasium import spaces



class Distribution(ABC):
    """Abstract base class for action distributions."""
    
    @abstractmethod
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create network layer(s) for the distribution parameters.
        
        Args:
            latent_dim: Input dimension (from policy network).
            
        Returns:
            Network module(s) for distribution parameters.
        """
        raise NotImplementedError
    
    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> 'Distribution':
        """
        Set distribution parameters.
        
        Returns:
            Self with updated parameters.
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of actions.
        
        Args:
            actions: Actions to evaluate.
            
        Returns:
            Log probabilities.
        """
        raise NotImplementedError
    
    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Get entropy of the distribution.
        
        Returns:
            Entropy value.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample actions from the distribution.
        
        Returns:
            Sampled actions.
        """
        raise NotImplementedError
    
    @abstractmethod
    def mode(self) -> torch.Tensor:
        """
        Get the mode (most likely action) of the distribution.
        
        Returns:
            Mode actions.
        """
        raise NotImplementedError


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete action spaces.
    """
    
    def __init__(self, action_dim: int):
        """
        Initialize categorical distribution.
        
        Args:
            action_dim: Number of discrete actions.
        """
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Optional[Categorical] = None
    
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create linear layer for logits.
        
        Args:
            latent_dim: Input dimension.
            
        Returns:
            Linear layer mapping to action logits.
        """
        return nn.Linear(latent_dim, self.action_dim)
    
    def proba_distribution(self, action_logits: torch.Tensor) -> 'CategoricalDistribution':
        """
        Set distribution parameters from logits.
        
        Args:
            action_logits: Logits for each action.
            
        Returns:
            Self with updated distribution.
        """
        self.distribution = Categorical(logits=action_logits)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions."""
        assert self.distribution is not None
        return self.distribution.log_prob(actions)
    
    def entropy(self) -> torch.Tensor:
        """Get entropy of the distribution."""
        assert self.distribution is not None
        return self.distribution.entropy()
    
    def sample(self) -> torch.Tensor:
        """Sample actions from the distribution."""
        assert self.distribution is not None
        return self.distribution.sample()
    
    def mode(self) -> torch.Tensor:
        """Get the mode (most likely action)."""
        assert self.distribution is not None
        return torch.argmax(self.distribution.probs, dim=-1)


class DiagGaussianDistribution(Distribution):
    """
    Diagonal Gaussian distribution for continuous action spaces.
    
    Uses a diagonal covariance matrix (independent action dimensions).
    """
    
    def __init__(self, action_dim: int):
        """
        Initialize Gaussian distribution.
        
        Args:
            action_dim: Dimension of action space.
        """
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Optional[Normal] = None
        self.log_std: Optional[nn.Parameter] = None
    
    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = 0.0,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create network for mean and log_std parameter.
        
        Args:
            latent_dim: Input dimension.
            log_std_init: Initial value for log standard deviation.
            
        Returns:
            Tuple of (mean network, log_std parameter).
        """
        mean_net = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_net, log_std
    
    def proba_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> 'DiagGaussianDistribution':
        """
        Set distribution parameters.
        
        Args:
            mean: Mean actions.
            log_std: Log standard deviation.
            
        Returns:
            Self with updated distribution.
        """
        std = torch.exp(log_std)
        self.distribution = Normal(mean, std)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions."""
        assert self.distribution is not None
        # Sum log probs over action dimensions
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """Get entropy of the distribution."""
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)
    
    def sample(self) -> torch.Tensor:
        """Sample actions from the distribution."""
        assert self.distribution is not None
        return self.distribution.rsample()  # Use rsample for reparameterization
    
    def mode(self) -> torch.Tensor:
        """Get the mode (mean) of the distribution."""
        assert self.distribution is not None
        return self.distribution.mean


class BetaDistribution(Distribution):
    """Independent Beta distributions for bounded continuous actions.

    The distribution itself operates in the unit interval.  Mapping between
    unit actions and the environment ``Box`` (including the affine Jacobian)
    is handled by the policy so this class remains reusable for arbitrary
    finite action bounds.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Optional[Beta] = None

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        return nn.Linear(latent_dim, self.action_dim), nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> 'BetaDistribution':
        self.distribution = Beta(alpha, beta)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.rsample()

    def mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    def mode(self) -> torch.Tensor:
        """Return a finite deterministic representative for every Beta shape.

        With the continuous-PPO default ``alpha,beta > 1`` this is the true
        interior mode.  The remaining branches keep future ``beta_min < 1``
        ablations numerically defined without affecting the default path.
        """
        assert self.distribution is not None
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        result = torch.empty_like(alpha)

        interior = (alpha > 1.0) & (beta > 1.0)
        left = (alpha <= 1.0) & (beta > 1.0)
        right = (alpha > 1.0) & (beta <= 1.0)
        ambiguous = ~(interior | left | right)

        denominator = (alpha + beta - 2.0).clamp_min(torch.finfo(alpha.dtype).eps)
        result[interior] = ((alpha - 1.0) / denominator)[interior]
        result[left] = 0.0
        result[right] = 1.0
        result[ambiguous] = self.distribution.mean[ambiguous]
        return result


class SquashedDiagGaussianDistribution(Distribution):
    """
    Squashed Diagonal Gaussian distribution for SAC.
    
    Uses tanh squashing to bound actions to [-1, 1].
    Applies the log probability correction for the tanh transformation.
    """
    
    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        """
        Initialize squashed Gaussian distribution.
        
        Args:
            action_dim: Dimension of action space.
            epsilon: Small value for numerical stability.
        """
        super().__init__()
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.distribution: Optional[Normal] = None
        self.gaussian_actions: Optional[torch.Tensor] = None
    
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = -3.0) -> Tuple[nn.Module, nn.Module]:
        """
        Create networks for mean and log_std.
        
        Args:
            latent_dim: Input dimension.
            log_std_init: Initial value for log standard deviation.
            
        Returns:
            Tuple of (mean_net, log_std_net).
        """
        mean_net = nn.Linear(latent_dim, self.action_dim)
        log_std_net = nn.Linear(latent_dim, self.action_dim)
        return mean_net, log_std_net
    
    def proba_distribution(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> 'SquashedDiagGaussianDistribution':
        """
        Set distribution parameters.
        
        Args:
            mean: Mean actions (before squashing).
            log_std: Log standard deviation.
            
        Returns:
            Self with updated distribution.
        """
        std = torch.exp(log_std)
        self.distribution = Normal(mean, std)
        return self
    
    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get log probability of squashed actions.
        
        Applies the correction for the tanh transformation:
        log_prob = gaussian_log_prob - sum(log(1 - tanh(x)^2))
        
        Args:
            actions: Squashed actions in [-1, 1].
            gaussian_actions: Pre-squash Gaussian samples (optional).
            
        Returns:
            Log probabilities.
        """
        assert self.distribution is not None
        
        if gaussian_actions is None:
            # Inverse tanh to get pre-squash values (atanh)
            gaussian_actions = torch.clamp(actions, -1 + self.epsilon, 1 - self.epsilon)
            gaussian_actions = 0.5 * (torch.log(1 + gaussian_actions) - torch.log(1 - gaussian_actions))
        
        # Gaussian log prob
        log_prob = self.distribution.log_prob(gaussian_actions).sum(dim=-1)
        
        # Correction for tanh squashing
        # log(1 - tanh(x)^2) = log(1 - action^2)
        log_prob -= torch.log(1 - actions ** 2 + self.epsilon).sum(dim=-1)
        
        return log_prob
    
    def entropy(self) -> torch.Tensor:
        """
        Get entropy of the distribution.
        
        Note: This returns the entropy of the Gaussian, not the squashed distribution.
        The true entropy of the squashed distribution is intractable.
        
        Returns:
            Entropy value.
        """
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample squashed actions from the distribution.
        
        Returns:
            Tuple of (squashed_actions, gaussian_actions).
        """
        assert self.distribution is not None
        # Reparameterized sample
        gaussian_actions = self.distribution.rsample()
        squashed_actions = torch.tanh(gaussian_actions)
        return squashed_actions, gaussian_actions
    
    def mode(self) -> torch.Tensor:
        """Get the mode (squashed mean) of the distribution."""
        assert self.distribution is not None
        return torch.tanh(self.distribution.mean)
    
    def log_prob_from_params(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample and compute log prob in one call.
        
        Args:
            mean: Mean actions.
            log_std: Log standard deviation.
            
        Returns:
            Tuple of (squashed_actions, log_prob).
        """
        self.proba_distribution(mean, log_std)
        actions, gaussian_actions = self.sample()
        log_prob = self.log_prob(actions, gaussian_actions)
        return actions, log_prob
    
    def actions_from_params(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get actions from distribution parameters.
        
        Args:
            mean: Mean actions.
            log_std: Log standard deviation.
            deterministic: Whether to use deterministic action (mode).
            
        Returns:
            Squashed actions.
        """
        self.proba_distribution(mean, log_std)
        if deterministic:
            return self.mode()
        actions, _ = self.sample()
        return actions


def make_proba_distribution(
    action_space: spaces.Space,
    distribution_type: str = 'auto',
    **kwargs,
) -> Distribution:
    """
    Create appropriate distribution for action space.
    
    Args:
        action_space: The action space.
        distribution_type: Type of distribution
            ('auto', 'categorical', 'gaussian', 'beta').
        **kwargs: Additional arguments for the distribution.
        
    Returns:
        Distribution instance.
    """
    if distribution_type == 'auto':
        if isinstance(action_space, spaces.Discrete):
            distribution_type = 'categorical'
        elif isinstance(action_space, spaces.Box):
            distribution_type = 'gaussian'
        else:
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")
    
    if distribution_type == 'categorical':
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("categorical distribution requires a Discrete action space")
        return CategoricalDistribution(action_space.n)
    elif distribution_type == 'gaussian':
        if not isinstance(action_space, spaces.Box):
            raise ValueError("gaussian distribution requires a Box action space")
        return DiagGaussianDistribution(int(np.prod(action_space.shape)))
    elif distribution_type == 'beta':
        if not isinstance(action_space, spaces.Box):
            raise ValueError("beta distribution requires a Box action space")
        if not np.all(np.isfinite(action_space.low)) or not np.all(np.isfinite(action_space.high)):
            raise ValueError("beta distribution requires finite Box action bounds")
        return BetaDistribution(int(np.prod(action_space.shape)))
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
