"""
Base Policy Class.

Abstract base class for RL policies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces


from .feature_extractor import BaseFeaturesExtractor, CNNFeatureExtractor


class BasePolicy(nn.Module, ABC):
    """
    Abstract base class for all policies.
    
    Defines the interface for:
    - Extracting features from observations
    - Predicting actions
    - Saving/loading
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = CNNFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        squash_output: bool = False,
    ):
        """
        Initialize policy.
        
        Args:
            observation_space: Observation space.
            action_space: Action space.
            features_extractor_class: Class for feature extraction.
            features_extractor_kwargs: Arguments for feature extractor.
            optimizer_class: Optimizer class.
            optimizer_kwargs: Arguments for optimizer.
            squash_output: Whether to squash output to action bounds.
        """
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.squash_output = squash_output
        
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs or {}
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        
        self.features_extractor: Optional[BaseFeaturesExtractor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
    
    @abstractmethod
    def _build(self) -> None:
        """Build the policy networks."""
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass to get actions.
        
        Args:
            obs: Observation tensor or dict.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Actions.
        """
        raise NotImplementedError
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get parameters needed to recreate the policy."""
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space,
        }
    
    def extract_features(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            obs: Observation tensor or dict.
            
        Returns:
            Feature tensor.
        """
        assert self.features_extractor is not None
        return self.features_extractor(obs)
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get action for given observation.
        
        Args:
            observation: Current observation.
            state: RNN state (not used for MLP policies).
            episode_start: Whether this is the start of an episode.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Tuple of (action, state).
        """
        self.set_training_mode(False)

        # Detect single (unbatched) observation and add batch dim so that all
        # downstream layers (CNN, Flatten, Linear) receive the expected 4-D /
        # 2-D tensors. During training observations are already batched by the
        # vectorised env (or manually unsqueezed before calling predict);
        # during leaderboard eval they arrive as single steps without batch dim.
        import torch as _torch
        obs_space = self.observation_space
        if isinstance(observation, dict):
            first_key = next(iter(observation))
            first_val = observation[first_key]
            expected_ndim = (
                len(obs_space[first_key].shape)
                if isinstance(obs_space, spaces.Dict)
                else len(obs_space.shape)
            )
            single_obs = first_val.ndim == expected_ndim
            if single_obs:
                observation = {
                    k: (np.expand_dims(v, 0) if isinstance(v, np.ndarray) else v.unsqueeze(0))
                    for k, v in observation.items()
                }
        else:
            single_obs = observation.ndim == len(obs_space.shape)
            if single_obs:
                observation = (
                    np.expand_dims(observation, 0)
                    if isinstance(observation, np.ndarray)
                    else observation.unsqueeze(0)
                )

        # Convert to tensors
        obs_tensor = self._obs_to_tensor(observation)
        
        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        
        # Convert to numpy
        actions = actions.cpu().numpy()

        # Drop the batch dimension added for single observations
        if single_obs:
            actions = actions.squeeze(0)

        # Clip actions if needed
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        return actions, state
    
    @abstractmethod
    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Internal prediction method.
        
        Args:
            observation: Observation tensor.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Action tensor.
        """
        raise NotImplementedError
    
    def _obs_to_tensor(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert observation to tensor.
        
        Args:
            observation: Numpy observation.
            
        Returns:
            Tensor observation.
        """
        device = next(self.parameters()).device
        
        if isinstance(observation, dict):
            return {
                key: torch.as_tensor(value, device=device).float()
                for key, value in observation.items()
            }
        else:
            return torch.as_tensor(observation, device=device).float()
    
    def set_training_mode(self, mode: bool) -> None:
        """Set training mode."""
        self.train(mode)
    
    def save(self, path: str) -> None:
        """
        Save policy to file.
        
        Args:
            path: Path to save file.
        """
        torch.save(
            {
                'state_dict': self.state_dict(),
                'data': self._get_constructor_parameters(),
            },
            path,
        )
    
    @classmethod
    def load(cls, path: str, device: Union[str, torch.device] = 'auto') -> 'BasePolicy':
        """
        Load policy from file.
        
        Args:
            path: Path to saved file.
            device: Device to load to.
            
        Returns:
            Loaded policy.
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        data = torch.load(path, map_location=device)
        
        policy = cls(**data['data'])
        policy.load_state_dict(data['state_dict'])
        policy.to(device)
        
        return policy


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> nn.Sequential:
    """
    Create a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        net_arch: List of hidden layer sizes.
        activation_fn: Activation function class.
        squash_output: Whether to apply tanh to output.
        
    Returns:
        MLP as nn.Sequential.
    """
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in net_arch:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_fn())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if squash_output:
        layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)
