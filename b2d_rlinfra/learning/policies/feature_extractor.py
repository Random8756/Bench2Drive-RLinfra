"""Feature extractors for RL policies (V1).

CNN encoder for BEV-mask observations plus a combined CNN + scalar extractor.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

__layer__ = (4, "Algorithm")



class BaseFeaturesExtractor(nn.Module, ABC):
    """
    Base class for feature extractors.
    
    Feature extractors process raw observations into feature vectors
    that can be used by the policy networks.
    """
    
    def __init__(self, observation_space: spaces.Space, features_dim: int = 256):
        """
        Initialize feature extractor.
        
        Args:
            observation_space: Observation space.
            features_dim: Output feature dimension.
        """
        super().__init__()
        self.observation_space = observation_space
        self._features_dim = features_dim
    
    @property
    def features_dim(self) -> int:
        """Get output feature dimension."""
        return self._features_dim
    
    @abstractmethod
    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Raw observations.
            
        Returns:
            Feature tensor of shape (batch_size, features_dim).
        """
        raise NotImplementedError


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN Feature Extractor for BEV Mask observations.
    
    Architecture:
    - Conv2d(in_channels, 32, 8, stride=4) -> ReLU
    - Conv2d(32, 64, 4, stride=2) -> ReLU
    - Conv2d(64, 64, 3, stride=1) -> ReLU
    - Flatten -> Linear -> features_dim
    
    Input: (B, C, H, W) BEV Mask image
    Output: (B, features_dim) feature vector
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ):
        """
        Initialize CNN feature extractor.
        
        Args:
            observation_space: Observation space (should be Box with shape (C, H, W)).
            features_dim: Output feature dimension.
            normalized_image: Whether input images are already normalized to [0, 1].
        """
        super().__init__(observation_space, features_dim)
        
        self.normalized_image = normalized_image
        
        # Get input shape
        if isinstance(observation_space, spaces.Dict):
            # Find the image observation
            for key, space in observation_space.spaces.items():
                if len(space.shape) == 3:  # (C, H, W) or (H, W, C)
                    obs_shape = space.shape
                    break
            else:
                raise ValueError("No image observation found in Dict space")
        else:
            obs_shape = observation_space.shape
        
        # Determine if (C, H, W) or (H, W, C) format
        if obs_shape[0] < obs_shape[1] and obs_shape[0] < obs_shape[2]:
            # (C, H, W) format
            in_channels = obs_shape[0]
            height, width = obs_shape[1], obs_shape[2]
        else:
            # (H, W, C) format - will need to transpose
            in_channels = obs_shape[2]
            height, width = obs_shape[0], obs_shape[1]
            self.needs_transpose = True
        
        self.needs_transpose = getattr(self, 'needs_transpose', False)
        
        # Build CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output dimension
        with torch.no_grad():
            if self.needs_transpose:
                sample = torch.zeros(1, in_channels, height, width)
            else:
                sample = torch.zeros(1, *obs_shape)
            cnn_output = self.cnn(sample)
            cnn_output_dim = cnn_output.shape[1]
        
        # Linear layer to features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Extract features from BEV observations.
        
        Args:
            observations: BEV mask tensor of shape (B, C, H, W) or dict containing it.
            
        Returns:
            Feature tensor of shape (B, features_dim).
        """
        # Handle dict observations
        if isinstance(observations, dict):
            # Priority order for finding image tensor
            for key in ['vector', 'bev_mask', 'obs']:
                if key in observations:
                    tensor = observations[key]
                    break
            else:
                # Find first tensor with 3+ dims
                for key, obs in observations.items():
                    if isinstance(obs, torch.Tensor) and len(obs.shape) >= 3:
                        tensor = obs
                        break
                else:
                    tensor = next(iter(observations.values()))
        else:
            tensor = observations
        
        # Ensure tensor is float
        if not tensor.is_floating_point():
            tensor = tensor.float()
        
        # Ensure 4D: (B, C, H, W)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        # Transpose if needed (H, W, C) -> (C, H, W)
        if self.needs_transpose:
            tensor = tensor.permute(0, 3, 1, 2)
        
        # Replace NaN/Inf with zeros to prevent propagation
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=255.0, neginf=0.0)
        
        # Normalize if not already done (BEV mask is 0/1, BEV image is 0-255)
        if not self.normalized_image:
            # Check if already normalized
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        
        # Extract features
        cnn_features = self.cnn(tensor)
        features = self.linear(cnn_features)
        
        # Final NaN check - replace with zeros if any NaN propagated
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        
        return features


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observations.
    
    Processes different observation types with appropriate extractors:
    - Image observations: CNN
    - Vector observations: MLP
    
    Then concatenates all features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        mlp_output_dim: int = 64,
        features_dim: int = 256,
    ):
        """
        Initialize combined extractor.
        
        Args:
            observation_space: Dict observation space.
            cnn_output_dim: Output dim for CNN extractors.
            mlp_output_dim: Output dim for MLP extractors.
            features_dim: Final combined feature dimension.
        """
        super().__init__(observation_space, features_dim)
        
        extractors = {}
        total_concat_dim = 0
        
        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 3:
                # Image observation - use CNN
                extractors[key] = CNNFeatureExtractor(subspace, cnn_output_dim)
                total_concat_dim += cnn_output_dim
            else:
                # Vector observation - use flatten + MLP
                flat_dim = int(np.prod(subspace.shape))
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_dim, mlp_output_dim),
                    nn.ReLU(),
                )
                total_concat_dim += mlp_output_dim
        
        self.extractors = nn.ModuleDict(extractors)
        
        # Final combination layer
        self.combine = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and combine features from all observation types.
        
        Args:
            observations: Dict of observation tensors.
            
        Returns:
            Combined feature tensor.
        """
        encoded = []
        for key, extractor in self.extractors.items():
            if key in observations:
                encoded.append(extractor(observations[key]))
        
        combined = torch.cat(encoded, dim=1)
        return self.combine(combined)
