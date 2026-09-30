"""Feature extractors V2 for RL policies.

Deeper CNN encoder for BEV observations:
- Configurable multi-layer CNN with LayerNorm
- Separate scalar-state MLP branch
- Late-fusion MLP
- Xavier initialisation for CNN layers
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

__layer__ = (4, "Algorithm")
from gymnasium import spaces

from .feature_extractor import BaseFeaturesExtractor


class CNNFeatureExtractorV2(BaseFeaturesExtractor):
    """
    Deep CNN feature extractor with optional LayerNorm.

    Progressively reduces spatial dimensions through multiple conv layers,
    each followed by optional LayerNorm and ReLU. Output spatial size is
    computed automatically via a dummy forward pass.

    Input: (B, C, H, W) BEV tensor
    Output: (B, cnn_flat_dim) flattened feature vector
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 1024,
        use_layer_norm: bool = True,
        cnn_channels: Optional[List[int]] = None,
    ):
        super().__init__(observation_space, features_dim)

        self.use_layer_norm = use_layer_norm

        if isinstance(observation_space, spaces.Dict):
            for key, space in observation_space.spaces.items():
                if len(space.shape) == 3:
                    obs_shape = space.shape
                    break
            else:
                raise ValueError("No image observation found in Dict space")
        else:
            obs_shape = observation_space.shape

        if obs_shape[0] < obs_shape[1] and obs_shape[0] < obs_shape[2]:
            in_channels = obs_shape[0]
            height, width = obs_shape[1], obs_shape[2]
        else:
            in_channels = obs_shape[2]
            height, width = obs_shape[0], obs_shape[1]
            self.needs_transpose = True
        self.needs_transpose = getattr(self, 'needs_transpose', False)
        self.input_shape_chw = (in_channels, height, width)

        if cnn_channels is None:
            cnn_channels = [8, 16, 32, 64, 128, 256]

        self.cnn = self._build_cnn(in_channels, height, width, cnn_channels)

        with torch.no_grad():
            sample = torch.zeros(1, in_channels, height, width)
            cnn_out = self.cnn(sample)
            self._cnn_flat_dim = cnn_out.shape[1]

        if not self._conv_output_shapes:
            raise ValueError("CNNFeatureExtractorV2 requires at least one conv layer")
        self._features_dim = self._cnn_flat_dim
        self._apply_xavier_init()

    def _build_cnn(
        self,
        in_channels: int,
        height: int,
        width: int,
        channels: List[int],
    ) -> nn.Sequential:
        """Build deep CNN with adaptive kernel/stride selection."""
        layers = []
        prev_ch = in_channels
        h, w = height, width
        self._conv_specs: List[Dict[str, Any]] = []
        self._conv_output_shapes: List[Tuple[int, int, int]] = []

        for out_ch in channels:
            if min(h, w) > 16:
                kernel, stride = 5, 2
            elif min(h, w) > 6:
                kernel, stride = 3, 2
            elif min(h, w) > 2:
                kernel, stride = 3, 1
            else:
                kernel, stride = 1, 1

            conv = nn.Conv2d(prev_ch, out_ch, kernel_size=kernel, stride=stride)
            layers.append(conv)

            h_out = (h - kernel) // stride + 1
            w_out = (w - kernel) // stride + 1
            self._conv_specs.append({
                'in_channels': prev_ch,
                'out_channels': out_ch,
                'kernel': kernel,
                'stride': stride,
                'input_hw': (h, w),
                'output_hw': (h_out, w_out),
            })
            self._conv_output_shapes.append((out_ch, h_out, w_out))

            if self.use_layer_norm and h_out > 0 and w_out > 0:
                layers.append(nn.LayerNorm((out_ch, h_out, w_out)))

            layers.append(nn.ReLU())
            prev_ch = out_ch
            h, w = h_out, w_out

        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def _apply_xavier_init(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    @property
    def cnn_flat_dim(self) -> int:
        return self._cnn_flat_dim

    def _prepare_tensor(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(observations, dict):
            for key in ['vector', 'bev_mask', 'obs']:
                if key in observations:
                    tensor = observations[key]
                    break
            else:
                for key, obs in observations.items():
                    if isinstance(obs, torch.Tensor) and len(obs.shape) >= 3:
                        tensor = obs
                        break
                else:
                    tensor = next(iter(observations.values()))
        else:
            tensor = observations

        if not tensor.is_floating_point():
            tensor = tensor.float()

        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        if self.needs_transpose:
            tensor = tensor.permute(0, 3, 1, 2)

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=255.0, neginf=0.0)

        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        return tensor

    def forward(self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        tensor = self._prepare_tensor(observations)
        features = self.cnn(tensor)

        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)

        return features

class CombinedExtractorV2(BaseFeaturesExtractor):
    """
    Combined feature extractor V2 for Dict observations.

    Architecture summary:
    - Image branch: deep CNN with LayerNorm -> flattened image features
    - Scalar branch: multi-layer MLP with LayerNorm -> scalar features
    - Late fusion: concat(image, scalar) -> fusion MLP with LayerNorm -> features_dim

    All sub-networks are configurable via constructor arguments.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        use_layer_norm: bool = True,
        cnn_channels: Optional[List[int]] = None,
        state_neurons: Optional[List[int]] = None,
        fusion_dims: Optional[List[int]] = None,
    ):
        super().__init__(observation_space, features_dim)

        self.use_layer_norm = use_layer_norm
        if cnn_channels is None:
            cnn_channels = [8, 16, 32, 64, 128, 256]
        if state_neurons is None:
            state_neurons = [256, 256]
        if fusion_dims is None:
            fusion_dims = [512]

        image_key = None
        scalar_keys = []

        for key, subspace in observation_space.spaces.items():
            if len(subspace.shape) == 3:
                image_key = key
            else:
                scalar_keys.append(key)

        if image_key is None:
            raise ValueError("CombinedExtractorV2 requires at least one 3D (image) observation")

        image_space = observation_space.spaces[image_key]
        self.image_key = image_key
        self.scalar_keys = scalar_keys

        self.image_encoder = CNNFeatureExtractorV2(
            observation_space=image_space,
            features_dim=0,
            use_layer_norm=use_layer_norm,
            cnn_channels=cnn_channels,
        )
        cnn_flat_dim = self.image_encoder.cnn_flat_dim

        total_scalar_dim = 0
        for key in scalar_keys:
            total_scalar_dim += int(np.prod(observation_space.spaces[key].shape))

        self.has_scalars = total_scalar_dim > 0
        if self.has_scalars:
            self.scalar_encoder = self._build_scalar_mlp(
                total_scalar_dim, state_neurons
            )
            scalar_out_dim = state_neurons[-1]
        else:
            scalar_out_dim = 0

        fusion_input_dim = cnn_flat_dim + scalar_out_dim
        self.fusion = self._build_fusion_mlp(
            fusion_input_dim, fusion_dims, features_dim
        )

    def _build_scalar_mlp(self, input_dim: int, neurons: List[int]) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for dim in neurons:
            layers.append(nn.Linear(prev_dim, dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        return nn.Sequential(*layers)

    def _build_fusion_mlp(
        self, input_dim: int, hidden_dims: List[int], output_dim: int
    ) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_obs = observations[self.image_key]
        image_features = self.image_encoder(image_obs)

        if self.has_scalars:
            scalar_parts = []
            for key in self.scalar_keys:
                s = observations[key]
                if not s.is_floating_point():
                    s = s.float()
                if len(s.shape) == 1:
                    s = s.unsqueeze(0)
                scalar_parts.append(s.flatten(start_dim=1))
            scalar_cat = torch.cat(scalar_parts, dim=1)
            scalar_features = self.scalar_encoder(scalar_cat)
            fused = torch.cat([image_features, scalar_features], dim=1)
        else:
            fused = image_features

        return self.fusion(fused)
