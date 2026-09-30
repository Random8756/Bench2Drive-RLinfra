"""Feature extractor for RGB camera stack, optional BEV, and scalar observations."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from .feature_extractor import BaseFeaturesExtractor
from .feature_extractor_v2 import CNNFeatureExtractorV2

__layer__ = (4, "Algorithm")


class RGBCameraEncoder(nn.Module):
    """Shared CNN stem for RGB observations shaped as ``(B, N, H, W, C)``."""

    def __init__(
        self,
        rgb_space: spaces.Box,
        camera_feature_dim: int = 128,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if len(rgb_space.shape) != 4:
            raise ValueError(
                f"RGB space must be (num_cameras, height, width, channels), got {rgb_space.shape}"
            )
        self.num_cameras = int(rgb_space.shape[0])
        self.height = int(rgb_space.shape[1])
        self.width = int(rgb_space.shape[2])
        self.channels = int(rgb_space.shape[3])
        if self.channels != 3:
            raise ValueError(f"RGB camera channels must be 3, got {self.channels}")

        self.camera_feature_dim = int(camera_feature_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=16, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 7)),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, 3, self.height, self.width)
            stem_dim = int(self.stem(sample).shape[1])

        projection_layers: List[nn.Module] = [
            nn.Linear(stem_dim, self.camera_feature_dim),
        ]
        if use_layer_norm:
            projection_layers.append(nn.LayerNorm(self.camera_feature_dim))
        projection_layers.append(nn.ReLU())
        self.projection = nn.Sequential(*projection_layers)
        self.output_dim = self.num_cameras * self.camera_feature_dim
        self._apply_xavier_init()

    def _apply_xavier_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if not rgb.is_floating_point():
            rgb = rgb.float()

        if len(rgb.shape) == 4:
            rgb = rgb.unsqueeze(0)
        if len(rgb.shape) != 5:
            raise ValueError(f"RGB tensor must be 5D (B,N,H,W,C), got {tuple(rgb.shape)}")

        if rgb.shape[-1] == 3:
            batch_size, num_cameras, height, width, channels = rgb.shape
            rgb = rgb.permute(0, 1, 4, 2, 3)
        elif rgb.shape[2] == 3:
            batch_size, num_cameras, channels, height, width = rgb.shape
        else:
            raise ValueError(f"RGB tensor channel dimension is not 3: {tuple(rgb.shape)}")

        if int(num_cameras) != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} RGB cameras, got {int(num_cameras)}")
        if int(channels) != 3:
            raise ValueError(f"Expected 3 RGB channels, got {int(channels)}")
        if int(height) != self.height or int(width) != self.width:
            raise ValueError(
                f"Expected RGB size {self.height}x{self.width}, got {int(height)}x{int(width)}"
            )

        rgb = rgb.reshape(batch_size * num_cameras, 3, self.height, self.width)
        rgb = rgb / 255.0
        features = self.projection(self.stem(rgb))
        features = features.reshape(batch_size, num_cameras * self.camera_feature_dim)
        if torch.isnan(features).any():
            features = torch.nan_to_num(features, nan=0.0)
        return features


class CombinedExtractorV3RGB(BaseFeaturesExtractor):
    """Combined extractor for ``Dict(rgb, scalars[, vector])`` observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        use_layer_norm: bool = True,
        cnn_channels: Optional[List[int]] = None,
        state_neurons: Optional[List[int]] = None,
        fusion_dims: Optional[List[int]] = None,
        rgb_camera_feature_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("CombinedExtractorV3RGB requires a Dict observation space")
        if "rgb" not in observation_space.spaces:
            raise ValueError("CombinedExtractorV3RGB requires observation key 'rgb'")

        self.use_layer_norm = bool(use_layer_norm)
        if cnn_channels is None:
            cnn_channels = [8, 16, 32, 64, 128, 256]
        if state_neurons is None:
            state_neurons = [256, 256]
        if fusion_dims is None:
            fusion_dims = [512]

        self.has_bev = "vector" in observation_space.spaces
        self.bev_key = "vector"
        self.rgb_key = "rgb"
        self.scalar_keys = [
            key for key in observation_space.spaces.keys()
            if key not in (self.bev_key, self.rgb_key)
        ]

        if self.has_bev:
            self.bev_encoder = CNNFeatureExtractorV2(
                observation_space=observation_space.spaces[self.bev_key],
                features_dim=0,
                use_layer_norm=use_layer_norm,
                cnn_channels=cnn_channels,
            )
            bev_out_dim = self.bev_encoder.cnn_flat_dim
        else:
            self.bev_encoder = None
            bev_out_dim = 0

        self.rgb_encoder = RGBCameraEncoder(
            observation_space.spaces[self.rgb_key],
            camera_feature_dim=rgb_camera_feature_dim,
            use_layer_norm=use_layer_norm,
        )

        scalar_input_dim = 0
        for key in self.scalar_keys:
            scalar_input_dim += int(np.prod(observation_space.spaces[key].shape))
        self.has_scalars = scalar_input_dim > 0
        if self.has_scalars:
            self.scalar_encoder = self._build_scalar_mlp(scalar_input_dim, state_neurons)
            scalar_out_dim = int(state_neurons[-1])
        else:
            scalar_out_dim = 0

        fusion_input_dim = (
            bev_out_dim
            + self.rgb_encoder.output_dim
            + scalar_out_dim
        )
        self.fusion = self._build_fusion_mlp(fusion_input_dim, fusion_dims, features_dim)

    def _build_scalar_mlp(self, input_dim: int, neurons: List[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in neurons:
            layers.append(nn.Linear(prev_dim, dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        return nn.Sequential(*layers)

    def _build_fusion_mlp(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
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
        features = []
        if self.has_bev:
            features.append(self.bev_encoder(observations[self.bev_key]))
        features.append(self.rgb_encoder(observations[self.rgb_key]))

        if self.has_scalars:
            scalar_parts = []
            for key in self.scalar_keys:
                scalar = observations[key]
                if not scalar.is_floating_point():
                    scalar = scalar.float()
                if len(scalar.shape) == 1:
                    scalar = scalar.unsqueeze(0)
                scalar_parts.append(scalar.flatten(start_dim=1))
            features.append(self.scalar_encoder(torch.cat(scalar_parts, dim=1)))

        return self.fusion(torch.cat(features, dim=1))
