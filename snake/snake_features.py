"""
Custom Feature Extractor for Snake RL Agent.
CNN-based feature extraction for grid observations.
"""

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class SnakeFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for Snake game observations with ADAPTIVE POOLING.

    Uses AdaptiveAvgPool2d to produce fixed-size output regardless of input grid size.
    This enables curriculum learning across different board sizes (6x6 -> 20x20).

    Architecture:
        - 3 convolutional layers with ReLU activations
        - Adaptive pooling to fixed 4x4 spatial size
        - Linear projection to features_dim

    Input: (batch, 8, n, n) observation tensor (any n)
    Output: (batch, features_dim) feature vector
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ):
        """
        Initialize the feature extractor.

        Args:
            observation_space: The observation space (Box with shape (C, H, W))
            features_dim: Output feature dimension
        """
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # Expected: 8

        # CNN with global average pooling - works on MPS and any grid size
        # More channels to compensate for spatial compression
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global avg pool - works on any size & MPS
            nn.Flatten(),
        )

        # Fixed flatten size: 256 channels * 1 * 1 = 256
        n_flatten = 256

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Batch of observations (B, C, H, W)

        Returns:
            Feature vectors (B, features_dim)
        """
        x = self.cnn(observations)
        return self.linear(x)


class SnakeFeatureExtractorLarge(BaseFeaturesExtractor):
    """
    Larger CNN feature extractor for more complex learning.

    Architecture:
        - 4 convolutional layers with batch norm
        - Max pooling for larger grids
        - Dropout for regularization
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        grid_size = observation_space.shape[1]

        # Determine if we need pooling (for larger grids)
        use_pooling = grid_size >= 14

        layers = [
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ]

        if use_pooling:
            layers.append(nn.MaxPool2d(2))

        layers.extend([
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ])

        if use_pooling:
            layers.append(nn.MaxPool2d(2))

        layers.extend([
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        ])

        self.cnn = nn.Sequential(*layers)

        # Compute output shape
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations)
        return self.linear(x)


class SnakeFeatureExtractorResidual(BaseFeaturesExtractor):
    """
    Feature extractor with residual connections for better gradient flow.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 64)

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output shape
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            x = self.input_conv(sample)
            x = self.res_block1(x) + x
            x = self.res_block2(x) + x
            n_flatten = self.final_conv(x).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def _make_res_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.input_conv(observations)
        x = th.relu(self.res_block1(x) + x)
        x = th.relu(self.res_block2(x) + x)
        x = self.final_conv(x)
        return self.linear(x)
