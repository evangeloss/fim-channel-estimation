"""Beamspace observation feature extractor. Copy into model/dilated_cnn.py."""
import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    """Two spatial convolutions with a residual connection; no downsampling."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
        )
        self.activation = nn.GELU()

    def forward(self, features):
        return self.activation(features + self.layers(features))


class DilatedCNN(nn.Module):
    """Map [B, 4*M, N_B, N_U] to [B, feature_channels, N_B, N_U].

    Input is already in beamspace, retaining the dataset channel ordering.
    All deformations are combined by the stem convolution. M is fixed at
    construction. This module contains no geometry conditioning or decoder.
    Uses zero padding, GELU, and no pooling, dropout, or normalization.
    """

    def __init__(self, n_deformations, feature_channels=64, dilations=(1, 2, 4)):
        super().__init__()
        for name, value in [('n_deformations', n_deformations),
                            ('feature_channels', feature_channels)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        dilations = tuple(dilations)
        if not dilations or any(not isinstance(d, int) or isinstance(d, bool) or d < 1
                                for d in dilations):
            raise ValueError('dilations must contain positive integers.')
        self.n_deformations = n_deformations
        self.feature_channels = feature_channels
        self.dilations = dilations
        self.stem = nn.Sequential(
            nn.Conv2d(4 * n_deformations, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(feature_channels, dilation) for dilation in dilations
        ])

    def forward(self, observations_beam):
        if observations_beam.ndim != 4:
            raise ValueError('Expected [B, 4*M, N_B, N_U].')
        if observations_beam.shape[1] != 4 * self.n_deformations:
            raise ValueError(f'Expected {4*self.n_deformations} input channels, '
                             f'got {observations_beam.shape[1]}.')
        if not torch.is_floating_point(observations_beam):
            raise TypeError('Input must be a real floating-point tensor with packed real/imaginary channels.')
        return self.blocks(self.stem(observations_beam))
