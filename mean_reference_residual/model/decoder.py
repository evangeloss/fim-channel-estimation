"""Copy into model/decoder.py for the beamspace CNN architecture.

This replaces the earlier attention model's decoder in the NEW project.
"""
import torch
from torch import nn


class BeamspaceResidualDecoder(nn.Module):
    """Decode [B, C, N_B, N_U] into [B, 4, N_B, N_U].

    Outputs a BEAMSPACE residual for two subcarriers in the order
    [real(k), imag(k), real(k+1), imag(k+1)]. No output activation:
    residual components may be positive, negative, or greater than one.
    """

    def __init__(self, feature_channels=64, hidden_channels=64):
        super().__init__()
        for value in (feature_channels, hidden_channels):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Channel counts must be positive integers.')
        self.feature_channels = feature_channels
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_head = nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1)
        # Start with small corrections while allowing gradients into the backbone.
        nn.init.normal_(self.output_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, conditioned_features):
        if conditioned_features.ndim != 4 or conditioned_features.shape[1] != self.feature_channels:
            raise ValueError(f'Expected [B, {self.feature_channels}, N_B, N_U].')
        if not torch.is_floating_point(conditioned_features):
            raise TypeError('Features must be real floating-point tensors.')
        return self.output_head(self.refinement(conditioned_features))
