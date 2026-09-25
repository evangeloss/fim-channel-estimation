"""Copy to model/geometry_modulation.py.

Global geometry-conditioned feature modulation after the dilated CNN.
Consumes GeometryEncoder.encode_grid() output, not raw coordinates.
"""
import torch
from torch import nn


class GeometryConditionedModulation(nn.Module):
    """Apply (1 + delta_gamma) * features + beta.

    features: [B, C, H, W] beamspace CNN features.
    geometry_features: [B, M, N_B, N_U, D] physical-pair embeddings.

    For each deformation, summarize each embedding channel using its mean
    and population standard deviation across antenna pairs. Concatenate the
    M summaries in the same deformation order as the observation input.
    An MLP produces [B, C, 1, 1] modulation coefficients.

    This is global conditioning, not a spatial correspondence between antenna
    pairs and beam bins. Pooling discards explicit pair locations; this is a
    deliberate baseline simplification, not a lossless geometry representation.
    """

    def __init__(self, n_deformations, geometry_dim=64, feature_channels=64,
                 hidden_dim=128):
        super().__init__()
        for name, value in [('n_deformations', n_deformations),
                            ('geometry_dim', geometry_dim),
                            ('feature_channels', feature_channels),
                            ('hidden_dim', hidden_dim)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        self.n_deformations = n_deformations
        self.geometry_dim = geometry_dim
        self.feature_channels = feature_channels
        self.conditioner = nn.Sequential(
            nn.Linear(2 * n_deformations * geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_channels),
        )
        # Start near identity while allowing gradients into the geometry branch
        # from the first backward pass (unlike an exactly zero weight matrix).
        nn.init.normal_(self.conditioner[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, features, geometry_features, return_parameters=False):
        if features.ndim != 4 or features.shape[1] != self.feature_channels:
            raise ValueError(f'features must have shape [B, {self.feature_channels}, H, W].')
        if geometry_features.ndim != 5:
            raise ValueError('Use geometry_encoder.encode_grid(): expected [B, M, N_B, N_U, D].')
        if geometry_features.shape[0] != features.shape[0]:
            raise ValueError('Features and geometry must have the same batch size.')
        if geometry_features.shape[1] != self.n_deformations or geometry_features.shape[-1] != self.geometry_dim:
            raise ValueError('Geometry deformation count or embedding dimension does not match configuration.')
        if any(size == 0 for size in geometry_features.shape):
            raise ValueError('Geometry dimensions must be nonempty.')
        if not torch.is_floating_point(features) or not torch.is_floating_point(geometry_features):
            raise TypeError('Features and geometry embeddings must be real floating-point tensors.')
        if features.device != geometry_features.device or features.dtype != geometry_features.dtype:
            raise ValueError('Features and geometry embeddings must have the same device and dtype.')

        mean = geometry_features.mean(dim=(2, 3))
        # Epsilon keeps gradients finite for constant or single-pair geometry.
        std = (geometry_features.var(dim=(2, 3), unbiased=False) + 1e-8).sqrt()
        summary = torch.cat((mean, std), dim=-1).flatten(start_dim=1)
        delta_gamma, beta = self.conditioner(summary).chunk(2, dim=-1)
        gamma = (1.0 + delta_gamma)[:, :, None, None]
        beta = beta[:, :, None, None]
        modulated = gamma * features + beta
        if return_parameters:
            return modulated, gamma, beta
        return modulated
