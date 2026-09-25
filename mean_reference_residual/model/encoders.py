import torch
from torch import nn


class SharedObservationEncoder(nn.Module):
    """Apply one shared pointwise MLP to every deformation and pair."""

    features_per_deformation = 4

    def __init__(self, n_deformations, d_model=64, hidden_dim=64, dropout=0.0):
        super().__init__()
        if n_deformations < 1:
            raise ValueError("n_deformations must be at least 1.")
        self.n_deformations = n_deformations
        self.d_model = d_model
        self.feature_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def encode_grid(self, observations):
        """Return [batch, M, N_B, N_U, d_model]."""
        if observations.ndim != 4:
            raise ValueError("observations must have shape [B, 4*M, N_B, N_U].")
        batch, channels, n_b, n_u = observations.shape
        expected_channels = 4 * self.n_deformations
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, received {channels}."
            )
        observation_grid = observations.reshape(
            batch, self.n_deformations, 4, n_b, n_u
        ).permute(0, 1, 3, 4, 2)
        return self.feature_mlp(observation_grid)

    def forward(self, observations):
        encoded = self.encode_grid(observations)
        batch, n_deformations, n_b, n_u, d_model = encoded.shape
        return encoded.reshape(
            batch, n_deformations * n_b * n_u, d_model
        )


class GeometryEncoder(nn.Module):
    """Map each 15-value physical pair descriptor to d_model features."""

    def __init__(
        self,
        n_deformations,
        input_dim=15,
        d_model=64,
        hidden_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        if n_deformations < 1:
            raise ValueError("n_deformations must be at least 1.")
        self.n_deformations = n_deformations
        self.input_dim = input_dim
        self.d_model = d_model
        self.geometry_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def encode_grid(self, geometry):
        """Return [batch, M, N_B, N_U, d_model]."""
        if geometry.ndim == 4:
            geometry = geometry.unsqueeze(0)
        if geometry.ndim != 5:
            raise ValueError(
                "geometry must have shape [M,N_B,N_U,15] or [B,M,N_B,N_U,15]."
            )
        if geometry.shape[1] != self.n_deformations:
            raise ValueError(
                f"Expected M={self.n_deformations}, received M={geometry.shape[1]}."
            )
        if geometry.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} geometry values, received {geometry.shape[-1]}."
            )
        return self.geometry_mlp(geometry)

    def forward(self, geometry):
        encoded = self.encode_grid(geometry)
        batch, n_deformations, n_b, n_u, d_model = encoded.shape
        return encoded.reshape(
            batch, n_deformations * n_b * n_u, d_model
        )
