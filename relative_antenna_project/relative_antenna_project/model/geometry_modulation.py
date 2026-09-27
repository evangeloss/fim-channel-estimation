import torch
from torch import nn


class SpatialGeometryModulation(nn.Module):
    """Per-observation, per-pair FiLM before fusion; gamma/beta [B,M,C,NB,NU]."""
    def __init__(self, geometry_dim=64, feature_channels=64, hidden_dim=128):
        super().__init__()
        self.conditioner = nn.Sequential(nn.Conv2d(geometry_dim,hidden_dim,1),
            nn.GELU(),nn.Conv2d(hidden_dim,2*feature_channels,1))
        nn.init.normal_(self.conditioner[-1].weight, std=1e-3)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, features, geometry_features, strength=1.0):
        b,m,d,nb,nu = geometry_features.shape
        params = self.conditioner(geometry_features.reshape(b*m,d,nb,nu))
        dg,beta = params.reshape(b,m,-1,nb,nu).chunk(2,dim=2)
        gamma, beta = 1 + strength*dg, strength*beta
        return gamma*features + beta, gamma, beta
