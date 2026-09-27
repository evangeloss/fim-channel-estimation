from torch import nn


class GeometryEncoder(nn.Module):
    """Retain each observation and antenna pair; no spatial pooling."""
    def __init__(self, input_dim=18, d_model=64, hidden_dim=64):
        super().__init__()
        self.geometry_mlp = nn.Sequential(nn.Linear(input_dim,hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim,d_model), nn.GELU())

    def forward(self, geometry):
        return self.geometry_mlp(geometry).permute(0,1,4,2,3)
