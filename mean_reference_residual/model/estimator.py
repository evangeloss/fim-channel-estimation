"""Copy into model/estimator.py. Full beamspace geometry-conditioned estimator."""
from torch import nn
from .beamspace import BeamspaceTransform
from .dilated_cnn import DilatedCNN
from .encoders import GeometryEncoder
from .geometry_modulation import GeometryConditionedModulation
from .decoder import BeamspaceResidualDecoder
from .residual_reconstruction import ResidualReconstruction


class ChannelEstimator(nn.Module):
    """Predict normalized antenna-domain H0 from observations and geometry.

    observations: [B, 4*M, N_B, N_U], real/imaginary adjacent subcarriers.
    geometry: [B, M, N_B, N_U, 6], fixed-origin coordinates / wavelength.
    output: [B, 4, N_B, N_U], normalized H0 (NOT a residual).
    """

    def __init__(self, n_deformations=8, n_h_b=5, n_v_b=5, n_h_u=5, n_v_u=5,
                 feature_channels=64, geometry_dim=64, geometry_hidden_dim=64,
                 modulation_hidden_dim=128, decoder_hidden_channels=64,
                 dilations=(1, 2, 4), residual_reference='mean'):
        super().__init__()
        self.config = dict(n_deformations=n_deformations, n_h_b=n_h_b, n_v_b=n_v_b,
            n_h_u=n_h_u, n_v_u=n_v_u, feature_channels=feature_channels,
            geometry_dim=geometry_dim, geometry_hidden_dim=geometry_hidden_dim,
            modulation_hidden_dim=modulation_hidden_dim,
            decoder_hidden_channels=decoder_hidden_channels, dilations=tuple(dilations), residual_reference=residual_reference)
        self.beamspace = BeamspaceTransform(n_h_b, n_v_b, n_h_u, n_v_u)
        self.cnn = DilatedCNN(n_deformations, feature_channels, dilations)
        self.geometry_encoder = GeometryEncoder(n_deformations=n_deformations,
            input_dim=6, d_model=geometry_dim, hidden_dim=geometry_hidden_dim, dropout=0.0)
        self.modulation = GeometryConditionedModulation(n_deformations,
            geometry_dim, feature_channels, modulation_hidden_dim)
        self.decoder = BeamspaceResidualDecoder(feature_channels, decoder_hidden_channels)
        self.reconstruction = ResidualReconstruction(n_h_b, n_v_b, n_h_u, n_v_u, residual_reference=residual_reference)

    def forward(self, observations, geometry, return_auxiliary=False):
        if observations.ndim != 4 or geometry.ndim != 5:
            raise ValueError('Expected observations [B,4*M,N_B,N_U] and geometry [B,M,N_B,N_U,6].')
        if (geometry.shape[0] != observations.shape[0] or
                tuple(geometry.shape[2:4]) != tuple(observations.shape[2:4]) or
                geometry.shape[-1] != 6):
            raise ValueError('Geometry batch, antenna dimensions, or descriptor width do not match.')
        features = self.cnn(self.beamspace(observations))
        geometry_features = self.geometry_encoder.encode_grid(geometry)
        features, gamma, beta = self.modulation(features, geometry_features, return_parameters=True)
        residual_beam = self.decoder(features)
        prediction, residual = self.reconstruction(residual_beam, observations, return_residual=True)
        if return_auxiliary:
            return prediction, {'residual_antenna': residual, 'residual_beam': residual_beam,
                                'gamma': gamma, 'beta': beta}
        return prediction
