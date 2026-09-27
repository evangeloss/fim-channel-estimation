import torch
from torch import nn
from .relative_preprocessing import RelativePreprocessing
from .dilated_cnn import SharedObservationCNN
from .encoders import GeometryEncoder
from .geometry_modulation import SpatialGeometryModulation
from .decoder import AntennaResidualDecoder
from .residual_reconstruction import ResidualReconstruction
from geometry.surfaces import generate_reference_geometry
from geometry.features import build_fixed_origin_features
import numpy as np


class ChannelEstimator(nn.Module):
    def __init__(self, n_deformations=8, n_h_b=5,n_v_b=5,n_h_u=5,n_v_u=5,
                 feature_channels=64,geometry_dim=64,geometry_hidden_dim=64,
                 modulation_hidden_dim=128,decoder_hidden_channels=64,
                 dilations=(1,2,4),residual_reference='mean',reference_index=0,
                 spacing_b=(0.125,0.125),spacing_u=(0.125,0.125),
                 nominal_geometry=None, variant='relative', architecture='relative_antenna_v1'):
        super().__init__()
        if residual_reference != 'mean' or architecture != 'relative_antenna_v1':
            raise ValueError('This architecture requires mean residual targets.')
        if variant not in ('relative','raw'):
            raise ValueError('variant must be relative or raw.')
        if nominal_geometry is None:
            pb,pu,_,_ = generate_reference_geometry(n_h_b,n_v_b,*spacing_b,
                n_h_u,n_v_u,*spacing_u,1.0)
            nominal_geometry = build_fixed_origin_features([pb],[np.zeros_like(pb)],
                [pu],[np.zeros_like(pu)],1.0,pb.mean(axis=1))[0]
        nominal = torch.as_tensor(nominal_geometry,dtype=torch.float32)
        if nominal.shape != (n_h_b*n_v_b,n_h_u*n_v_u,6) or not torch.isfinite(nominal).all():
            raise ValueError('Invalid nominal geometry.')
        self.register_buffer('nominal_geometry',nominal.clone())
        self.config = dict(n_deformations=n_deformations,n_h_b=n_h_b,n_v_b=n_v_b,
            n_h_u=n_h_u,n_v_u=n_v_u,feature_channels=feature_channels,
            geometry_dim=geometry_dim,geometry_hidden_dim=geometry_hidden_dim,
            modulation_hidden_dim=modulation_hidden_dim,decoder_hidden_channels=decoder_hidden_channels,
            dilations=tuple(dilations),residual_reference=residual_reference,
            reference_index=reference_index,spacing_b=tuple(spacing_b),spacing_u=tuple(spacing_u),
            nominal_geometry=nominal.tolist(),variant=variant,architecture=architecture)
        self.variant = variant
        self.preprocessing = RelativePreprocessing(n_deformations,reference_index)
        self.cnn = SharedObservationCNN(feature_channels,dilations,8 if variant=='relative' else 4)
        self.geometry_encoder = GeometryEncoder(18,geometry_dim,geometry_hidden_dim)
        self.modulation = SpatialGeometryModulation(geometry_dim,feature_channels,modulation_hidden_dim)
        self.decoder = AntennaResidualDecoder(feature_channels,decoder_hidden_channels)
        self.reconstruction = ResidualReconstruction()

    def forward(self,observations,geometry,return_auxiliary=False,modulation_strength=1.0):
        x,g = self.preprocessing(observations,geometry,self.nominal_geometry)
        if self.variant == 'raw':
            b,_,nb,nu = observations.shape
            x = observations.reshape(b,-1,4,nb,nu)
        f = self.cnn(x)
        f,gamma,beta = self.modulation(f,self.geometry_encoder(g),modulation_strength)
        residual = self.decoder(f.mean(dim=1))
        prediction = self.reconstruction(residual,observations)
        if return_auxiliary:
            return prediction,dict(residual_antenna=residual,gamma=gamma,beta=beta)
        return prediction
