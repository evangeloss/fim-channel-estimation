import torch
from torch import nn


class RelativePreprocessing(nn.Module):
    """Coordinates are in wavelength units in a shared, fixed coordinate frame.

    Descriptor: [G0, Gr-G0, Gm-Gr], each with BS xyz and UE xyz.
    The input reference is a measured state; G0 is known geometry, never H0.
    """
    def __init__(self, n_deformations, reference_index=0):
        super().__init__()
        if not 0 <= reference_index < n_deformations:
            raise ValueError('reference_index must identify an observation.')
        self.n_deformations = n_deformations
        self.reference_index = reference_index

    def forward(self, observations, geometry, nominal_geometry):
        if observations.ndim != 4:
            raise ValueError('Expected observations [B,4*M,NB,NU].')
        b, c, nb, nu = observations.shape
        m = self.n_deformations
        if c != 4*m or geometry.shape != (b,m,nb,nu,6):
            raise ValueError('Observation/geometry shapes do not agree.')
        if nominal_geometry.shape != (nb,nu,6):
            raise ValueError('Nominal geometry must have shape [NB,NU,6].')
        y = observations.reshape(b,m,4,nb,nu)
        yr = y[:,self.reference_index:self.reference_index+1]
        gr = geometry[:,self.reference_index:self.reference_index+1]
        g0 = nominal_geometry[None,None].expand(b,m,-1,-1,-1)
        channels = torch.cat((yr.expand_as(y), y-yr), dim=2)
        descriptors = torch.cat((g0, (gr-g0), geometry-gr), dim=-1)
        return channels, descriptors
