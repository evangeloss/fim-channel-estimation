from torch import nn


class ResidualReconstruction(nn.Module):
    """Dataset target and skip both use the mean of normalized observations."""
    def reference(self, observations):
        b,c,nb,nu = observations.shape
        if c < 4 or c % 4:
            raise ValueError('Expected 4 channels per observation.')
        return observations.reshape(b,c//4,4,nb,nu).mean(dim=1)

    def forward(self, residual, observations):
        reference = self.reference(observations)
        if residual.shape != reference.shape:
            raise ValueError('Residual must have shape [B,4,NB,NU].')
        return reference + residual
