"""Copy into model/residual_reconstruction.py."""
import torch
from torch import nn
from .beamspace import BeamspaceTransform


class ResidualReconstruction(nn.Module):
    """Recover the normalized antenna-domain H0 estimate.

    observations: original normalized ANTENNA-domain [B, 4*M, N_B, N_U].
    residual_beam: decoder output [B, 4, N_B, N_U] in BEAMSPACE.

    Dataset target = normalized H0 - self.reference(observations). Therefore:
    residual_antenna = inverse_beamspace(residual_beam)
    prediction = self.reference(observations) + residual_antenna

    Outputs remain in normalized units; this module does not infer or undo
    the dataset's sample-dependent scale. Keep that scale separately if needed.
    """

    def __init__(self, n_h_b, n_v_b, n_h_u, n_v_u, residual_reference='mean'):
        super().__init__()
        if residual_reference not in ('mean', 'first'):
            raise ValueError('residual_reference must be mean or first.')
        self.residual_reference = residual_reference
        self.beamspace = BeamspaceTransform(n_h_b, n_v_b, n_h_u, n_v_u)

    def forward(self, residual_beam, observations, return_residual=False):
        if observations.ndim != 4 or observations.shape[1] < 4 or observations.shape[1] % 4:
            raise ValueError('observations must have shape [B, 4*M, N_B, N_U].')
        expected = (observations.shape[0], 4, observations.shape[2], observations.shape[3])
        if tuple(residual_beam.shape) != expected:
            raise ValueError(f'residual_beam must have shape {expected}.')
        if observations.dtype not in (torch.float32, torch.float64):
            raise TypeError('Observations must be float32 or float64.')
        if observations.device != residual_beam.device or observations.dtype != residual_beam.dtype:
            raise ValueError('Observations and residual must share device and dtype.')
        residual_antenna = self.beamspace.inverse(residual_beam)
        prediction = self.reference(observations) + residual_antenna
        if return_residual:
            return prediction, residual_antenna
        return prediction


    def reference(self, observations):
        """Complex arithmetic mean by default; legacy first-state option."""
        if observations.ndim != 4 or observations.shape[1] < 4 or observations.shape[1] % 4:
            raise ValueError('Expected [B,4*M,N_B,N_U].')
        if self.residual_reference == 'first':
            return observations[:, :4]
        b, channels, nb, nu = observations.shape
        return observations.reshape(b, channels // 4, 4, nb, nu).mean(dim=1)
