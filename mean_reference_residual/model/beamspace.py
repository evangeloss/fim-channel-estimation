"""Unitary reference-array beamspace transform for packed channel estimates.

Copy into model/beamspace.py. Uses PyTorch, with autograd support.
Input/output: [B, 4*M, N_B, N_U], ordered per deformation as
[real(k), imag(k), real(k+1), imag(k+1)].
Array indexing: horizontal index varies fastest (v * n_h + h).
"""
import torch
from torch import nn


class BeamspaceTransform(nn.Module):
    """Apply F_B^H H F_U; inverse applies F_B H_beam F_U^H.

    F = kron(F_vertical, F_horizontal), where F uses the negative-exponent
    unitary DFT convention. No fftshift is applied. Full bases are retained.
    Geometry does not enter this fixed reference-basis transform.
    """

    def __init__(self, n_h_b, n_v_b, n_h_u, n_v_u):
        super().__init__()
        dimensions = (n_h_b, n_v_b, n_h_u, n_v_u)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in dimensions):
            raise ValueError('Array dimensions must be positive integers.')
        self.n_h_b, self.n_v_b, self.n_h_u, self.n_v_u = dimensions
        self.n_b = n_h_b * n_v_b
        self.n_u = n_h_u * n_v_u

    def _unpack(self, observations):
        if observations.ndim != 4:
            raise ValueError('Expected [B, 4*M, N_B, N_U].')
        batch, channels, n_b, n_u = observations.shape
        if channels == 0 or channels % 4 != 0:
            raise ValueError('The channel dimension must be a positive multiple of four.')
        if (n_b, n_u) != (self.n_b, self.n_u):
            raise ValueError(f'Expected spatial shape {(self.n_b, self.n_u)}, got {(n_b, n_u)}.')
        if observations.dtype not in (torch.float32, torch.float64):
            raise TypeError('Use float32 or float64 packed real/imaginary observations.')
        packed = observations.reshape(batch, channels // 4, 4, n_b, n_u)
        complex_channels = torch.stack((
            torch.complex(packed[:, :, 0], packed[:, :, 1]),
            torch.complex(packed[:, :, 2], packed[:, :, 3]),
        ), dim=2)
        return complex_channels.reshape(
            batch, channels // 4, 2,
            self.n_v_b, self.n_h_b, self.n_v_u, self.n_h_u,
        )

    def _pack(self, channels):
        batch, m_count = channels.shape[:2]
        channels = channels.reshape(batch, m_count, 2, self.n_b, self.n_u)
        return torch.stack((channels[:, :, 0].real, channels[:, :, 0].imag,
                            channels[:, :, 1].real, channels[:, :, 1].imag),
                           dim=2).reshape(batch, 4*m_count, self.n_b, self.n_u)

    def forward(self, observations):
        channels = self._unpack(observations)
        # F_B^H on BS axes, F_U on UE axes.
        channels = torch.fft.ifftn(channels, dim=(-4, -3), norm='ortho')
        channels = torch.fft.fftn(channels, dim=(-2, -1), norm='ortho')
        return self._pack(channels)

    def inverse(self, beamspace_observations):
        channels = self._unpack(beamspace_observations)
        channels = torch.fft.fftn(channels, dim=(-4, -3), norm='ortho')
        channels = torch.fft.ifftn(channels, dim=(-2, -1), norm='ortho')
        return self._pack(channels)
