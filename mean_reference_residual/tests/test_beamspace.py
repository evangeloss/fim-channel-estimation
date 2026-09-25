"""Copy into tests/test_beamspace.py; run python test_beamspace.py there."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from model.beamspace import BeamspaceTransform

def dft(n):
    indices = torch.arange(n, dtype=torch.float64)
    return torch.exp(-2j * torch.pi * indices[:, None] * indices[None, :] / n) / n**0.5


def main():
    torch.manual_seed(42)
    # Unequal, nonsquare arrays catch accidental axis swaps.
    transform = BeamspaceTransform(3, 2, 2, 4)
    x = torch.randn(2, 12, 6, 8, dtype=torch.float64, requires_grad=True)
    y = transform(x)
    torch.testing.assert_close(transform.inverse(y), x, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(y.square().sum(), x.square().sum(), atol=1e-10, rtol=1e-12)
    fb = torch.kron(dft(2), dft(3))
    fu = torch.kron(dft(4), dft(2))
    # Check every deformation and both subcarriers independently.
    for start in range(0, 12, 4):
        for offset in (0, 2):
            c = start + offset
            h = torch.complex(x[:, c], x[:, c+1])
            expected = fb.conj().T @ h @ fu
            actual = torch.complex(y[:, c], y[:, c+1])
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # A single reference-array mode must occupy exactly its selected beam bin.
    h = fb[:, 4, None] @ fu[:, 3, None].conj().T
    mode = torch.zeros(1, 4, 6, 8, dtype=torch.float64)
    mode[0, 0], mode[0, 1] = h.real, h.imag
    expected = torch.zeros_like(mode)
    expected[0, 0, 4, 3] = 1
    torch.testing.assert_close(transform(mode), expected, atol=1e-12, rtol=1e-12)
    y.square().sum().backward()
    torch.testing.assert_close(x.grad, 2*x.detach(), atol=1e-11, rtol=1e-11)
    actual_size = BeamspaceTransform(5, 5, 5, 5)
    sample = torch.randn(1, 32, 25, 25)
    torch.testing.assert_close(actual_size.inverse(actual_size(sample)), sample, atol=2e-6, rtol=2e-5)
    print('BEAMSPACE SMOKE TEST PASSED')
    print('Checked inverse, energy, explicit DFT, ordering, beam bin, gradients, and 5x5 arrays.')


if __name__ == '__main__':
    main()
