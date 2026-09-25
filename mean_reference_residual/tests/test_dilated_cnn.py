"""Copy into tests/ and run python test_dilated_cnn.py.

Requires model/beamspace.py and model/dilated_cnn.py. No dataset needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model.beamspace import BeamspaceTransform
from model.dilated_cnn import DilatedCNN


def check_pipeline(device):
    torch.manual_seed(42)
    beamspace = BeamspaceTransform(5, 5, 5, 5).to(device)
    cnn = DilatedCNN(n_deformations=8).to(device)
    observations = torch.randn(2, 32, 25, 25, device=device, requires_grad=True)
    observations_beam = beamspace(observations)
    observations_beam.retain_grad()
    features = cnn(observations_beam)
    assert features.shape == (2, 64, 25, 25)
    assert torch.isfinite(features).all().item()

    # A nonconstant objective verifies differentiation through both modules.
    loss = (features - torch.randn_like(features)).square().mean()
    loss.backward()
    for name, tensor in [('antenna input', observations), ('beamspace input', observations_beam)]:
        assert tensor.grad is not None, f'Missing gradient: {name}'
        assert torch.isfinite(tensor.grad).all().item(), f'Nonfinite gradient: {name}'
        assert tensor.grad.abs().sum().item() > 0, f'Zero gradient: {name}'
    for name, parameter in cnn.named_parameters():
        assert parameter.grad is not None, f'Missing parameter gradient: {name}'
        assert torch.isfinite(parameter.grad).all().item(), f'Nonfinite parameter gradient: {name}'
        assert parameter.grad.abs().sum().item() > 0, f'Zero parameter gradient: {name}'

    # Confirm ordinary optimization updates the stem using those gradients.
    before = cnn.stem[0].weight.detach().clone()
    torch.optim.SGD(cnn.parameters(), lr=0.01).step()
    assert not torch.equal(before, cnn.stem[0].weight.detach())

    with torch.no_grad():
        # Different spatial dimensions and batch size must work without pooling.
        smaller = DilatedCNN(n_deformations=3, feature_channels=16).to(device)
        result = smaller(torch.randn(1, 12, 6, 8, device=device))
        assert result.shape == (1, 16, 6, 8)
        assert torch.isfinite(result).all().item()
    try:
        cnn(torch.zeros(1, 4, 25, 25, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect deformation count was silently accepted.')
    print(f'PASS ({device}): shapes, finite values, input/parameter gradients, and optimizer update.')


if __name__ == '__main__':
    check_pipeline(torch.device('cpu'))
    if torch.cuda.is_available():
        check_pipeline(torch.device('cuda'))
    print('DILATED CNN SMOKE TEST PASSED')
    print('This checks connectivity and trainability, not channel-estimation accuracy.')
