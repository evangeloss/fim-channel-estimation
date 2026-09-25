"""Copy into tests/; run python test_geometry_modulation.py.

Tests the complete feature path, without a decoder or training dataset.
Requires model/{beamspace,dilated_cnn,encoders,geometry_modulation}.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model.beamspace import BeamspaceTransform
from model.dilated_cnn import DilatedCNN
from model.encoders import GeometryEncoder
from model.geometry_modulation import GeometryConditionedModulation


def main():
    torch.manual_seed(42)
    m_count = 8
    beamspace = BeamspaceTransform(5, 5, 5, 5)
    cnn = DilatedCNN(m_count)
    encoder = GeometryEncoder(n_deformations=m_count, input_dim=6,
                              d_model=64, hidden_dim=64, dropout=0.0)
    modulation = GeometryConditionedModulation(m_count)
    observations = torch.randn(2, 4*m_count, 25, 25, requires_grad=True)
    # Synthetic descriptors test wiring only; the dataset supplies real geometry.
    geometry = torch.randn(2, m_count, 25, 25, 6, requires_grad=True)
    features = cnn(beamspace(observations))
    embeddings = encoder.encode_grid(geometry)
    output, gamma, beta = modulation(features, embeddings, return_parameters=True)
    assert output.shape == (2, 64, 25, 25)
    assert gamma.shape == beta.shape == (2, 64, 1, 1)
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output, gamma * features + beta)

    # Hold observations fixed and alter geometry: conditioning must respond.
    with torch.no_grad():
        changed = modulation(features, encoder.encode_grid(geometry + 1.0))
        assert (changed-output).abs().max().item() > 1e-7
    (output - torch.randn_like(output)).square().mean().backward()
    for name, tensor in [('observations', observations), ('geometry', geometry)]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item(), name
        assert tensor.grad.abs().sum().item() > 0, name
    for module_name, module in [('cnn', cnn), ('encoder', encoder), ('modulation', modulation)]:
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, f'Missing gradient: {module_name}.{name}'
            assert torch.isfinite(parameter.grad).all().item(), f'Invalid gradient: {module_name}.{name}'
            assert parameter.grad.abs().sum().item() > 0, f'Zero gradient: {module_name}.{name}'

    # Identity coefficients must preserve features exactly.
    identity = GeometryConditionedModulation(m_count)
    with torch.no_grad():
        identity.conditioner[-1].weight.zero_()
        identity.conditioner[-1].bias.zero_()
        torch.testing.assert_close(identity(features.detach(), embeddings.detach()), features.detach(), rtol=0, atol=0)

    # Degenerate geometry must not create NaN gradients in the standard deviation.
    constant = torch.zeros(1, m_count, 1, 1, 64, requires_grad=True)
    modulation(torch.ones(1, 64, 2, 2), constant).sum().backward()
    assert torch.isfinite(constant.grad).all().item()
    print('GEOMETRY MODULATION SMOKE TEST PASSED')
    print('Checked full feature path, conditioning response, identity behavior, and gradients.')
    print('This does not establish estimation accuracy; the decoder and training come next.')


if __name__ == '__main__':
    main()
