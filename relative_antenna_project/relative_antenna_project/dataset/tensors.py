import torch
from torch.utils.data import TensorDataset


def make_tensor_dataset(observations, geometry, targets):
    """Convert PyTorch-layout NumPy arrays into one three-item dataset."""
    return TensorDataset(
        torch.as_tensor(observations, dtype=torch.float32),
        torch.as_tensor(geometry, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
