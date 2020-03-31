import torch


def to_tensor(x, device, type_=torch.float):
    return torch.tensor(x, dtype=type_, device=device)