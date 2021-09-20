import torch


def truncated_relu(x):
    return torch.clamp(x, min=0, max=1)