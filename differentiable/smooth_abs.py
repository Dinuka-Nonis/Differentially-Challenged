import torch

def smooth_abs(x, eps=1e-3):
    return torch.sqrt(x * x + eps)
