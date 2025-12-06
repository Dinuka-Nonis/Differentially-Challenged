import torch

def soft_max(x, temperature=1.0):
    """
    Smooth approximation of max(x)
    Lower temperature → sharper max
    """
    x = torch.tensor(x, dtype=torch.float32)
    return torch.sum(x * torch.softmax(x / temperature, dim=0))
