import torch

def soft_argmax(x, temperature=1.0):
    """
    Differentiable approximation of argmax.
    Returns a weighted average of indices.
    """
    x = torch.tensor(x, dtype=torch.float32)

    probs = torch.softmax(x / temperature, dim=0)
    indices = torch.arange(len(x), dtype=torch.float32)

    return torch.sum(indices * probs)
