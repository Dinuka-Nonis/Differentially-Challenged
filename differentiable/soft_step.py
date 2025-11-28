import torch

def soft_step(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    Smooth approximation of the step function using a sigmoid.
    Real step jumps from 0 to 1 instantly (not differentiable).
    This soft version transitions smoothly and allows gradients.

    soft_step(x) = sigmoid(k * x)

    k controls sharpness:
        small k = very smooth
        large k = closer to real step
    """
    return torch.sigmoid(k * x)
