import torch

def smooth_relu(x, eps=1e-3):
    """
    Smooth approximation of ReLU using softplus.

    smooth_relu(x) = eps * log(1 + exp(x / eps))

    Args:
        x (torch.Tensor): input tensor.
        eps (float): smoothing parameter.

    Returns:
        torch.Tensor: output tensor.

        When x is positive → exp becomes huge → output ≈ x

        When x is negative → exp becomes very small → output ≈ 0
    """
    return eps * torch.log1p(torch.exp(x / eps))
