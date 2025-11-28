import torch
from differentiable.soft_step import soft_step

def test_soft_step_forward():
    x = torch.tensor([-2.0, 0.0, 2.0])
    y = soft_step(x, k=5.0)

    # Very negative input → close to 0
    assert y[0] < 0.1

    # Around zero → around 0.5
    assert 0.4 < y[1] < 0.6

    # Positive input → close to 1
    assert y[2] > 0.9

def test_soft_step_grad():
    x = torch.tensor(1.0, requires_grad=True)
    y = soft_step(x, k=1.0)

    y.backward()

    # Gradient must exist
    assert x.grad is not None
