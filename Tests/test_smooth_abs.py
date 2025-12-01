import torch
from differentiable.smooth_abs import smooth_abs

def test_smooth_abs_positive():
    x = torch.tensor([3.0])
    y = smooth_abs(x)
    assert torch.isclose(y, torch.tensor([3.0]), atol=1e-3)

def test_smooth_abs_negative():
    x = torch.tensor([-4.0])
    y = smooth_abs(x)
    assert torch.isclose(y, torch.tensor([4.0]), atol=1e-3)

def test_smooth_abs_gradient():
    x = torch.tensor([0.0], requires_grad=True)
    y = smooth_abs(x)
    y.backward()
    assert x.grad is not None
