import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)

y = x*w + w**2

y.backward()

print("dy/dx =", x.grad)
print("dy/dw =", w.grad)
