import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from differentiable.soft_step import soft_step

# Create inputs
x = torch.linspace(-5, 5, 100, requires_grad=True)

# Choose sharpness k
k = 5.0

# Compute soft_step outputs
y = soft_step(x, k=k)

# Compute gradients
y.sum().backward()  # sum() to allow backward through all elements
grad = x.grad

# Plot output
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.title("Soft Step Output")
plt.xlabel("x")
plt.ylabel("soft_step(x)")

# Plot gradient
plt.subplot(1, 2, 2)
plt.plot(x.detach().numpy(), grad.detach().numpy())
plt.title("Gradient of Soft Step")
plt.xlabel("x")
plt.ylabel("dy/dx")

plt.tight_layout()
plt.show()
