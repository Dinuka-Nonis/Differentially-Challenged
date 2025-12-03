import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from differentiable.smooth_relu import smooth_relu

x = torch.linspace(-5, 5, 300)
eps_values = [1e-1, 1e-2, 1e-3]

plt.figure(figsize=(8,5))

for eps in eps_values:
    y = smooth_relu(x, eps=eps)
    plt.plot(x, y, label=f"eps={eps}")

plt.title("Smooth ReLU (Softplus)")
plt.xlabel("x")
plt.ylabel("smooth_relu(x)")
plt.grid(True)
plt.legend()
plt.show()
