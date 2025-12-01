import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from differentiable.smooth_abs import smooth_abs

# Create a range of x values
x = torch.linspace(-5, 5, 200)

# Try different eps values
eps_values = [1e-1, 1e-2, 1e-3, 1e-5]

plt.figure(figsize=(8, 5))

for eps in eps_values:
    y = smooth_abs(x, eps=eps)
    plt.plot(x.numpy(), y.detach().numpy(), label=f'eps={eps}')

plt.title("Smooth Absolute Value Function")
plt.xlabel("x")
plt.ylabel("smooth_abs(x)")
plt.legend()
plt.grid(True)
plt.show()
