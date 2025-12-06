import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 400)

# Hard max (ReLU style)
hard_max = torch.maximum(x, torch.tensor(0.0))

# Soft max (smooth)
temperature = 0.5
soft_max = temperature * torch.log(1 + torch.exp(x / temperature))

plt.plot(x, hard_max, label="Hard max (ReLU)")
plt.plot(x, soft_max, label="Soft max (smooth)", linestyle="--")

plt.legend()
plt.grid()
plt.title("Hard max vs Smooth max")
plt.show()
