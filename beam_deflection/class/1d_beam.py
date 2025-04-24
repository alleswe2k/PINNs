from dnn import PINN

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Use CPU for this one
device = torch.device("cpu")

E = 70e9
I = 0.1**4 / 12 # Square cross section with lenght 0.1m
EI = E * I
L = 3.0
Q = -20000.0

qx1 = lambda x: np.full_like(x, Q) # Constant
qx2 = lambda x: Q/L * x # Increasing
qx3 = lambda x: Q/L * (L - x) # Decrasing


# Model
p = 1000
x = np.random.uniform(0, L, p)
x = np.append(x, [0, L]) # Add boundary points
x = np.sort(x).reshape(-1, 1)
x_hat = x / L

q = qx1(x)
q0 = np.max(np.abs(q))
q_hat = q / q0
lb = 'fixed'
rb = 'free'

x_t = torch.tensor(x_hat, device=device, requires_grad=True).float()
q_t = torch.tensor(q_hat, device=device).float()
layers = [1] + 3*[20] + [2]

model = PINN(layers, x_t, q_t, lb, rb)
model.train(1000)