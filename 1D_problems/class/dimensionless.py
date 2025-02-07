import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
import os
import time


from network import PINN

np.random.seed(1234)


# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA")
else:
    device = torch.device("cpu")
    print("CPU")


EI = 11.57e10*0.1**4/12
Q = -1.0e6
L = 3.0

q1 = lambda x: Q / (24*EI) * (x**4 - 4*L*x**3 + 6*L**2*x**2) # Fixed, free, constant q
q2 = lambda x: Q / (120*EI) * (x**5/L - 10*L*x**3 + 20*L**2*x**2) # Fixed, free, increasing q
q3 = lambda x: Q / (120*EI) * (-x**5/L + 5*x**4 - 10*L*x**3 + 10*L**2*x**2) # Fixed, free, decreasing q
q4 = lambda x: Q / (24*EI) * (x**4/L - 2*x**3 + x/L**2) # Pinned, pinned, constant q
q5 = lambda x: Q / (180*EI) * (3*x**5/L**2 - 10*x**3 + 7*L**2*x) # Pinned, pinned, increasing q
q6 = lambda x: Q / (180*EI) * (-3*x**5/L**2 + 15*x**4/L - 20*x**3 + 8*L**2*x) # Pinned, pinned, decreasing q
q7 = lambda x: Q / (48*EI) * (2*x**4 - 5*L*x**3 + 3*L**2*x**2) # Fixed, pinned, constant q
exact_dict = {
    "fixed_free_constant": q1,
    "fixed_free_increasing": q2,
    "fixed_free_decreasing": q3,
    "pinned_pinned_constant": q4,
    "pinned_pinned_increasing": q5,
    "pinned_pinned_decreasing": q6,
    "fixed_pinned_constant": q7,
}

qx1 = lambda x: np.full_like(x, Q/L) # Constant
qx2 = lambda x: Q*2/L**2 * x # Increasing
qx3 = lambda x: Q*2/L**2 * (L - x) # Decrasing

epochs = 5000
n_points = 10


x_t = np.random.uniform(0, L, n_points-2)
# x_t = np.linspace(0, L, n_points-2)
x_t = np.append(x_t, [0, L])
x_t = np.sort(x_t).reshape(-1, 1)

q = qx2(x_t)

x_hat = x_t / L
q_c = np.max(np.abs(q))

q_hat = q / q_c


nodes = 20
layers = [1, nodes, nodes, nodes, 1]
lb = 'pinned'
rb = 'pinned'

model = PINN(x_hat, layers, lb, rb, q_hat, device)

start = time.time()
model.train(epochs)
end = time.time()

trainning_time = end - start
print(f"Time: {trainning_time:.3f}")