from pinn_class import DNN
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# Create DNN
layers = [1] + 3*[20] + [2]
model = DNN(layers)

# For importing the model file
dirname = os.path.dirname(__file__)
location = os.path.join(dirname, "models/")
model_parameters = location + "fixed_free_constant.pth"


# Load the parameters into the DNN
model.load_state_dict(torch.load(model_parameters, weights_only=True))

# Beam parameters
E = 70e9    # Young's modulus
I = 0.1**4 / 12     # Square cross section with lenght 0.1m
EI = E * I
L = 3.0     # Length of beam in m
Q = -20000.0    # Applied external load in N/m

# Exact analytical solutions
q1 = lambda x: Q / (24*EI) * (x**4 - 4*L*x**3 + 6*L**2*x**2) # Fixed, free, constant q
q2 = lambda x: Q / (48*EI) * (2*x**4 - 5*L*x**3 + 3*L**2*x**2) # Fixed, pinned, constant q
q3 = lambda x: Q / (24*EI) * (x**4 - 2*x**3*L + x*L**3) # Pinned, pinned, constant q


x = np.linspace(0, L, 200).reshape(-1, 1)
q = np.full_like(x, Q) 
q0 = np.max(np.abs(q))
x_t = torch.tensor(x/L, dtype=torch.float)
out = model(x_t)
out = out.detach().cpu().numpy()
w = out[:, 0:1]

w_pred = (L**4 * q0 / EI) * w

w_exact = q1(x)
w_error = np.linalg.norm(w_pred - w_exact) / np.linalg.norm(w_exact,2)
error = w_pred - w_exact
print(f"L2 Error: {w_error:.3e}")

# Plotting the prediction, exact solution and absolute error
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

ax1.plot(x, w_exact, label="Exact Solution", color="navy", zorder=1)
ax1.scatter(x[::4], w_pred[::4], label="PINN solution", s=6, color="orange", zorder=2)
ax1.set_ylabel('Displacement (m)', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid()

ax2.plot(x, error, 'tab:red')
ax2.set_xlabel('x (m)', fontsize=12)
ax2.set_ylabel('Error', fontsize=12)
ax2.grid()

# Tick label font size
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))  # Defines when to switch to scientific notation

ax2.yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()