from pinn import PINN, DNN

import torch
import numpy as np

# Use CPU for this one
device = torch.device("cpu")
filename = "fixed_free_constant.pth"

# Beam parameters
L = 3.0     # Length of beam in m
Q = -20000.0    # Applied external load in N/m


# Creating training inputs and model parameters
p = 1000
x = np.random.uniform(0, L, p)
x = np.append(x, [0, L]) # Add boundary points
x = np.sort(x).reshape(-1, 1)
x_hat = x / L

q = np.full_like(x, Q) 
q0 = np.max(np.abs(q))
q_hat = q / q0
lb = 'fixed'
rb = 'free'

x_t = torch.tensor(x_hat, device=device, requires_grad=True).float()
q_t = torch.tensor(q_hat, device=device).float()
layers = [1] + 3*[20] + [2]

# Create a Deep Neural Network with the choosen size
model = DNN(layers).to(device)
pinn = PINN(model)
pinn.set_var(x_t, q_t, lb, rb)

# Train the model
pinn.train()

# Save the model
pinn.save_model(filename)