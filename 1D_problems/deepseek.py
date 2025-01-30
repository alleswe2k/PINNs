import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class BeamNet(torch.nn.Module):
    def __init__(self):
        super(BeamNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Parameters
L = 1.0    # Beam length
P = 1.0    # Load at free end
EI = 1.0   # Flexural rigidity
model = BeamNet()

# Collocation points (interior)
num_collocation = 200
x_collocation = torch.rand((num_collocation, 1)) * L
x_collocation.requires_grad = True

# Boundary points (fixed and free ends)
x_fixed = torch.zeros((1, 1))  # x=0
x_free = L * torch.ones((1, 1))  # x=L

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute boundary conditions at x=0
    y_fixed = model(x_fixed)
    dy_dx_fixed = torch.autograd.grad(y_fixed, x_fixed, 
                                     create_graph=True, 
                                     grad_outputs=torch.ones_like(y_fixed))[0]
    loss_bc1 = torch.mean(y_fixed**2) + torch.mean(dy_dx_fixed**2)
    
    # Compute boundary conditions at x=L
    x_free_in = x_free.clone().requires_grad_(True)
    y_free = model(x_free_in)
    dy_dx = torch.autograd.grad(y_free, x_free_in, 
                               create_graph=True, 
                               grad_outputs=torch.ones_like(y_free))[0]
    d2y_dx2 = torch.autograd.grad(dy_dx, x_free_in, 
                                 create_graph=True, 
                                 grad_outputs=torch.ones_like(dy_dx))[0]
    d3y_dx3 = torch.autograd.grad(d2y_dx2, x_free_in, 
                                 create_graph=True, 
                                 grad_outputs=torch.ones_like(d2y_dx2))[0]
    
    # Moment (EI*d2y/dx2) and Shear (EI*d3y/dx3)
    loss_moment = (EI * d2y_dx2).pow(2).mean()  # Should be zero
    loss_shear = (EI * d3y_dx3 - P).pow(2).mean()
    loss_bc2 = loss_moment + loss_shear
    
    # Compute residual loss (Euler-Bernoulli equation: d4y/dx4 = 0)
    y_res = model(x_collocation)
    dy_dx_res = torch.autograd.grad(y_res, x_collocation, 
                                   create_graph=True, 
                                   grad_outputs=torch.ones_like(y_res))[0]
    d2y_dx2_res = torch.autograd.grad(dy_dx_res, x_collocation, 
                                     create_graph=True, 
                                     grad_outputs=torch.ones_like(dy_dx_res))[0]
    d3y_dx3_res = torch.autograd.grad(d2y_dx2_res, x_collocation, 
                                     create_graph=True, 
                                     grad_outputs=torch.ones_like(d2y_dx2_res))[0]
    d4y_dx4_res = torch.autograd.grad(d3y_dx3_res, x_collocation, 
                                     create_graph=True, 
                                     grad_outputs=torch.ones_like(d3y_dx3_res))[0]
    
    residual_loss = (d4y_dx4_res.pow(2)).mean()
    
    # Total loss
    total_loss = loss_bc1 + loss_bc2 + residual_loss
    total_loss.backward()
    optimizer.step()
    
    loss_history.append(total_loss.item())
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')

# Plot training loss
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Compare with analytical solution
x_test = torch.linspace(0, L, 100).reshape(-1, 1)
y_pred = model(x_test).detach().numpy()
y_analytical = (P * x_test**2 / (6 * EI) * (3 * L - x_test)).numpy()

plt.figure()
plt.plot(x_test, y_analytical, label='Analytical Solution', linestyle='--')
plt.plot(x_test, y_pred, label='PINN Prediction')
plt.xlabel('x')
plt.ylabel('Deflection (y)')
plt.legend()
plt.title('Cantilever Beam Deflection')
plt.show()