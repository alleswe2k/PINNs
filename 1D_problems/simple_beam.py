import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA")
else:
    device = torch.device("cpu")
    print("CPU")

np.random.seed(1234)

class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    

class PINN():
    def __init__(self, X, layers, u_bc, u_x_bc, E, I, F, L):
        
        self.x = torch.tensor(X["Domain"], requires_grad=True).float().to(device)
        self.x_bc = torch.tensor(X["BC"], dtype=torch.float32, requires_grad=True).to(device)
        self.u_bc = torch.tensor(u_bc).float().to(device)
        self.u_x_bc = torch.tensor(u_x_bc).float().to(device)

        self.E = E
        self.I = I
        self.F = F
        self.L = L

        # DNN
        self.dnn = DNN(layers).to(device)

        # Optimizer
        self.optimizer_lfbgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.1,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x):
        u = self.dnn(x)
        return u

    def pde_loss(self, x):
        u = self.net_u(x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        residual = self.E * self.I * u_xx - self.F * (self.L - x)
        return torch.mean(residual**2)
    
    def boundary_loss(self, x_bc, u_exact, u_x_exact):
        u_pred = self.net_u(x_bc)
        u_x_pred = torch.autograd.grad(u_pred, x_bc, torch.ones_like(u_pred), create_graph=True)[0]
        boundary = (u_pred - u_exact) + (u_x_pred - u_x_exact)
        return torch.mean(boundary**2)
    
    def loss_func(self):
        loss = self.pde_loss(self.x) + self.boundary_loss(self.x_bc, self.u_bc, self.u_x_bc)

        self.optimizer_lfbgs.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(f"Iter: {self.iter}, Loss: {loss.item():.6f}")
        return loss
    
    def train(self, epochs=1000):
        self.dnn.train()
        for epoch in range(epochs):
            loss = self.pde_loss(self.x) + 2*self.boundary_loss(self.x_bc, self.u_bc, self.u_x_bc)

            self.optimizer_adam.zero_grad()
            loss.backward()
            self.optimizer_adam.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        self.optimizer_lfbgs.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X, requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x)
        u = u.detach().cpu().numpy()
        return u
    

E = 1.0
I = 1.0
F = -1.0
L = 1.0

x = np.linspace(0, L, 1000)
layers = [1, 20, 20, 20, 1]

x_domain = np.random.choice(x, 100).reshape(-1, 1)
x_bc = np.array([0.0])
u_bc = np.array([0.0])
u_x_bc = np.array([0.0])

x_train = {"Domain": x_domain, "BC": x_bc}

model = PINN(x_train, layers, u_bc, u_x_bc, E, I, F, L)
model.train(2000)