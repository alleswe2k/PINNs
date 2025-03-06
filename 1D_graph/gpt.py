import torch
from torch_geometric.data import Data

import torch.nn as nn
import torch_geometric.nn as pyg_nn

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import numpy as np

print(torch.cuda.get_device_name(torch.cuda.current_device()))  # GPU name

# CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
L = 1.0
EI = 1
Q = -2000

num_nodes = 10
dx = L / (num_nodes - 1) # Distance between nodes

edge_index = []
for i in range(num_nodes - 1):
    edge_index.append([i, i + 1]) # Forward connection
    edge_index.append([i + 1, i]) # Backward connection (undirected graph)

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor([dx] * edge_index.shape[1], dtype=torch.float).view(-1, 1)

node_pos = torch.tensor([[i * dx] for i in range(num_nodes)], dtype=torch.float).requires_grad_()
bc = torch.zeros((num_nodes, 1))
forces = torch.full((num_nodes, 1), Q)
lbc = (node_pos == 0)
rbc = (node_pos == 1)

bc[lbc] = 1
bc[rbc] = 2

forces[lbc] = 0

x = torch.cat((node_pos, bc, forces), dim=1)
print(x)

beam_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(beam_graph)


# Define the GNN model
class BeamGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BeamGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.conv3 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.conv4 = pyg_nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = nn.functional.tanh(self.conv1(x, edge_index))
        x = nn.functional.tanh(self.conv2(x, edge_index))
        x = nn.functional.tanh(self.conv3(x, edge_index))        
        x = self.conv4(x, edge_index)  # Output: Predicted displacements
        return x

class PINN_graph():
    def __init__(self, graph):
        
        self.graph = graph

        self.input_dim = 3
        self.hidden_dim = 32
        self.output_dim = 2

        self.gnn = BeamGNN(self.input_dim, self.hidden_dim, self.output_dim)

        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.01)

    def model_value(self, graph):
        out = self.gnn(graph)
        u = out[:, 0:1]
        m = out[:, 1:2]
        return u, m
    
    def loss_func(self, graph):
        u, m = self.model_value(graph)
        print(u)
        x_pos = graph.x[:,0:1]
        bc = graph.x[:, 1:2]
        q = graph.x[:, 2:3]

        lbc = (bc == 1)
        rbc = (bc == 2)

        u_x = torch.autograd.grad(u, x_pos, torch.ones_like(u), create_graph=True)[0]
        u_2x = torch.autograd.grad(u_x, x_pos, torch.ones_like(u_x), create_graph=True)[0]

        m_x = torch.autograd.grad(m, x_pos, torch.ones_like(m), create_graph=True)[0]
        m_2x = torch.autograd.grad(m_x, x_pos, torch.ones_like(m_x), create_graph=True)[0]

        bc_loss = torch.mean(u[lbc]**2) + torch.mean(u_x[lbc]**2)
        bc_loss += torch.mean(m[rbc]**2) + torch.mean(m_x[rbc]**2)

        pde_loss = torch.mean(torch.pow(m_2x + q, 2))
        pde_loss += torch.mean(torch.pow(u_2x + m, 2))

        return pde_loss, bc_loss
    
    def train(self, epochs=1000):
        self.gnn.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pde_loss, bc_loss = self.loss_func(self.graph)
            loss = pde_loss + bc_loss
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {'{:e}'.format(loss.item())}")
                print(f"PDE: {'{:e}'.format(pde_loss.item())}, BC: {'{:e}'.format(bc_loss.item())}")


model = PINN_graph(beam_graph)
print(model.gnn)


model.train(2000)