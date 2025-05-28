import torch
import torch.nn as nn
import numpy as np
import os

from typing import List, Tuple


class DNN(nn.Module):
    """
    Simple feed-forward network with tanh activations between hidden layers
    and Xavier initialization
    """
    def __init__(self, layers: List[int]) -> None:
        super(DNN, self).__init__()

        if len(layers) < 2:
            raise ValueError("`layers` must contain at least input and output sizes.")

        modules = []
        # Build hidden layers with tanh activations
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.Tanh())
        # Final output layer (no activation)
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.network = nn.Sequential(*modules)

        # Xavier initialize all layers
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class PINN():
    """
    Physics-Informed Neural Network for the Euler-Bernoulli beam equation.
    Approximates two outputs, w(x) and m(x), from input x.
    """

    VALID_BC = {"pinned", "fixed", "free"}

    def __init__(self, dnn: torch.nn.Module) -> None:
        """
        Args:
            dnn: The deep neural network used to approximate the solution.
        """

        self.track_loss = []

        # Learning rate
        self.lr_lbfgs = 0.01

        # Deep Neural Network
        self.dnn = dnn

        # Optimizer
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=self.lr_lbfgs,
            max_iter=2000,
            max_eval=2000,
            history_size=50,
            tolerance_grad=1e-10,
            tolerance_change= 1e-10
        )

        self.iter = 0

    def set_var(self, x: torch.Tensor, q: torch.Tensor, lb: str, rb: str) -> None:
        """ 
        Set input variables and boundary masks.

        Args:
            x: 1D tensor of training input locations.
            q: 1D tensor of load values at x.
            lb: left boundary condition type.
            rb: right boundary condition type.
        """
        # Validate BC strings
        for bc_name in (lb, rb):
            if bc_name not in self.VALID_BC:
                raise ValueError(
                    f"Invalid boundary condition '{bc_name}'."
                    f"Choose from {self.VALID_BC}."
                )

        self.x = x
        self.q = q
        self.lb = lb
        self.rb = rb

    def model_value(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the model's predicted output variables at given points.
        Args:
            x: Input Tensor of shape (N, 1).
        Returns:
            Tuple[Tensor, Tensor]: w(x), m(x)
        """
        out = self.dnn(x)
        return out[:, 0:1], out[:, 1:2]
    
    def boundary_cond(
        self, 
        cond: str,
        w: torch.Tensor, 
        w_x: torch.Tensor, 
        m: torch.Tensor, 
        m_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss based on a named condition.
        Args: 
            w, m: Output values from model.
            w_x, m_x: Calculated derivatives using AD
        Returns:
            Tensor: Mean squared boundary loss.
        """

        bc_loss = 0

        match cond:
            case 'pinned':
                bc_loss += w**2 + m**2
            case 'fixed':
                bc_loss += w**2 + w_x**2
            case 'free':
                bc_loss += m**2 + m_x**2
        return bc_loss

    def loss_func(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both the PDE and boundary condition loses for the given conditions.
        Args:
            x: Input Tensor for the network.
        Returns:
            Tuple[Tensor, Tensor]: PDE and boundary condition loss.
        """

        # Get network output
        w, m = self.model_value(x)

        # Compute derivatives of w(x)
        w_x = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
        w_2x = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]

        # Compute derivatives of m(x)
        m_x = torch.autograd.grad(m, x, torch.ones_like(m), create_graph=True)[0]
        m_2x = torch.autograd.grad(m_x, x, torch.ones_like(m_x), create_graph=True)[0]

        # Boundary condition loss
        bc_loss = self.boundary_cond(self.lb, w[0], w_x[0], m[0], m_x[0])
        bc_loss += self.boundary_cond(self.rb, w[-1], w_x[-1], m[-1], m_x[-1])

        # PDE loss
        pde_loss = torch.mean(torch.pow(m_2x + self.q, 2))
        pde_loss += torch.mean(torch.pow(w_2x + m, 2))

        return pde_loss, bc_loss


    def lbfgs_func(self) -> torch.Tensor:
        """
        Wrapper for L-BFGS optimizer. Called internally by PyTorch's L-BFGS step.
        Returns:
            Tensor: Total loss.
        """
        pde_loss, bc_loss = self.loss_func(self.x)
        loss = pde_loss + bc_loss

        self.optimizer_lbfgs.zero_grad()
        loss.backward()
        self.track_loss.append(loss.item())

        if self.iter % 100 == 0:
            print(f"Iter: {self.iter}, Loss: {'{:e}'.format(loss.item())}")
            print(f"PDE: {'{:e}'.format(pde_loss.item())}, BC: {'{:e}'.format(bc_loss.item())}")
        self.iter += 1
        return loss
    
    def train(self) -> None:
        """
        Train the model using L-BFGS optimizer
        """
        self.dnn.train()
        self.optimizer_lbfgs.step(self.lbfgs_func)

    
    def save_model(self, filename: str) -> None:
        """
        Saves the parameters of the trained model
        Args: The filename to save the model as, e.g., 'model.pth'
        """

        dirname = os.path.dirname(__file__)
        location = os.path.join(dirname, "models/")
        print(location)
        torch.save(self.dnn.state_dict(), location + filename)