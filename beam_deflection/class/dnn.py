import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple, Union


class DNN(nn.Module):
    """
    Simple feed-forward network with tanh activations between hidden layers
    and Xavier initialization.
    """
    def __init__(self, layers: List[int]) -> None:
        super().__init__()

        modules: List[nn.Module] = []

        # Build hidden layers with tanh activations
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.Tanh())

        # Final output layer (no activation)
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.network = nn.Sequential(*modules)

        # Xavier initialize all Linear layers
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class PINN():
    """
    Physics-Informed Neural Network for a 4th-order beam equation.
    Approximates two outputs, u(x) and m(x), from input x.
    """

    VALID_BC = {"pinned", "fixed", "free", "roller"}

    def __init__(
            self,
            device: Union[str, torch.device],
            layers: List[int],
            x: np.ndarray,
            q: np.ndarray,
            lb: str,
            rb: str
        ) -> None:

        # Validate BC strings
        for bc_name in (lb, rb):
            if bc_name not in self.VALID_BC:
                raise ValueError(
                    f"Invalid boundary condition '{bc_name}'. "
                    f"Choose from {self.VALID_BC}."
                )

        self.device = torch.device(device)

        # Convert inputs to torch tensors
        self.x = torch.tensor(x, dtype=torch.float, requires_grad=True, device=self.device).view(-1, 1)
        self.q = torch.tensor(q, dtype=torch.float, device=self.device).view(-1, 1)

        self.lb = lb
        self.rb = rb

        # Build model
        self.dnn = DNN(layers).to(self.device)

        # Two optimizers: first Adam pre-training, then LBFGS fine-tuning
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.01)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.1,
            max_iter=2000,
            max_eval=2000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )


        self.iter = 0

    def model_value(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (u(x), m(x)) from the network output."""
        out = self.dnn(x)
        return out[:, :1], out[:, 1:2]
    
    def boundary_cond(
        self,
        cond: str,
        u: torch.Tensor,
        u_x: torch.Tensor,
        m: torch.Tensor,
        m_x: torch.Tensor,
    ) -> torch.Tensor:
        
        """
        Compute the BC loss term for a single boundary point,
        based on a named condition.
        """

        bc_loss = 0

        match cond:
            case 'pinned':
                bc_loss += u**2 + m**2
            case 'fixed':
                bc_loss += u**2 + u_x**2
            case 'free':
                bc_loss += m**2 + m_x**2
            case 'roller':
                bc_loss += u_x**2 + m_x**2
        return bc_loss

    def loss_func(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PDE and BC loss terms over the interior and both boundaries.
        Returns (pde_loss, bc_loss).
        """

        # Forward pass
        u, m = self.model_value(x)

        # Compute derivatives
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        m_x = torch.autograd.grad(m, x, torch.ones_like(m), create_graph=True)[0]
        m_xx = torch.autograd.grad(m_x, x, torch.ones_like(m_x), create_graph=True)[0]

        # Boundary losses (first and last point)
        bc_loss = self.boundary_cond(self.lb, u[0], u_x[0], m[0], m_x[0])
        bc_loss += self.boundary_cond(self.rb, u[-1], u_x[-1], m[-1], m_x[-1])

        # PDE residuals
        pde_loss = torch.mean(torch.pow(m_xx + self.q, 2))
        pde_loss += torch.mean(torch.pow(u_xx + m, 2))

        return pde_loss, bc_loss


    def _lbfgs_step(self) -> torch.Tensor:
        """
        Closure for LBFGS optimizer. Returns weighted total loss.
        """
        def closure() -> torch.Tensor:
            self.optimizer_lbfgs.zero_grad()
            pde_loss, bc_loss = self.loss_func(self.x)
            loss = pde_loss + bc_loss
            loss.backward()
            if self.iter % 100 == 0:
                print(f"LBFGS Iter {self.iter}: total={loss:.3e}")
                print(f"PDE={pde_loss:.3e}, BC={bc_loss:.3e}")
            self.iter += 1
            return loss

        return closure()
    
    def train(self, epochs: int = 1000) -> None:
        """
        Train the model using Adam for a fixed number of epochs,
        then refine using LBFGS.
        
        :param epochs: Number of Adam optimizer epochs to run.
        """

        self.dnn.train()

        for epoch in range(1, epochs + 1):
            pde_loss, bc_loss = self.loss_func(self.x)
            loss = pde_loss + bc_loss

            self.optimizer_adam.zero_grad()
            loss.backward()
            self.optimizer_adam.step()

            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.3e}")
                print(f"PDE={pde_loss:.3e}, BC={bc_loss:.3e}")
        
        # Automatically switch to LBFGS after Adam is done
        self.optimizer_lbfgs.step(self._lbfgs_step)


    def predict(self, x):
        x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=self.device)
        self.dnn.eval()
        u_c, m_c = self.model_value(x)
        u_c = u_c.detach().cpu().numpy()
        m_c = m_c.detach().cpu().numpy()
        return u_c, m_c
    
    def save_model(self, location: str, filename: str) -> None:
        torch.save(self.dnn.state_dict(), location + filename)