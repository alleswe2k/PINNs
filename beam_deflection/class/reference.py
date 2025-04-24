import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Configure a global logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class DNN(nn.Module):
    """
    Simple feed-forward network with tanh activations between hidden layers
    and Xavier initialization.
    """
    def __init__(self, layers: List[int]) -> None:
        """
        :param layers: List of layer sizes.  For example [1, 20, 20, 2]
                       defines a network: Linear(1→20)–Tanh–Linear(20→20)–Tanh–Linear(20→2).
        :raises ValueError: if fewer than 2 layers are provided.
        """
        super().__init__()
        if len(layers) < 2:
            raise ValueError("`layers` must contain at least input and output sizes.")

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


class PINN:
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
        rb: str,
        w_pde: float = 1.0,
        w_bc: float = 1.0,
    ) -> None:
        """
        :param device: torch device specifier, e.g. "cpu" or "cuda:0".
        :param layers: network architecture for DNN.
        :param x: 1D array of training input locations.
        :param q: 1D array of load values at x.
        :param lb: left boundary condition type.
        :param rb: right boundary condition type.
        :param w_pde: weight for PDE loss.
        :param w_bc: weight for boundary-condition loss.
        :raises ValueError: on invalid boundary condition names.
        """
        # Validate BC strings
        for bc_name in (lb, rb):
            if bc_name not in self.VALID_BC:
                raise ValueError(
                    f"Invalid boundary condition '{bc_name}'. "
                    f"Choose from {self.VALID_BC}."
                )

        self.device = torch.device(device)
        # Convert inputs to torch tensors
        self.x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True).view(-1, 1)
        self.q = torch.tensor(q, dtype=torch.float32, device=self.device).view(-1, 1)

        self.lb = lb
        self.rb = rb

        self.w_pde = w_pde
        self.w_bc = w_bc

        # Build model
        self.dnn = DNN(layers).to(self.device)

        # Two optimizers: first Adam pre-training, then LBFGS fine-tuning
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-2)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.1,
            max_iter=2_000,
            max_eval=2_000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-15,
            line_search_fn="strong_wolfe",
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
        if cond == "pinned":
            return u**2 + m**2
        elif cond == "fixed":
            return u**2 + u_x**2
        elif cond == "free":
            return m**2 + m_x**2
        elif cond == "roller":
            return u_x**2 + m_x**2
        else:
            # Should never happen due to earlier validation
            raise RuntimeError(f"Unknown BC '{cond}' encountered during training.")

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
        bc = self.boundary_cond(self.lb, u[0], u_x[0], m[0], m_x[0])
        bc = bc + self.boundary_cond(self.rb, u[-1], u_x[-1], m[-1], m_x[-1])

        # PDE residuals squared, averaged
        pde = torch.mean((m_xx + self.q) ** 2 + (u_xx + m) ** 2)

        return pde, bc

    def _lbfgs_step(self) -> torch.Tensor:
        """
        Closure for LBFGS optimizer.  Returns weighted total loss.
        """
        def closure() -> torch.Tensor:
            self.optimizer_lbfgs.zero_grad()
            pde, bc = self.loss_func(self.x)
            loss = self.w_pde * pde + self.w_bc * bc
            loss.backward()
            if self.iter % 100 == 0:
                logger.info(
                    f"LBFGS Iter {self.iter}: total={loss:.3e}, "
                    f"PDE={pde:.3e}, BC={bc:.3e}"
                )
            self.iter += 1
            return loss

        return closure()

    def train(
        self,
        epochs: int = 1000,
        switch_to_lbfgs_after: int = None
    ) -> None:
        """
        Train first with Adam, then optionally switch to LBFGS.
        
        :param epochs: number of Adam epochs
        :param switch_to_lbfgs_after: if set, run LBFGS after this many Adam epochs
        """
        self.dnn.train()
        for epoch in range(1, epochs + 1):
            self.optimizer_adam.zero_grad()
            pde, bc = self.loss_func(self.x)
            loss = self.w_pde * pde + self.w_bc * bc
            loss.backward()
            self.optimizer_adam.step()

            if epoch % max(1, epochs // 10) == 0:
                logger.info(
                    f"Adam Epoch {epoch}/{epochs}: total={loss:.3e}, "
                    f"PDE={pde:.3e}, BC={bc:.3e}"
                )

            if switch_to_lbfgs_after and epoch == switch_to_lbfgs_after:
                logger.info("Switching to LBFGS optimizer...")
                # run a few LBFGS iterations
                for _ in range(10):
                    self._lbfgs_step()

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the network outputs for new x values.
        :returns: (u_values, m_values) as numpy arrays.
        """
        self.dnn.eval()
        with torch.no_grad():
            x_tensor = (
                torch.tensor(x, dtype=torch.float32, device=self.device)
                .view(-1, 1)
                .requires_grad_(True)
            )
            u, m = self.model_value(x_tensor)
        return u.cpu().numpy(), m.cpu().numpy()
