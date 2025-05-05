import torch
import torch.nn as nn
import torch.optim as optim

from numbers import Real
from typing import Callable

from utils import lqr


def quad_loss(x: torch.Tensor, u: torch.Tensor) -> Real:
    return torch.sum(x.T @ x + u.T @ u)


class GPC_OLD(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor = None,
        R: torch.Tensor = None,
        K: torch.Tensor = None,
        start_time: int = 0,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], Real] = None,
        H: int = 3,
        HH: int = 2,
        lr_scale: Real = 0.005,
        decay: bool = True,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (torch.Tensor): system dynamics
            B (torch.Tensor): system dynamics
            Q (torch.Tensor): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (torch.Tensor): cost matrices (i.e. cost = x^TQx + u^TRu)
            K (torch.Tensor): Starting policy (optional). Defaults to LQR gain.
            start_time (int): 
            cost_fn (Callable[[torch.Tensor, torch.Tensor], Real]): loss function
            H (int): history of the controller
            HH (int): history of the system
            lr_scale (Real): learning rate scale
            decay (bool): whether to decay learning rate
        """
        super(GPC_OLD, self).__init__()

        cost_fn = quad_loss

        d_state, d_action = B.shape  # State & Action Dimensions

        self.A, self.B = A, B  # System Dynamics

        self.t = 0  # Time Counter (for decaying learning rate)

        self.H, self.HH = H, HH

        self.lr_scale, self.decay = lr_scale, decay

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        # TODO: need to address problem of LQR with PyTorch
        self.K = K if K is not None else lqr(A, B, Q, R)

        self.M = torch.zeros((H, d_action, d_state), dtype=torch.float32)

        # Past H + HH noises ordered increasing in time
        self.noise_history = torch.zeros((H + HH, d_state, 1), dtype=torch.float32)

        # past state and past action
        self.state, self.action = torch.zeros((d_state, 1), dtype=torch.float32), torch.zeros((d_action, 1), dtype=torch.float32)

        self.cost_fn = cost_fn



    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (torch.Tensor): current state

        Returns:
            torch.Tensor: action to take
        """
        action = self.get_action(state)
        self.update(state, action)
        return action

    def update(self, state: torch.Tensor, u: torch.Tensor) -> None:
        """
        Description: update agent internal state.

        Args:
            state (torch.Tensor): current state
            u (torch.Tensor): current action

        Returns:
            None
        """
        noise = state - self.A @ self.state - self.B @ u
        self.noise_history = torch.roll(self.noise_history, shifts=-1, dims=0)
        self.noise_history[0] = noise

        delta_M, delta_bias = self._compute_gradients(self.M, self.noise_history)

        lr = self.lr_scale
        lr *= (1 / (self.t + 1)) if self.decay else 1
        self.M -= lr * delta_M
        self.bias -= lr * delta_bias

        # update state
        self.state = state

        self.t += 1

    def _compute_gradients(self, M: torch.Tensor, noise_history: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients for the policy loss.

        Args:
            M (torch.Tensor): Model parameter
            noise_history (torch.Tensor): History of noise

        Returns:
            torch.Tensor: gradients
        """
        # Example of a simple gradient computation. Replace with actual policy loss computation
        action = -self.K @ self.state + torch.tensordot(M, noise_history, dims=([0, 2], [0, 1]))
        loss = self.cost_fn(self.state, action)
        loss.backward()
        return M.grad, None  # Placeholder for delta_bias

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Description: get action from state.

        Args:
            state (torch.Tensor): current state

        Returns:
            torch.Tensor: action
        """
        return -self.K @ state + torch.tensordot(self.M, self.noise_history, dims=([0, 2], [0, 1]))

