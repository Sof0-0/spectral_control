import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr

class GRC(torch.nn.Module):
    def __init__(self, A, B, C, Q, R, h, eta=0.001, T=100, name="ModifiedGRC", nl=False):

        super().__init__()
        self.name = name
        self.nl = nl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Register system matrices as buffers
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(C, dtype=torch.float32))
        self.register_buffer("Q", torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R, dtype=torch.float32))
        self.register_buffer("K", torch.tensor(lqr(A, B, Q, R), dtype=torch.float32))
        
        # Set controller parameters
        self.h = h  # number of past y_nat values to consider
        self.n, self.m_control = B.shape  # state and control dimensions
        self.p = C.shape[0]  # output dimension
        
        self.eta = eta  
        self.T = T
        self.W_test = self._initialize_sinusoidal_disturbances()
        
        # Initialize the M matrices for control computation
        # Shape: [m_control, p, h+1] - For each control dimension, each output dimension, and each time step
        self.M = torch.nn.Parameter(torch.ones(self.m_control, self.p, self.h+1))
        self.bias = torch.nn.Parameter(torch.zeros(self.m_control, 1))
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        self.register_buffer("w_history", torch.zeros(self.h + self.h, self.n, 1))

        # Store control history for y_nat computation
        self.max_history_len = T  # Maximum length of control history to keep
        self.register_buffer("u_history", torch.zeros(self.max_history_len, self.m_control, 1))

        # Store y_nat history for control computation
        self.register_buffer("y_nat_history", torch.zeros(self.h+1, self.p, 1))  # Store h+1 y_nat values

        # Tracking variables for states and controls
        self.x_trajectory = []
        self.u_trajectory = []
        self.losses = None

    def _initialize_sinusoidal_disturbances(self):
        """Initialize sinusoidal disturbances for testing"""
        # Keep frequency = 3.0 and magnitude = 1.0 (noise example)
        magnitude = 1.0
        freq = 3.0

        t_range = torch.arange(self.T)
        sin_values = torch.sin(t_range * 2 * torch.pi * freq / self.T)
    
        w_matrix = magnitude * sin_values.repeat(self.n, 1).T
        
        # Convert to list of tensors
        W_test = [
            w_matrix[i].reshape(-1, 1)
            for i in range(w_matrix.shape[0])
        ]

        return W_test
        
    def nonlinear_dynamics(self, x):
        """Apply nonlinear dynamics if nl flag is True"""
        return leaky_relu(x)

    def compute_y_nat_vectorized(self, y_obs, t):
        """
        Compute y_nat using vectorized operations:
        y_t^nat = y_t - C∑(A^i * B * u_{t-i}) for i=0 to t
        """
        if t == 0: 
            return y_obs.clone()  # No control effect at t=0
        
        # Calculate how many previous controls to consider
        history_len = min(t + 1, self.max_history_len)
        
        # Extract relevant control history - shape: [history_len, m_control, 1]
        u_history_relevant = self.u_history[-history_len:].flip(0)
        
        # Initialize control effect
        control_effect = torch.zeros_like(y_obs, device=self.device)
        
        # Create CB matrix for i=0 case
        CB = self.C @ self.B  # Shape: [p, m_control]
        
        # Add the i=0 term
        control_effect += CB @ u_history_relevant[0]
        
        # Add remaining terms
        for i in range(1, history_len):
            # Compute A^i
            A_power_i = torch.matrix_power(self.A, i)
            # Compute C * A^i * B
            CAB = self.C @ A_power_i @ self.B  # Shape: [p, m_control]
            # Add contribution to control effect
            control_effect += CAB @ u_history_relevant[i]
            
        # Compute y_nat
        y_nat = y_obs - control_effect
        
        return y_nat

    def compute_control_vectorized(self):
        """
        Highly optimized control computation using the formula:
        u_t = sum_{i=0}^{h} M_i^t * y_{t-i}^{nat}
        """
        # Initialize control perturbation
        u_pert = torch.zeros(self.m_control, 1, device=self.device)
        
        # For each time step in the history
        for i in range(self.h + 1):
            # If we have enough history
            if i < len(self.y_nat_history):
                # Get the appropriate y_nat value
                y_nat_i = self.y_nat_history[-(i+1)]  # Shape: [p, 1]
                
                # Apply the corresponding M matrix
                # M shape: [m_control, p, h+1]
                # For each control dimension and each output dimension
                for j in range(self.p):
                    u_pert += self.M[:, j:j+1, i] * y_nat_i[j]
        
        return u_pert

    def run(self):
        """Run the controller for T time steps"""
        self.to(self.device)
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

        # Reset tracking variables
        self.x_trajectory = []
        self.u_trajectory = []
        
        # Weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)
        
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        x = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        y_obs = torch.zeros(self.p, 1, dtype=torch.float32, device=self.device)
        
        x_prev = torch.zeros_like(x)
        u_prev = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)

        for t in range(self.T):
            # Get or compute perturbation
            if self.w_test is not None and t < len(self.w_test): 
                w_t = self.w_test[t]  # Perturbation for this time step
            else:
                # If no test perturbation provided, infer it from state dynamics
                if t > 0: 
                    w_t = x - self.A @ x_prev - self.B @ u_prev
                else: 
                    w_t = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
            
            # State update
            if t > 0: 
                x = self.A @ x_prev + self.B @ u_prev + w_t
                    
            y_obs = self.C @ x  # Introduce the y_t as a linear projection of x
            
            # Update perturbation history
            self.w_history = torch.roll(self.w_history, -1, dims=0)
            self.w_history[-1] = w_t

            # Compute y_nat and update history
            y_nat = self.compute_y_nat_vectorized(y_obs, t)
            
            # Update y_nat history (roll and add new y_nat)
            with torch.no_grad():  # Prevent tracking for buffer updates
                self.y_nat_history = torch.roll(self.y_nat_history, -1, dims=0)
                self.y_nat_history[-1] = y_nat
            
            # Calculate control using LQR + learned perturbation compensation
            u_nominal = -self.K @ y_obs
            
            # Add the learned perturbation compensation using vectorized operations
            u_pert = self.compute_control_vectorized()
            
            # Total control: nominal + perturbation compensation + bias
            u = u_nominal + u_pert + self.bias
            
            # Update control history
            self.u_history = torch.roll(self.u_history, -1, dims=0)
            self.u_history[-1] = u

            # Save trajectories
            self.x_trajectory.append(x.detach().clone())
            self.u_trajectory.append(u.detach().clone())

            # Compute quadratic cost
            cost = y_obs.T @ self.Q @ y_obs + u.T @ self.R @ u
            self.losses[t] = cost.item()
            
            # Skip gradient updates until we have enough history (warm-up phase)
            if t >= self.h:
                optimizer.zero_grad()
                
                # Compute proxy loss (forward simulation of cost over horizon)
                proxy_loss = self.compute_proxy_loss()
                proxy_loss.backward()
                
                optimizer.step()
                scheduler.step()

            # Save current state and control for next iteration
            x_prev = x.clone()
            u_prev = u.clone()
    
    def compute_proxy_loss(self):
        """
        Compute the proxy loss by simulating future states and controls
        """
        y_obs_sim = torch.zeros(self.p, 1, dtype=torch.float32, device=self.device)
        proxy_cost = 0.0
        
        # Simulate dynamics for h steps ahead
        for h in range(self.h):
            # Calculate control using LQR + learned perturbation compensation
            v_nominal = -self.K @ y_obs_sim
            
            # Compute perturbation compensation using vectorized operations
            v_pert = self.compute_control_vectorized()
            
            # Total control: nominal + perturbation compensation + bias
            v = v_nominal + v_pert + self.bias
            
            # Next perturbation from history
            w_next = self.w_history[h + self.h]
            
            # Apply state transition
            state_sim = self.A @ y_obs_sim + self.B @ v + w_next
            
            # Apply state observation
            y_obs_sim = self.C @ state_sim

            # Compute cost at this horizon step
            step_cost = y_obs_sim.T @ self.Q @ y_obs_sim + v.T @ self.R @ v
            proxy_cost += step_cost
    
        return proxy_cost


def plot_loss(controller, title):
    """Plot the controller's loss over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(controller.losses.cpu().numpy())
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()