import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu
from scipy.linalg import solve_discrete_are

class GRC(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, Q_noise, R_noise, h=3, T=100, lr=0.01, name="GRC", nl=False):
        super().__init__()
        self.name = name
        self.nl = nl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert arrays to tensors on correct device
        A = torch.tensor(A, dtype=torch.float32, device=self.device)
        B = torch.tensor(B, dtype=torch.float32, device=self.device)
        C = torch.tensor(C, dtype=torch.float32, device=self.device)
        Q = torch.tensor(Q, dtype=torch.float32, device=self.device)
        Q_obs = torch.tensor(Q_obs, dtype=torch.float32, device=self.device)
        R = torch.tensor(R, dtype=torch.float32, device=self.device)
        Q_noise = torch.tensor(Q_noise, dtype=torch.float32, device=self.device)
        R_noise = torch.tensor(R_noise, dtype=torch.float32, device=self.device)

        # Register system matrices as buffers
        self.register_buffer("A", A) 
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("Q", Q)
        self.register_buffer("Q_obs", Q_obs)
        self.register_buffer("R", R)
        self.register_buffer("Q_noise", Q_noise)
        self.register_buffer("R_noise", R_noise)

        self.d = A.shape[0]          # hidden state dimension
        self.m_control = B.shape[1]  # control input dimension
        self.p = C.shape[0]          # observation dimension
        
        # GRC specific parameters
        self.h = h  # history length for the controller
        self.lr = lr  # learning rate for updating M matrices
        
        # Initialize M matrices (controller parameters to be learned)
        # M_i maps from observation y_t-i to control input u_t
        self.M = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 0.1)
            for _ in range(h + 1)  # M_0 to M_h
        ])

        # Set up optimizer for M matrices
        self.optimizer = torch.optim.SGD(self.M.parameters(), lr=self.lr)
        
        self.T = T
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

    def nonlinear_dynamics(self, x):
        """Apply nonlinear transformation to state if nonlinear mode is enabled"""
        return leaky_relu(x)

    def compute_loss(self, y_obs, u_t):
        """
        Standard quadratic cost function: y^T Q_obs y + u^T R u
        """
        return y_obs.t() @ self.Q_obs @ y_obs + u_t.t() @ self.R @ u_t

    def compute_natural_observation(self, y_history, u_history):
        """
        Compute y^nat_t = y_t - C ∑_{i=0}^{t} A^i B u_{t-i}
        This removes the effects of control from the observation
        """
        t = len(y_history) - 1
        
        # Start with the current observation
        y_nat = y_history[-1].clone()
        
        # Subtract the effect of previous controls
        for i in range(min(t+1, len(u_history))):
            if i >= len(u_history):
                break
                
            # Compute A^i
            A_i = torch.eye(self.d, device=self.device)
            for _ in range(i):
                A_i = self.A @ A_i
                
            # Subtract C * A^i * B * u_{t-i}
            y_nat = y_nat - self.C @ A_i @ self.B @ u_history[-(i+1)]
            
        return y_nat

    def compute_control(self, y_nat_history):
        """
        Compute control u_t = ∑_{i=0}^{h} M_i y^nat_{t-i}
        """
        u_t = torch.zeros(self.m_control, 1, device=self.device)
        
        # Sum the contributions from each history step
        for i in range(min(self.h + 1, len(y_nat_history))):
            if i >= len(y_nat_history):
                break
            u_t = u_t + self.M[i] @ y_nat_history[-(i+1)]
            
        return u_t

    def compute_proxy_loss(self, y_nat, u_t):
        """
        Compute the proxy loss for GRC: ℓ_t(M_{0:h}) = c_t(y_t(M_{0:h}), u_t(M_{0:h}))
        
        This is a key part of the GRC algorithm - we use the natural observation y^nat
        instead of the actual observation in the loss calculation.
        """
        # Proxy loss using natural observation (y^nat) instead of actual observation (y)
        return y_nat.t() @ self.Q_obs @ y_nat + u_t.t() @ self.R @ u_t

    def update_M_matrices(self, y_nat_history):
        """
        Update M matrices using gradient descent:
        M^{t+1}_{0:h} ← Π_K [M^t_{0:h} - η∇ℓ_t(M^t_{0:h})]
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Create tensor versions of y_nat_history for gradient computation
        y_nat_tensors = [y.clone().detach().requires_grad_(True) for y in y_nat_history]
        
        # Compute the control using current M matrices
        u_t = self.compute_control(y_nat_tensors)
        
        # Compute the proxy loss using natural observation (y^nat)
        # This is the key difference from the standard LQR approach
        proxy_loss = self.compute_proxy_loss(y_nat_tensors[-1], u_t)
        
        # Backpropagate
        proxy_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Return loss value for tracking
        return proxy_loss.item()

    def run(self, initial_state=None, add_noise=False, use_control=True, num_trials=1):
        self.to(self.device)
        
        # Store costs for multiple trials
        all_costs = torch.zeros((num_trials, self.T), dtype=torch.float32, device=self.device)
        
        for trial in range(num_trials):
            # Initialize true state
            if initial_state is not None:
                x = initial_state.to(self.device)
            else: x = torch.randn(self.d, 1, dtype=torch.float32, device=self.device)
            
            # Track observation and control history
            y_history = []
            y_nat_history = []
            u_history = []
            costs = torch.zeros(self.T, dtype=torch.float32, device=self.device)
            
            # Initial observation
            y_obs = self.C @ x
            y_history.append(y_obs)
            y_nat_history.append(y_obs)  # Initially, y_nat = y since no control has been applied
            
            for t in range(self.T):
                # Compute control if enabled
                if use_control and t > 0:  # Skip first step since we need history
                    u_t = self.compute_control(y_nat_history)
                else: u_t = torch.zeros((self.m_control, 1), device=self.device)
                
                # Add control to history
                u_history.append(u_t)
                #print(u_history)
                
                # Generate process noise if needed
                if add_noise:
                    noise_dist = torch.distributions.MultivariateNormal(
                        torch.zeros(self.d, device=self.device), 
                        self.Q_noise
                    )
                    w_t = noise_dist.sample().view(-1, 1)
                else: w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)
                
                # Update true state
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else:
                    x = self.A @ x + self.B @ u_t + w_t
                
                # Get new observation
                y_obs = self.C @ x
                y_history.append(y_obs)
                
                # Compute natural observation
                y_nat = self.compute_natural_observation(y_history, u_history)
                y_nat_history.append(y_nat)
                
                # Compute actual quadratic cost for this step (for evaluation)
                actual_cost = self.compute_loss(y_obs, u_t)
                costs[t] = actual_cost.item()
                
                # Update controller parameters using proxy loss
                if use_control:
                    self.update_M_matrices(y_nat_history)
            
            all_costs[trial, :] = costs
            
        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses

  
