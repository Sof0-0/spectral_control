import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu
from scipy.linalg import solve_discrete_are

from PO.utils import lqr, get_hankel_new

class DSC(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, Q_noise, R_noise, h=3, m=None, T=100, lr=0.01, name="DSC", nl=False, gamma=0.1):
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
        # Set m to be larger than h if not specified
        self.m = m if m is not None else h + 2  
        self.lr = lr  # learning rate for updating M matrices
        self.gamma = gamma  # Discount factor for Hankel matrix

        #### NOISE PARAMS ####
        self.noise_mode = "sinusoid"  # Options: "gaussian", "sinusoid"
        self.sin_freq = 0.1  # Frequency of the sinusoid
        self.sin_amplitude = 0.5  # Amplitude of the sinusoid
        self.sin_phase = torch.rand(self.d, device=self.device) * 2 * np.pi  # Random phase per dimension
        #### NOISE PARAMS ####
        
        # Compute Hankel matrix eigendecomposition first to know dimensions
        Z_m = get_hankel_new(self.m+1, self.gamma)
        eigvals_m, eigvecs_m = torch.linalg.eigh(Z_m)
        
        # Register eigenvalues and eigenvectors
        self.register_buffer("sigma", eigvals_m[-(self.h+1):].clone().detach().to(torch.float32))  # Top-h eigenvalues
        self.register_buffer("phi", eigvecs_m[:,-(self.h+1):].clone().detach().to(torch.float32))  # Corresponding eigenvectors

        
        # Initialize M_bar matrix (first term in formula)
        self.M_bar = torch.nn.Parameter(
            torch.ones(self.m_control, self.p, device=self.device) * 0.01
        )
        
        # Initialize M matrices for the formula
        # M lives in h+1 x h x n x p where n is control dimension and p is observation dimension
        num_eigenvalues = len(self.sigma)
        self.M = torch.nn.Parameter(
            torch.ones(self.h+1, num_eigenvalues, self.m_control, self.p, device=self.device) * 0.01
        )
        
        # Set up optimizer for M matrices
        self.optimizer = torch.optim.SGD([self.M_bar, self.M], lr=self.lr)
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
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
        Compute control using the formula from the screenshot:
        
        u_t = M_bar^T y_nat_t + 
              sum_i=1^h sum_j=1^m lambda_i^(1/4) [phi_i]_j M_ij^T y_nat_t-j +
              sum_l=0^h sum_k=0^m sigma_l^(1/4) [phi_l]_k M_0l^T y_nat_t-k +
              sum_i=1^h sum_j=1^m sum_l=0^h sum_k=0^m (sigma_l*lambda_i)^(1/4) [phi_l]_k [phi_i]_j M_il^T y_nat_t-j-k
        """
        # First term: M_bar^T y_nat_t
        u_t = self.M_bar @ y_nat_history[-1]
        
        # Calculate available history and make sure we don't exceed index bounds
        available_history = len(y_nat_history)
        
        # Limit indices to what's available
        h_actual = min(self.h, len(self.sigma))
        m_actual = min(self.m, self.phi.shape[0]-1)  # -1 to avoid index errors

        # Second term: sum_i=1^h sum_j=1^m lambda_i^(1/4) [phi_i]_j M_ij^T y_nat_t-j
        for i in range(h_actual):
            for j in range(min(m_actual, available_history-1)):
                # Ensure indices are in bounds for sigma and phi
                if i < len(self.sigma) and j < self.phi.shape[0]:
                    lambda_i = self.sigma[i]
                    phi_i_j = self.phi[j, i] if i < self.phi.shape[1] else 0.0
                    
                    # Make sure M indices are in bounds
                    if i+1 < self.M.shape[0] and j < self.M.shape[1]:
                        # Add the contribution (j+1 to get the right historical observation)
                        if j+1 < available_history:
                            u_t = u_t + lambda_i.pow(0.25) * phi_i_j * (self.M[i+1, j] @ y_nat_history[-(j+1)])
        
        # Third term: sum_l=0^h sum_k=0^m sigma_l^(1/4) [phi_l]_k M_0l^T y_nat_t-k
        for l in range(min(h_actual+1, len(self.sigma))):
            for k in range(min(m_actual+1, available_history)):
                # Ensure indices are in bounds
                if l < len(self.sigma) and k < self.phi.shape[0]:
                    sigma_l = self.sigma[l]
                    phi_l_k = self.phi[k, l] if l < self.phi.shape[1] else 0.0
                    
                    # Make sure M indices are in bounds
                    if 0 < self.M.shape[0] and l < self.M.shape[1]:
                        # Handle the index for the historical observation
                        hist_idx = max(1, k) if k > 0 else 1
                        if hist_idx < available_history:
                            u_t = u_t + sigma_l.pow(0.25) * phi_l_k * (self.M[0, l] @ y_nat_history[-hist_idx])
        
        # Fourth term: sum_i=1^h sum_j=1^m sum_l=0^h sum_k=0^m (sigma_l*lambda_i)^(1/4) [phi_l]_k [phi_i]_j M_il^T y_nat_t-j-k
        for i in range(1, min(h_actual+1, self.M.shape[0])):
            for j in range(1, min(m_actual+1, available_history)):
                for l in range(min(h_actual+1, self.M.shape[1])):
                    for k in range(min(m_actual+1, available_history - j)):
                        # Check indices are in bounds
                        if i-1 < len(self.sigma) and l < len(self.sigma):
                            if j-1 < self.phi.shape[0] and k < self.phi.shape[0]:
                                # Get eigenvalues and eigenvectors with bounds checking
                                lambda_i = self.sigma[i-1]
                                sigma_l = self.sigma[l]
                                
                                phi_i_j = self.phi[j-1, i-1] if i-1 < self.phi.shape[1] else 0.0
                                phi_l_k = self.phi[k, l] if l < self.phi.shape[1] else 0.0
                                
                                # Compute the combined factor
                                combined_factor = (sigma_l * lambda_i).pow(0.25) * phi_l_k * phi_i_j
                                
                                # Add the contribution with bounds checking for history
                                combined_idx = j + k
                                if combined_idx < available_history:
                                    u_t = u_t + combined_factor * (self.M[i, l] @ y_nat_history[-combined_idx])
        
        return u_t

    def compute_proxy_loss(self, y_nat, u_t):
        """
        Compute the proxy loss for GRC: ℓ_t(M_{0:h}) = c_t(y_t(M_{0:h}), u_t(M_{0:h}))
        
        This is a key part of the GRC algorithm - we use the natural observation y^nat
        instead of the actual observation in the loss calculation.
        """
        # Proxy loss using natural observation (y^nat) instead of actual observation (y)
        return y_nat.t() @ self.Q_obs @ y_nat + u_t.t() @ self.R @ u_t


    def compute_predicted_state(self, y_nat_current, u_history):
        """
        Compute predicted state by adding the effect of past controls to the natural observation
        Similar to: final_state = y_nat[0] + jnp.tensordot(G, us, axes=([0, 2], [0, 1]))
        """
        # Start with current natural observation
        predicted_state = y_nat_current.clone()
        
        # Construct the G matrix (similar to DRC's G)
        # G represents the effect of past controls on the current state
        G = []
        A_power = torch.eye(self.d, device=self.device)
        for i in range(min(len(u_history), self.h)):
            G.append(self.C @ A_power @ self.B)
            A_power = A_power @ self.A
        
        # Add effect of past controls
        for i in range(min(len(u_history), self.h)):
            if i >= len(u_history): break
            
            predicted_state = predicted_state + G[i] @ u_history[-(i+1)]
        
        return predicted_state

    def compute_action_from_state(self, state, y_nat_history):
        """
        Compute action given a specific state and natural observation history
        Now uses the new control computation formula
        """
        return self.compute_control(y_nat_history)

    def compute_rollout_loss(self, y_nat, u_history, y_nat_history):
        """
        Compute the proxy loss for GRC matching the JAX implementation
        """
        # Predict the state by adding the effect of past controls
        predicted_state = self.compute_predicted_state(y_nat, u_history)
        
        # Compute the action for this predicted state
        predicted_action = self.compute_action_from_state(predicted_state, y_nat_history)
        
        # Compute the loss using the predicted state and action
        return predicted_state.t() @ self.Q_obs @ predicted_state + predicted_action.t() @ self.R @ predicted_action
       

    def update_M_matrices(self, y_nat_history, u_history):
        """
        Update M matrices using gradient descent:
        M^{t+1}_{0:h} ← Π_K [M^t_{0:h} - η∇ℓ_t(M^t_{0:h})]
        """
        # Zero gradients
        self.optimizer.zero_grad()

        y_nat_current = y_nat_history[-1].detach().clone()
        
        # Compute the proxy loss using the rollout method
        proxy_loss = self.compute_rollout_loss(y_nat_current, u_history, y_nat_history)
        
        # Backpropagate
        proxy_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
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
                u_history.append(u_t.detach())
                #print(u_history)
                #print(len(y_nat_history))
                
                # Generate process noise if needed
                if add_noise:
                    if self.noise_mode == "gaussian":
                        noise_dist = torch.distributions.MultivariateNormal(
                            torch.zeros(self.d, device=self.device), 
                            self.Q_noise
                        ) #multivariate normal (Gaussian) distribution
                        w_t = noise_dist.sample().view(-1, 1) 
                    elif self.noise_mode == "sinusoid":
                        t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)  # current timestep
                        sinusoid = self.sin_amplitude * torch.sin(2 * np.pi * self.sin_freq * t_tensor + self.sin_phase)
                        w_t = sinusoid.view(-1, 1)

                else:
                    w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)
                
                # Update true state
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else:
                    x = self.A @ x + self.B @ u_t + w_t
                
                # Get new observation
                y_obs = self.C @ x
                y_history.append(y_obs.detach())
                
                # Compute natural observation
                y_nat = self.compute_natural_observation(y_history, u_history)
                y_nat_history.append(y_nat.detach())
                
                # Compute actual quadratic cost for this step (for evaluation)
                actual_cost = self.compute_loss(y_obs, u_t)
                costs[t] = actual_cost.item()
                
                # Update controller parameters using proxy loss
                if use_control and t >= 0:
                    self.update_M_matrices(y_nat_history, u_history)
            
            all_costs[trial, :] = costs
            
        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses