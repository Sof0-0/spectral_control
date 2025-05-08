import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr, get_hankel_new

class OSC_PO(torch.nn.Module):
    def __init__(self, A, B, C,  Q, R, h, H, gamma, eta=0.001, T=100, name="GRC", nl=False):

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
        self.h = h  # filter dimension parameter for columns
        self.H = H  # filter dimension parameter for rows (FIGURE OUT WHAT THESE ARE)
        self.m = 5  # number of past y_nat to consider (columns)
        self.M = 5  # number of history rows to consider (FIGURE OUT WHAT THESE ARE)
        self.n, self.m_control = B.shape  # state and control dimensions
        self.eta = eta  
        self.gamma = gamma
        self.T = T
        self.W_test = self._initialize_sinusoidal_disturbances()

        # Initialize Hankel matrix and compute spectral decomposition
        # TODO: check dimensions of the filters below 
        # Compute Hankel matrix for columns

        ######### FILTERS ##########
        Z_m = self.get_hankel_new(self.m, self.gamma)
        eigvals_m, eigvecs_m = torch.linalg.eigh(Z_m)
        
        # Register top-h eigenvalues and eigenvectors for columns
        self.register_buffer("sigma_m", eigvals_m[-self.h:].clone().detach().to(torch.float32)) # Top-h eigenvalues
        self.register_buffer("phi_m", eigvecs_m[:, -self.h:].clone().detach().to(torch.float32)) # Corresponding eigenvectors
        
        # Compute Hankel matrix for rows
        Z_M = self.get_hankel_new(self.M, self.gamma)
        eigvals_M, eigvecs_M = torch.linalg.eigh(Z_M)
        
        # Register top-H eigenvalues and eigenvectors for rows
        self.register_buffer("sigma_M", eigvals_M[-self.H:].clone().detach().to(torch.float32)) # Top-H eigenvalues
        self.register_buffer("phi_M", eigvecs_M[:, -self.H:].clone().detach().to(torch.float32)) # Corresponding eigenvectors
        ######### FILTERS ##########


        # E has shape (m_control, n, self.m) - maps past m perturbations to control
        self.E = torch.nn.Parameter(torch.ones(self.m_control, self.n, self.m) * 1e-1) # control by state by memory (4 by 10 by 5)
        self.E_new = torch.nn.Parameter(torch.ones(self.m_control, self.n, self.m, self.M), * 1e-1) # control by state by memory1 by memory2 (4 by 10 by 5 by ?)
        self.bias = torch.nn.Parameter(torch.zeros(self.m_control, 1))
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        self.register_buffer("w_history", torch.zeros(self.h + self.m, self.n, 1))

        # Tracking variables for states and controls
        self.x_trajectory = []
        self.u_trajectory = []
        self.e_history = []

        # Store control history for y_nat computation
        self.max_history_len = T  # Maximum length of control history to keep
        self.register_buffer("u_history", torch.zeros(self.max_history_len, self.m_control, 1))

        # Store y_nat history for control computation
        self.register_buffer("y_nat_history", torch.zeros(self.m, self.C.shape[0], 1))  # Store last m y_nat values

    def _initialize_sinusoidal_disturbances(self):
       
        # Keep freqyency = 3.0 and magnitude = 1.0 (noise example)
        magnitude = 1.0
        freq = 3.0

        t_range = torch.arange(self.T)
        sin_values = torch.sin(t_range * 2 * torch.pi * freq / self.T)
    
        w_matrix = magnitude * sin_values.repeat(self.n, 1).T
        
        # Convert to list of tensors matching your original format
        W_test = [
            w_matrix[i].reshape(-1, 1)
            for i in range(w_matrix.shape[0])
        ]

        return W_test
        
    def nonlinear_dynamics(self, x):
        return leaky_relu(x)

    def compute_y_nat(self, y_obs, t):
        """
        Compute y_nat using the formula: y_t^nat = y_t - C∑(A^i * B * u_{t-i}) for i=0 to t
        """
        if t == 0: return y_obs.clone()  # No control effect at t=0
            
        # Initialize sum term
        control_effect = torch.zeros_like(y_obs, device=self.device)
        
        # Compute the sum of control effects
        for i in range(t + 1):
            if i > self.max_history_len - 1: break  # Avoid index out of bounds
                
            
            u_t_minus_i = self.u_history[-(i+1)] # get past u_ts (t-i)
            
            if i == 0: term = self.C @ self.B @ u_t_minus_i # just need C*B*u_t
            else:
                # Else, we introduce A_power_i (after first step)
                A_power_i = torch.matrix_power(self.A, i)
                term = self.C @ A_power_i @ self.B @ u_t_minus_i
                
            control_effect += term
            
        y_nat = y_obs - control_effect
        
        return y_nat
    
    def construct_Y_nat_tensor(self):
        """
        Construct Y_nat tensor from y_nat history according to the formula:
        Y_t,m,M = [y_t^nat, ..., y_{t-m+1}^nat; 
                   y_{t-1}^nat, ..., y_{t-m}^nat;
                   ...
                   y_{t-M}^nat, ..., y_{t-m-M+1}^nat] ∈ ℝ^{M×m×d}
        """
        # Initialize tensor of shape (M, m, n)
        Y_nat = torch.zeros(self.M, self.m, self.n, device=self.device)
        
        # Fill the tensor with appropriate y_nat values from history
        for i in range(self.M):
            for j in range(self.m):
                idx = i + j
                if idx < len(self.y_nat_history):
                    # Extract from history and reshape to (n,) for proper assignment
                    Y_nat[i, j, :] = self.y_nat_history[idx].squeeze()
        
        return Y_nat
    
    def construct_filter_matrices(self):
        """
        Construct the two filter matrices for filtering Y_nat tensor:
        1. [σ_1^(1/4)φ_1^T; ...; σ_h^(1/4)φ_h^T] ∈ ℝ^{h×d}
        2. [1_{M+1}; σ_1^(1/4)φ_1^T; ...; σ_H^(1/4)φ_H^T] ∈ ℝ^{(H+1)×d}
        """
        # First filter matrix (h×m) - for columns
        filter_matrix_1 = torch.zeros(self.h, self.m, device=self.device)
        for i in range(self.h):
            filter_matrix_1[i] = torch.pow(self.sigma_m[i], 0.25) * self.phi_m[:, i]
        
        # Second filter matrix ((H+1)×M) - for rows
        filter_matrix_2 = torch.zeros(self.H+1, self.M, device=self.device)
        # First row is 1_{M}
        filter_matrix_2[0] = torch.ones(self.M, device=self.device)
        # Remaining rows follow same pattern as filter_matrix_1 but with M parameters
        for i in range(self.H):
            filter_matrix_2[i+1] = torch.pow(self.sigma_M[i], 0.25) * self.phi_M[:, i]
            
        return filter_matrix_1, filter_matrix_2


    # TODO: check the following function
    def filter_Y_nat(self, Y_nat):
        """
        Filter Y_nat tensor using the two filter matrices to get Y_nat^fltr
        Y_t,h,H^{nat,fltr} = Y_t,m,M^nat ×₁ filter_matrix_1 ×₂ filter_matrix_2 ∈ ℝ^{h×(H+1)×d}
        """
        filter_matrix_1, filter_matrix_2 = self.construct_filter_matrices()
        
        # First mode multiplication (along m dimension) - results in tensor of shape (M, h, d)
        # For each of the M matrices in Y_nat, we multiply from the left by filter_matrix_1
        intermediate = torch.zeros(self.M, self.h, self.n, device=self.device)
        for i in range(self.M):
            intermediate[i] = filter_matrix_1 @ Y_nat[i]
        
        # Second mode multiplication (along M dimension) - results in tensor of shape (h, H+1, d)
        # For each of the h matrices in intermediate (transposed), we multiply from the left by filter_matrix_2
        Y_nat_filtered = torch.zeros(self.h, self.H+1, self.n, device=self.device)
        for i in range(self.h):
            Y_nat_filtered[i] = filter_matrix_2 @ intermediate[:, i, :]
            
        return Y_nat_filtered
        
  
    def run(self):
        
        self.to(self.device)
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

        # Reset tracking variables
        self.x_trajectory = []
        self.u_trajectory = []
        self.e_history = []
        
        # weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)
        
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
      
        x = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        y_obs = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        y_nat = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        
        x_prev = torch.zeros_like(x)
        u_prev = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
        

        # Save initial E values
        #self.e_history.append(self.E.detach().clone())

        for t in range(self.T):
            
            if self.w_test is not None and t < len(self.w_test): w_t = self.w_test[t] # perturbation for this time step
            else:
                # If no test perturbation provided, infer it from state dynamics
                if t > 0: w_t = x - self.A @ x_prev - self.B @ u_prev
                else: w_t = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
            
            # State update:
            if t > 0: x = self.A @ x_prev + self.B @ u_prev + w_t
                    
            y_obs = self.C @ x # introduce the y_t as a linear projection of x
            
            # Update perturbation history (roll and add new perturbation)
            #with torch.no_grad():
            self.w_history = torch.roll(self.w_history, -1, dims=0)
            self.w_history[-1] = w_t

            y_nat = self.compute_y_nat(y_obs, t) # get y_nat

            # Update y_nat history (roll and add new y_nat)
            with torch.no_grad():  # Prevent tracking for buffer updates
                self.y_nat_history = torch.roll(self.y_nat_history, -1, dims=0)
                self.y_nat_history[-1] = y_nat

            # Construct Y_nat tensor and apply filtering
            Y_nat_tensor = self.construct_Y_nat_tensor()
            Y_nat_filtered = self.filter_Y_nat(Y_nat_tensor)
            
            # Calculate control using LQR + learned perturbation compensation
            u_nominal = -self.K @ y_obs  # updated with y_nat
            
            # Add the learned perturbation compensation using the most recent m perturbations
            u_pert = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
            
            #y_flat = Y_nat_filtered.view(-1)  # Flatten for easier processing (TODO: see if needed to view to multiply by E)
            ##### MAIN STEP 
             # Sum over dimensions using the E matrix with corrected dimensions
            for n_idx in range(self.n):
                for i in range(self.m):
                    for j in range(self.M):
                        # Map indices to filtered tensor using learned E matrix
                        if i < self.h and j < self.H+1:
                            for state_idx in range(self.n):
                                u_pert[:, 0] += self.E[:, n_idx, i, j] * Y_nat_filtered[i, j, state_idx]

            ##### MAIN STEP END
            
            # Total control: nominal + perturbation compensation + bias
            u = u_nominal + u_pert + self.bias
            
            #self.u_trajectory.append(u.detach().clone())

            # Update control history (roll and add new control)
            #with torch.no_grad():  # Prevent tracking for buffer updates
            self.u_history = torch.roll(self.u_history, -1, dims=0)
            self.u_history[-1] = u

            # Compute quadratic cost
            cost = y_obs.T @ self.Q @ y_obs + u.T @ self.R @ u
            self.losses[t] = cost.item()
            
            # Skip gradient updates until we have enough history (warm-up phase)
            if t >= self.h + self.m:
                optimizer.zero_grad()
                
                # Compute proxy loss (forward simulation of cost over horizon)
                proxy_loss = self.compute_proxy_loss()
                proxy_loss.backward()
                
                optimizer.step()
                scheduler.step()
                

                with torch.no_grad():
                    total_norm = torch.norm(self.E)
                    max_norm = 10.0 
                    if total_norm > max_norm:
                        self.E *= max_norm / total_norm
            

            #self.e_history.append(self.E.detach().clone())

            # Save current state and control for next iteration
            x_prev = x.clone()
            u_prev = u.clone()
    
    
    # TODO: rewrite below based on new code above
    def compute_proxy_loss(self):
        """
        Compute the proxy loss by simulating future states and controls
        """

        y_obs_sim = torch.zeros_like(self.A[:, 0:1])
        proxy_cost = 0.0

        # Get current Y_nat tensor and its filtered version
        Y_nat_tensor = self.construct_Y_nat_tensor() # construct an empty tensor to hold information
        Y_nat_filtered = self.filter_Y_nat(Y_nat_tensor)
        
        # Simulate dynamics for h steps ahead
        for h in range(self.h):
            # Compute y_nat for this simulation step
            #sim_time_step = t + h
            #y_nat_sim = self.compute_y_nat(y_obs_sim, min(sim_time_step, t))

            # TODO: figure out if I should keep Y_nat_filtered history? 
            y_nat_hist_window = self.y_nat_history[-self.m:]  # Get the last m y_nat values
            
            
            # Calculate control using LQR + learned perturbation compensation
            v_nominal = -self.K @ y_obs_sim
            
            # Compute perturbation compensation
            v_pert = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
            
            # MAIN STEP - similar to run() method
            # Use filtered Y_nat for control computation - similar to run() method
           
            for i in range(self.m): # for each memory 
                for j in range(self.M): # for each submemory in y_nat_hist
                    # Map indices to filtered tensor using learned E matrix
                    if i < self.h and j < self.H+1:
                        for state_idx in range(self.n): # for each state
                            # TODO: should I use y_nat_filtered history?
                            v_pert[:, 0] += self.E[:, state_idx, i, j] * Y_nat_filtered[i, j, state_idx]
            
            # Total control: nominal + perturbation compensation + bias
            v = v_nominal + v_pert + self.bias
            
            w_next = self.w_history[h + self.m]
            y_obs_sim = self.A @ y_obs_sim + self.B @ v + w_next
            y_obs_sim = self.C @ y_obs_sim

            Y_nat_tensor = self.construct_Y_nat_tensor(y_obs_sim) # use the new y_obs_sim to get a tensor 
            Y_nat_filtered = self.filter_Y_nat(Y_nat_tensor)


            # Compute cost at this horizon step
            step_cost = y_obs_sim.T @ self.Q @ y_obs_sim + v.T @ self.R @ v
            proxy_cost += step_cost
    
        return proxy_cost
    

def plot_loss(controller, title):
    plt.figure(figsize=(10, 6))
    plt.plot(controller.losses.cpu().numpy())
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()