import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr, get_hankel_new

class DSC(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, h, h_tilde, m, m_tilde,gamma, Q_noise=None, R_noise=None,   eta=0.001, T=500, name="DSC", nl=False):

        super().__init__()
        self.name = name
        self.nl = nl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Register system matrices as buffers
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(C, dtype=torch.float32))
        self.register_buffer("Q", torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("Q_obs", torch.tensor(Q_obs, dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R, dtype=torch.float32))
        self.register_buffer("K", torch.tensor(lqr(A, B, Q, R), dtype=torch.float32))

        # Store noise parameters for compatibility with LQG
        if Q_noise is not None: self.register_buffer("Q_noise", torch.tensor(Q_noise, dtype=torch.float32))
        else: self.register_buffer("Q_noise", torch.eye(A.shape[0], dtype=torch.float32))
            
        if R_noise is not None: self.register_buffer("R_noise", torch.tensor(R_noise, dtype=torch.float32))
        else: self.register_buffer("R_noise", torch.eye(C.shape[0], dtype=torch.float32) * 1e-1)
        
        # Set controller parameters
        self.h = h      # filter dimension parameter (h1 in the formula)
        self.h_tilde = h     # filter dimension parameter (h2 in the formula), using same value as h
        self.m = m      # number of row eigenvectors to use
        self.m_tilde = m     # number of past y_nat to consider (m1 in the formula)
       
        self.d = A.shape[0]   # hidden state dimension
        self.m_control = B.shape[1]  # control input dimension
        self.p = C.shape[0]  # observation dimension
        
        self.eta = eta  
        self.gamma = gamma
        self.T = T

        #### FILTERS ######
        # Initialize Hankel matrix and compute spectral decomposition
        # Compute Hankel matrix for columns
        Z_m = get_hankel_new(self.m, self.gamma)
        eigvals_m, eigvecs_m = torch.linalg.eigh(Z_m)
        
        # Register eigenvalues and eigenvectors for columns
        self.register_buffer("sigma_m", eigvals_m[-self.h:].clone().detach().to(torch.float32))  # Top-h eigenvalues
        self.register_buffer("phi_m", eigvecs_m[:, -self.h:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        
        # Compute Hankel matrix for rows
        Z_m_tilde = get_hankel_new(self.m_tilde, self.gamma)
        eigvals_m_tilde, eigvecs_m_tilde = torch.linalg.eigh(Z_m_tilde)
        
        # Register eigenvalues and eigenvectors for rows
        self.register_buffer("sigma_m_tilde", eigvals_m_tilde[-self.h_tilde:].clone().detach().to(torch.float32))  # Top-H eigenvalues
        self.register_buffer("phi_m_tilde", eigvecs_m_tilde[:, -self.h_tilde:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        #### FILTERS ######


        # Precompute terms for efficiency
        # For first term: σ_j^{1/4} φ_{jl}
        sigma_j_term = torch.pow(self.sigma_m, 0.25)  # Shape: [h]
        self.register_buffer("sigma_phi_m", sigma_j_term.view(self.h, 1) * self.phi_m.T)  # Shape: [h, m]
        
        # For second term: σ_i^{1/4} φ_{ik}
        sigma_i_term = torch.pow(self.sigma_M, 0.25)  # Shape: [H]
        self.register_buffer("sigma_phi_M", sigma_i_term.view(self.H, 1) * self.phi_M.T)  # Shape: [H, M]
        
        # Second term parameters: M_{ij} for the second summation
        # TODO: figure out matrix Ms inits here
        # Shape: [m_control, H, h, n] (control dimensions, h2 filter dimensions, h1 filter dimensions, state dimensions)

        # Initialize controller matrices M_0 to M_h
        # self.M = torch.nn.ParameterList([
        #     torch.nn.Parameter(torch.zeros(self.m_control, self.p, device=self.device))
        #     for _ in range(h+1)
        # ])

        # # Initialize controller matrices M_0 to M_h
        # self.M_tilde = torch.nn.ParameterList([
        #     torch.nn.Parameter(torch.zeros(self.m_control, self.p, device=self.device))
        #     for _ in range(h+1)
        # ])

          # Initialize controller parameters
        # M_bar_0: Direct term
        self.M_bar_0 = torch.nn.Parameter(torch.zeros(self.m_control, self.p, device=self.device))
        
        # M_bar_i: First summation term
        self.M_bar = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(self.m_control, self.p, device=self.device))
            for _ in range(self.h_tilde)
        ])
        
        # M_0l: Second summation term
        self.M_0l = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(self.m_control, self.p, device=self.device))
            for _ in range(self.h + 1)  # Including l=0
        ])
        
        # M_il: Third summation term (double-indexed)
        self.M_il = torch.nn.ParameterList([
            torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(self.m_control, self.p, device=self.device))
                for _ in range(self.h + 1)  # Including l=0
            ])
            for _ in range(self.h_tilde)  # i from 1 to h_tilde
        ])

        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)


        
    def nonlinear_dynamics(self, x):
        """Apply nonlinear dynamics if nl flag is True"""
        return leaky_relu(x)


    def compute_control_vectorized(self):
        """
        Compute the control input using the updated formula:
        
        u_t = M_bar_0 * y_nat_t +
              sum_i=1^h_tilde sum_j=1^m_tilde lambda_i^(1/4) * [varphi_i]_j * M_bar_i * y_nat_{t-j} +
              sum_l=0^h sum_k=0^m sigma_l^(1/4) * [phi_l]_k * M_0l * y_nat_{t-k} +
              sum_i=1^h_tilde sum_j=1^m_tilde sum_l=0^h sum_k=0^m (sigma_l*lambda_i)^(1/4) * [phi_l]_k * [varphi_i]_j * M_il * y_nat_{t-j-k}
        """
        # Get the most recent y_nat - shape: [n, 1]
        
        # ====== Second Term Computation ======
        # For the second term, we need to loop through m2 history values
        # This is difficult to fully vectorize due to the history access
        second_term = torch.zeros(self.m_control, device=self.device)
        
        for k in range(min(self.m2, len(self.y_nat_history) - 1)):
            # Get y_{t-k+1}^{nat} - shape: [n, 1]
            y_t_minus_k_plus_1_nat = self.y_nat_history[-(k+1)]
            
            # Reshape for broadcasting - shape: [n]
            y_nat_k_flat = y_t_minus_k_plus_1_nat.view(self.d)
            
            # Compute the weighted values of M_tensor by y_nat for each state
            # Shape: [m_control, H, h, n] * [n] -> [m_control, H, h, n]
            M_tensor_weighted = self.M_tensor * y_nat_k_flat.view(1, 1, 1, self.d)
            
            # Sum over the state dimension
            # Shape: [m_control, H, h, n] -> [m_control, H, h]
            M_tensor_state_sum = M_tensor_weighted.sum(dim=3)
            
            # Get the appropriate row vector from sigma_phi_M
            # Shape: [H]
            sigma_phi_M_k = self.sigma_phi_M[:, k]
            
            # Multiply with the appropriate sigma_phi term for dimension i
            # Shape: [m_control, H, h] * [H] -> [m_control, H, h]
            M_tensor_i_weighted = M_tensor_state_sum * sigma_phi_M_k.view(1, self.H, 1)
            
            # For each l in m1, compute the contribution
            for l in range(self.m):
                # Get the appropriate column vector from sigma_phi_m
                # Shape: [h]
                sigma_phi_m_l = self.sigma_phi_m[:, l]
                
                # Multiply with the appropriate sigma_phi term for dimension j
                # Shape: [m_control, H, h] * [h] -> [m_control, H, h]
                M_tensor_j_weighted = M_tensor_i_weighted * sigma_phi_m_l.view(1, 1, self.h)
                
                # Sum over the H and h dimensions
                # Shape: [m_control, H, h] -> [m_control]
                second_term_contrib = M_tensor_j_weighted.sum(dim=(1, 2))
                
                # Add to the second term
                second_term += second_term_contrib
        
        # Combine both terms and reshape to [m_control, 1]
        u_pert = second_term.view(self.m_control, 1)
        
        return u_pert

    def run(self, initial_state=None, add_noise=False, use_control=True, num_trials=1):
        """Run the controller for T time steps"""
        self.to(self.device)

        # Store costs for multiple trials
        all_costs = torch.zeros((num_trials, self.T), dtype=torch.float32, device=self.device)

        # Weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        
        for trial in range(num_trials):
            # Initialize state
            if initial_state is not None: x = initial_state.to(self.device)
            else: x = torch.randn(self.d, 1, dtype=torch.float32, device=self.device)


            # Initialize histories for observations and controls
            u_history = [torch.zeros((self.m_control, 1), device=self.device) for _ in range(self.h+1)]
            y_history = [torch.zeros((self.p, 1), device=self.device) for _ in range(self.h+1)]
            y_nat_history = [torch.zeros((self.p, 1), device=self.device) for _ in range(self.h+1)]

            costs = torch.zeros(self.T, dtype=torch.float32, device=self.device)

            for t in range(self.T):

                # STEP 1: Get observation from current state
                y_obs = self.C @ x  # introduce the y_t as a linear projection of x
                y_history.append(y_obs)

                # Compute natural output y_nat_t = y_t - C ∑_{i=0}^t A^i B u_{t-i}
                # This is the part of the observation not affected by past controls
                y_nat_t = y_obs.clone()
                for i in range(min(t+1, len(u_history))):
                    A_power_i = torch.matrix_power(self.A, i)
                    y_nat_t -= self.C @ (A_power_i @ self.B @ u_history[-(i+1)])
                
                y_nat_history.append(y_nat_t)

                # STEP 3: Calculate control using LQR + learned perturbation compensation
                # TODO: figure out control here
                if use_control:
                    u_nominal = -self.K @ y_obs
                    u_pert = self.compute_control_vectorized()
                    u = u_nominal + u_pert + self.bias
        
                else: 
                    # If use_control is False, use zero control for compatibility
                    u = torch.zeros((self.m_control, 1), device=self.device)
                
                u_history.append(u_t)

                # STEP 4: Get or compute perturbation for next state
                if add_noise:
                        noise_dist = torch.distributions.MultivariateNormal(
                            torch.zeros(self.d, device=self.device), 
                            self.Q_noise
                        )
                        w_t = noise_dist.sample().view(-1, 1)
                else: w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)

                # STEP 5: State update for next time step
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else: x = self.A @ x + self.B @ u_t + w_t
                
                # STEP 6: Compute quadratic cost
                cost = y_obs.t() @ self.Q_obs @ y_obs.t() + u_t.t() @ self.R @ u_t
                self.losses[t] = cost.item()

                if use_control: # TODO: figure this update out 

                    # Construct loss gradient
                    # For simplicity, we use a direct approach for gradient calculation
                    grad_M = []
                    for i in range(self.h+1):
                        if i < len(y_nat_history):
                            # dL/dM_i = 2 * R * u_t * y_nat_{t-i}^T
                            dL_dM = 2.0 * (self.R @ u_t @ y_nat_history[-(i+1)].t())
                            grad_M.append(dL_dM)
                        else:
                            grad_M.append(torch.zeros_like(self.M[i]))
                    
                    # Update each M_i using gradient descent
                    for i in range(self.h+1):
                        # Apply projection to keep controllers in constraint set K
                        # For simplicity, we just clamp values
                        updated_M = self.M[i] - self.lr * grad_M[i]
                
                # Maintain fixed-length history
                if len(u_history) > self.h + t + 1: u_history.pop(0)
                if len(y_history) > self.h + t + 1: y_history.pop(0)
                if len(y_nat_history) > self.h + t + 1: y_nat_history.pop(0)
                            
          

            all_costs[trial, :] = costs

            # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses
    
    def reset(self):
        """Reset all trajectories and losses"""
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)