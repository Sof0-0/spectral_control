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


    def compute_control_vectorized(self, y_nat_history):
        """
        Compute the control input using the updated formula:
        
        u_t = M_bar_0 * y_nat_t +
              sum_i=1^h_tilde sum_j=1^m_tilde lambda_i^(1/4) * [varphi_i]_j * M_bar_i * y_nat_{t-j} +
              sum_l=0^h sum_k=0^m sigma_l^(1/4) * [phi_l]_k * M_0l * y_nat_{t-k} +
              sum_i=1^h_tilde sum_j=1^m_tilde sum_l=0^h sum_k=0^m (sigma_l*lambda_i)^(1/4) * [phi_l]_k * [varphi_i]_j * M_il * y_nat_{t-j-k}
        """
         # Make sure we have enough history
        if len(y_nat_history) < max(self.m, self.m_tilde) + 1:
            return torch.zeros(self.m_control, 1, device=self.device)

        # Current y_nat (y_nat_t)
        y_nat_t = y_nat_history[-1]
        
        # First term: M_bar_0 * y_nat_t
        first_term = self.M_bar_0 @ y_nat_t
        
        # Second term: sum_i=1^h_tilde sum_j=1^m_tilde lambda_i^(1/4) * [varphi_i]_j * M_bar_i * y_nat_{t-j}
        second_term = torch.zeros(self.m_control, 1, device=self.device)
        for i in range(self.h_tilde):
            for j in range(min(self.m_tilde, len(y_nat_history) - 1)):
                if j < len(y_nat_history) - 1:
                    y_nat_t_minus_j = y_nat_history[-(j+2)]  # -1 for zero-indexing, -1 for going back j steps
                    second_term += self.lambda_powered[i] * self.varphi[j, i] * (self.M_bar[i] @ y_nat_t_minus_j)
        
        # Third term: sum_l=0^h sum_k=0^m sigma_l^(1/4) * [phi_l]_k * M_0l * y_nat_{t-k}
        third_term = torch.zeros(self.m_control, 1, device=self.device)
        for l in range(self.h + 1):  # +1 to include l=0
            sigma_factor = 1.0 if l == 0 else self.sigma_powered[l-1]  # Special case for l=0
            for k in range(min(self.m, len(y_nat_history) - 1)):
                if k < len(y_nat_history) - 1:
                    y_nat_t_minus_k = y_nat_history[-(k+2)]  # -1 for zero-indexing, -1 for going back k steps
                    phi_factor = 1.0 if l == 0 else self.phi[k, l-1]  # Special case for l=0
                    third_term += sigma_factor * phi_factor * (self.M_0l[l] @ y_nat_t_minus_k)
        
        # Fourth term: sum_i=1^h_tilde sum_j=1^m_tilde sum_l=0^h sum_k=0^m (sigma_l*lambda_i)^(1/4) * [phi_l]_k * [varphi_i]_j * M_il * y_nat_{t-j-k}
        fourth_term = torch.zeros(self.m_control, 1, device=self.device)
        for i in range(self.h_tilde):
            for j in range(min(self.m_tilde, len(y_nat_history) - 1)):
                for l in range(self.h + 1):  # +1 to include l=0
                    for k in range(min(self.m, len(y_nat_history) - j - 1)):
                        idx = j + k + 2  # +2 for zero-indexing and for going back j+k steps
                        if idx < len(y_nat_history):
                            y_nat_t_minus_j_minus_k = y_nat_history[-idx]
                            
                            # Special cases for l=0
                            combined_factor = self.lambda_powered[i] if l == 0 else self.combined_powered[l-1, i]
                            phi_factor = 1.0 if l == 0 else self.phi[k, l-1]
                            
                            fourth_term += combined_factor * phi_factor * self.varphi[j, i] * (self.M_il[i][l] @ y_nat_t_minus_j_minus_k)
        
        # Combine all terms
        u = first_term + second_term + third_term + fourth_term
        
        return u

    def run(self, initial_state=None, add_noise=False, use_control=True, num_trials=1):
        """Run the controller for T time steps"""
        self.to(self.device)

        # Store costs for multiple trials
        all_costs = torch.zeros((num_trials, self.T), dtype=torch.float32, device=self.device)

        # Weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)

             # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease

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
                y_history.append(y_obs.detach())

                # Compute natural output y_nat_t = y_t - C ∑_{i=0}^t A^i B u_{t-i}
                # This is the part of the observation not affected by past controls
                y_nat_t = y_obs.clone()
                for i in range(min(t+1, len(u_history))):
                    A_power_i = torch.matrix_power(self.A, i)
                    y_nat_t -= self.C @ (A_power_i @ self.B @ u_history[-(i+1)])
                
                y_nat_history.append(y_nat_t.detach())

                # STEP 3: Calculate control using LQR + learned perturbation compensation
                # TODO: figure out control here
                if use_control:
                    u_nominal = -self.K @ y_obs
                    u_pert = self.compute_control_vectorized(y_nat_history)
                    u_t = u_nominal + u_pert
        
                else: 
                    # If use_control is False, use zero control for compatibility
                    u_t = torch.zeros((self.m_control, 1), device=self.device)
                
                u_history.append(u_t.detach())

                # STEP 4: Get or compute perturbation for next state
                if add_noise:
                        noise_dist = torch.distributions.MultivariateNormal(
                            torch.zeros(self.d, device=self.device), 
                            self.Q_noise
                        )
                        w_t = noise_dist.sample().view(-1, 1)
                else: w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)

                x = x.detach()  # Detach to prevent growing the graph over time

                # STEP 5: State update for next time step
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else: x = self.A @ x + self.B @ u_t + w_t
                
                # STEP 6: Compute quadratic cost
                cost = y_obs.t() @ self.Q_obs @ y_obs.t() + u_t.t() @ self.R @ u_t
                costs[t] = cost.detach().item()

                # Maintain fixed-length history
                max_history = max(self.m, self.m_tilde) + 2  # +2 for padding
                if len(u_history) > max_history: u_history.pop(0)
                if len(y_history) > max_history: y_history.pop(0)
                if len(y_nat_history) > max_history: y_nat_history.pop(0)

                if use_control:
                    # Use autograd for gradient computation
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                    scheduler.step()
                    
            # Store costs for this trial               
            all_costs[trial, :] = costs

        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses
    
    def reset(self):
        """Reset all trajectories and losses"""
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)