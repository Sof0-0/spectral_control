import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr, get_hankel_new

class DSC(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, h, h_tilde, m, m_tilde,gamma, Q_noise=None, R_noise=None,  eta=0.001, T=500, name="DSC", nl=False):

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
        self.register_buffer("sigma", eigvals_m[-self.h:].clone().detach().to(torch.float32))  # Top-h eigenvalues
        self.register_buffer("phi", eigvecs_m[:, -self.h:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        
        # Compute Hankel matrix for rows
        Z_m_tilde = get_hankel_new(self.m_tilde, self.gamma)
        eigvals_m_tilde, eigvecs_m_tilde = torch.linalg.eigh(Z_m_tilde)
        
        # Register eigenvalues and eigenvectors for rows
        self.register_buffer("lambda_e", eigvals_m_tilde[-self.h_tilde:].clone().detach().to(torch.float32))  # Top-H eigenvalues
        self.register_buffer("phi_tilde", eigvecs_m_tilde[:, -self.h_tilde:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        #### FILTERS ######



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

        # self.M_tilde = torch.nn.Parameter(torch.ones(self.h, self.m, self.m_control, self.p, device=self.device) * 1e-2)  # (h, m, m_c, p)
        # self.M = torch.nn.Parameter(torch.ones(self.h, self.m, self.h, self.m, self.m_control, self.p, device=self.device) * 1e-2)  # (h, m, h, m, m_c, p)

         # Initialize controller matrices according to the control law formula
        # First term: M_0^t
        self.M_0 = torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 1e-15)
        
        # Second term: M_i^t for i=1,...,h_tilde
        self.M_tilde = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 1e-15)
            for _ in range(self.h_tilde)
        ])
        
        # Third term: M_0l^t for l=0,...,h
        self.M_0l = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 1e-15)
            for _ in range(self.h)
        ])
        
        # Fourth term: M_il^t for i=1,...,h_tilde, l=0,...,h
        self.M = torch.nn.ParameterList([
            torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 1e-15)
                for _ in range(self.h)
            ])
            for _ in range(self.h_tilde)
        ])


        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)


        
    def nonlinear_dynamics(self, x):
        """Apply nonlinear dynamics if nl flag is True"""
        return leaky_relu(x)


    def compute_control_vectorized(self, y_nat_history):
        """
        Compute control action based on natural output history according to the formula:
        
        u_t = M_0^t y_t^nat + 
              sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^{1/4} [phi_i]_j M_i^t y_{t-j}^nat +
              sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^{1/4} [phi_l]_k M_{0l}^t y_{t-k}^nat +
              sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l lambda_i)^{1/4} [phi_l]_k [phi_i]_j M_{il}^t y_{t-j-k}^nat
        """
        # Need at least m + m_tilde observations for the control law
        required_history = max(self.m, self.m_tilde)
        if len(y_nat_history) <= required_history: return torch.zeros(self.m_control, 1, device=self.device)

        # Get current natural output
        y_nat_t = y_nat_history[-1]
        
        # First term: M_0^t y_t^nat
        u = self.M_0 @ y_nat_t
        
        # Second term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^{1/4} [phi_i]_j M_i^t y_{t-j}^nat
        for i in range(self.h_tilde):
            for j in range(1, self.m_tilde + 1):  # j starts from 1
                if j < len(y_nat_history):
                    y_t_j = y_nat_history[-j-1]  # y_{t-j}^nat
                    lambda_i_factor = self.lambda_e[i] ** 0.25
                    phi_i_j = self.phi_tilde[j-1, i]  # [phi_i]_j
                    u += lambda_i_factor * phi_i_j * (self.M_tilde[i] @ y_t_j)
        
        # Third term: sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^{1/4} [phi_l]_k M_{0l}^t y_{t-k}^nat
        for l in range(self.h):
            for k in range(self.m):
                if k < len(y_nat_history):
                    y_t_k = y_nat_history[-k-1]  # y_{t-k}^nat
                    sigma_l_factor = self.sigma[l] ** 0.25
                    phi_l_k = self.phi[k, l]  # [phi_l]_k
                    u += sigma_l_factor * phi_l_k * (self.M_0l[l] @ y_t_k)
        
        # Fourth term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l lambda_i)^{1/4} [phi_l]_k [phi_i]_j M_{il}^t y_{t-j-k}^nat
        for i in range(self.h_tilde):
            for j in range(1, self.m_tilde + 1):  # j starts from 1
                for l in range(self.h):
                    for k in range(self.m):
                        j_k = j + k
                        if j_k < len(y_nat_history):
                            y_t_jk = y_nat_history[-j_k-1]  # y_{t-j-k}^nat
                            sigma_lambda_factor = (self.sigma[l] * self.lambda_e[i]) ** 0.25
                            phi_l_k = self.phi[k, l]  # [phi_l]_k
                            phi_i_j = self.phi_tilde[j-1, i]  # [phi_i]_j
                            u += sigma_lambda_factor * phi_l_k * phi_i_j * (self.M[i][l] @ y_t_jk)
        
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
            if initial_state is not None:  x = initial_state.to(self.device)
            else:  x = torch.randn(self.d, 1, dtype=torch.float32, device=self.device)
                

            # Initialize histories for observations and controls
            u_history = [torch.zeros((self.m_control, 1), device=self.device) for _ in range(self.h+1)]
            y_history = [torch.zeros((self.p, 1), device=self.device) for _ in range(self.h+1)]
            y_nat_history = [torch.zeros((self.p, 1), device=self.device) for _ in range(self.h+1)]

            costs = torch.zeros(self.T, dtype=torch.float32, device=self.device)

            for t in range(self.T):

                # STEP 1: Get observation from current state
                y_obs = self.C @ x  # introduce the y_t as a linear projection of x
                y_history.append(y_obs.detach())
                print("HERE")

                # Compute natural output y_nat_t = y_t - C ∑_{i=0}^t A^i B u_{t-i}
                # This is the part of the observation not affected by past controls
                y_nat_t = y_obs.clone()
                for i in range(min(t+1, len(u_history))):
                    A_power_i = torch.matrix_power(self.A, i)
                    y_nat_t -= self.C @ (A_power_i @ self.B @ u_history[-(i+1)])
                
                y_nat_history.append(y_nat_t.detach())
                #print(y_nat_t)

                # STEP 3: Calculate control using LQR + learned perturbation compensation
                if use_control:
                    u_pert = self.compute_control_vectorized(y_nat_history)  
                    u_t = u_pert 
        
                else:  u_t = torch.zeros((self.m_control, 1), device=self.device)
                u_t += -self.K @ x 
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

                x = x.detach()  # Detach to prevent growing the graph over time

                # STEP 6: Compute quadratic cost

                cost = y_obs.t() @ self.Q_obs @ y_obs + u_t.t() @ self.R @ u_t
                costs[t] = cost.detach().item()

                if use_control:
                    print("I AM OUT")
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                    scheduler.step()
           
                max_history = max(self.m, self.m_tilde) + 2  # +2 for padding
                if len(u_history) > max_history: u_history.pop(0)
                if len(y_history) > max_history: y_history.pop(0)
                if len(y_nat_history) > max_history: y_nat_history.pop(0)
                              
        all_costs[trial, :] = costs

        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses
    
    def reset(self):
        """Reset all trajectories and losses"""
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)