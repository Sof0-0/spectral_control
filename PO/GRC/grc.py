import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu
from scipy.linalg import solve_discrete_are

from PO.utils import lqr


class GRC(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, Q_noise=None, R_noise=None, h=3, T=100, name="GRC", lr=0.001):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nl = False
        self.eta = lr
        
        # Register system matrices as buffers
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("C", torch.tensor(C, dtype=torch.float32))
        self.register_buffer("Q", torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("Q_obs", torch.tensor(Q_obs, dtype=torch.float32))
        self.register_buffer("K", torch.tensor(lqr(A, B, Q, R), dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R, dtype=torch.float32))
        
        # Store noise parameters for compatibility with LQG
        if Q_noise is not None: self.register_buffer("Q_noise", torch.tensor(Q_noise, dtype=torch.float32))
        else: self.register_buffer("Q_noise", torch.eye(A.shape[0], dtype=torch.float32))
            
        if R_noise is not None: self.register_buffer("R_noise", torch.tensor(R_noise, dtype=torch.float32))
        else: self.register_buffer("R_noise", torch.eye(C.shape[0], dtype=torch.float32) * 1e-15)
        
        self.d = A.shape[0]   # hidden state dimension
        self.m_control = B.shape[1]  # control input dimension
        self.p = C.shape[0]  # observation dimension
        
        # GRC specific parameters
        self.h = h  # History length
        self.lr = 0.1  # Learning rate (eta in the algorithm)
        
        # Initialize controller matrices M_0 to M_h
        self.M = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 0.001)
            for _ in range(h+1)
        ])
        
        # Simulation parameters
        self.T = T
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

    def run(self, initial_state=None, add_noise=False, use_control=True, num_trials=1):

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
             
                y_obs = self.C @ x    # Get observation
                y_history.append(y_obs)
                
                # Compute natural output y_nat_t = y_t - C ∑_{i=0}^t A^i B u_{t-i}
                # This is the part of the observation not affected by past controls
                y_nat_t = y_obs.clone()
                for i in range(min(t+1, len(u_history))):
                    A_power_i = torch.matrix_power(self.A, i)
                    y_nat_t -= self.C @ (A_power_i @ self.B @ u_history[-(i+1)])
                
                y_nat_history.append(y_nat_t)
                
                # Compute control input u_t = ∑_{i=0}^h M_i^t y_nat_{t-i}

                if use_control:
                    u_t = torch.zeros((self.m_control, 1), device=self.device)
                    for i in range(min(self.h+1, len(y_nat_history))):
                        u_t += self.M[i] @ y_nat_history[-(i+1)]
                        #print("UTS", u_t)
                else:
                    # If use_control is False, use zero control for compatibility with LQG
                    u_t = torch.zeros((self.m_control, 1), device=self.device)

                u_t += -self.K @ x 
                u_history.append(u_t)


                # Generate process noise if needed
                if add_noise:
                    noise_dist = torch.distributions.MultivariateNormal(
                        torch.zeros(self.d, device=self.device), 
                        self.Q_noise
                    )
                    w_t = noise_dist.sample().view(-1, 1)
                else: w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)
                
                # Update state
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else: x = self.A @ x + self.B @ u_t + w_t

                #x = x.detach()  # Detach to prevent growing the graph over time

                #print(u_t.shape, y_obs.shape, x.shape)
                #print(self.Q_obs.shape, self.R.shape)
                # Compute quadratic cost (same as LQG)
                cost = y_obs.t() @ self.Q_obs @ y_obs + u_t.t() @ self.R @ u_t
                costs[t] = cost.item()

                # Maintain fixed-length history
                if len(u_history) > self.h + t + 1: u_history.pop(0)
                if len(y_history) > self.h + t + 1: y_history.pop(0)
                if len(y_nat_history) > self.h + t + 1: y_nat_history.pop(0)

                #u_history.append(u_t)
                
            # Update controller matrices using gradient descent
            if use_control and t >= 10:
                #print("HERE")
                # Construct loss gradient
                # Use autograd for gradient computation
                optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # Add gradient clipping
                optimizer.step()
                scheduler.step()
                

            all_costs[trial, :] = costs

            x = x.detach()
            
                     
        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses
    
    def reset(self):
        """Reset all trajectories and losses"""
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

  
