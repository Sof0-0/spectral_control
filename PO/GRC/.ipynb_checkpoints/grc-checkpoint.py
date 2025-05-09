import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr

class GRC(torch.nn.Module):
    def __init__(self, A, B, C,  Q, R, h=5, eta=0.001, T=100, name="GRC", nl=False):

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
        self.h = h  # horizon length 
        self.m = 10  
        self.n, self.m_control = B.shape  # state and control dimensions
        self.eta = eta  
        self.T = T
        self.W_test = self._initialize_sinusoidal_disturbances()
        
        # E has shape (m_control, n, self.m) - maps past m perturbations to control
        self.E = torch.nn.Parameter(torch.ones(self.m_control, self.n, self.m) *1e-1)
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
            
            # Calculate control using LQR + learned perturbation compensation
            u_nominal = -self.K @ y_obs  # updated with y_nat
            
            # Add the learned perturbation compensation using the most recent m perturbations
            u_pert = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
            

            ##### MAIN STEP 
            for i in range(self.m): # for the last m perturbations
                if i < min(t+1, self.m): # check if we have enough history
                    y_nat_i = self.y_nat_history[-(i+1)]  # Get past y_nat values in reverse order
                    for j in range(y_nat_i.shape[0]): # over h
                        u_pert += self.E[:, j:j+1, i] * y_nat_i[j]

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
                

                # with torch.no_grad():
                #     total_norm = torch.norm(self.E)
                #     max_norm = 10.0 
                #     if total_norm > max_norm:
                #         self.E *= max_norm / total_norm
            

            #self.e_history.append(self.E.detach().clone())

            # Save current state and control for next iteration
            x_prev = x.clone()
            u_prev = u.clone()
    
    
    def compute_proxy_loss(self):
        """
        Compute the proxy loss by simulating future states and controls
        """
        # Start with the current state for proxy simulation
        #x_sim = self.x_trajectory[-1].clone()
        #y_obs_sim = self.C @ x_sim

        # Clone the current histories for simulation
        #y_nat_history_sim = self.y_nat_history.clone()
        #u_history_sim = self.u_history.clone()

        y_obs_sim = torch.zeros_like(self.A[:, 0:1])
        proxy_cost = 0.0
        
        # Simulate dynamics for h steps ahead
        for h in range(self.h):
            # Compute y_nat for this simulation step
            #sim_time_step = t + h
            #y_nat_sim = self.compute_y_nat(y_obs_sim, min(sim_time_step, t))

            y_nat_hist_window = self.y_nat_history[-self.m:]  # Get the last m y_nat values
            
            # Update y_nat history for simulation
            #y_nat_history_sim = torch.roll(y_nat_history_sim, -1, dims=0)
            #y_nat_history_sim[-1] = y_nat_sim
            
            # Calculate control using LQR + learned perturbation compensation
            v_nominal = -self.K @ y_obs_sim
            
            # Compute perturbation compensation
            v_pert = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
            n = self.A.shape[0]
            
            # MAIN STEP - similar to run() method
            for i in range(self.m):
                if i < len(y_nat_hist_window):
                    #y_nat_i = y_nat_history_sim[-(i+1)]  # Get past y_nat values in reverse order
                    for j in range(n):
                        v_pert += self.E[:, j:j+1, i] * y_nat_hist_window[i][j]
            
            # Total control: nominal + perturbation compensation + bias
            v = v_nominal + v_pert + self.bias
            
            w_next = self.w_history[h + self.m]
            y_obs_sim = self.A @ y_obs_sim + self.B @ v + w_next
            y_obs_sim = self.C @ y_obs_sim
            
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
