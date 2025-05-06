import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from FullO.utils import lqr, get_hankel

class GPCwSTU(torch.nn.Module):
    def __init__(self, A, B, Q, R, h=5, eta=0.001, T=100, name="GPC", nl=False):
        super().__init__()
        self.name = name
        self.nl = nl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Register system matrices as buffers
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("Q", torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("R", torch.tensor(R, dtype=torch.float32))
        self.register_buffer("K", torch.tensor(lqr(A, B, Q, R), dtype=torch.float32))
        
        # Set controller parameters
        self.h = h  # horizon length 
        self.m = 5  
        self.n, self.m_control = B.shape  # state and control dimensions
        self.eta = eta  
        self.T = T
        self.W_test = self._initialize_sinusoidal_disturbances()
        
        # E has shape (m_control, n, self.m) - maps past m perturbations to control
        self.E = torch.nn.Parameter(torch.ones(self.m_control, self.n, self.m) * 1e1)
        self.bias = torch.nn.Parameter(torch.zeros(self.m_control, 1))

        ### STU-PART ###
        # New E_stu parameter with shape (20, m_control, n)
        self.E_stu = torch.nn.Parameter(torch.ones(20, self.m_control, self.n) * 1e-1)

         # STU filters:
        Z_T = get_hankel(self.T) 
        eigvals, eigvecs = torch.linalg.eigh(Z_T)  # Compute eigenpairs
        self.register_buffer("sigma_stu", eigvals[-20:].clone().detach().to(torch.float32))  # Top-k eigenvalues
        self.register_buffer("phi_stu", eigvecs[:, -20:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        ### STU-PART ###
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        self.register_buffer("w_history", torch.zeros(self.h + self.m, self.n, 1))

        # New buffer to store control history
        self.register_buffer("u_history", torch.zeros(self.T, self.m_control, 1))  # was m by T

        # Tracking variables for states and controls
        self.x_trajectory = []
        self.u_trajectory = []
        self.e_history = []

        

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
        
  
    def run(self):
        
        self.to(self.device)
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

        # Reset tracking variables
        self.x_trajectory = []
        self.u_trajectory = []
        self.e_history = []

        # Reset control history
        self.u_history = torch.zeros(self.T, self.m_control, 1, dtype=torch.float32, device=self.device)
        
        # weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)
        
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
      
        x = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        x_prev = torch.zeros_like(x)
        u_prev = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
        

        # Save initial E values
        self.e_history.append(self.E.detach().clone())

        for t in range(self.T):

            # Save current state
            self.x_trajectory.append(x.detach().clone())

            if self.w_test is not None and t < len(self.w_test): 
                w_t = self.w_test[t] # perturbation for this time step
            else:
                # If no test perturbation provided, infer it from state dynamics
                if t > 0: w_t = x - self.A @ x_prev - self.B @ u_prev
                else: w_t = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
            
            # State update:
            if t > 0:
                # Get available control history up to time t
                u_t_history = self.u_history[:t]

                ### STU-PART ###
                if t > 0: # adjust if warn-up phase
                    # Apply STU filters to input signal to get STU signal
                    u_t_history_reshaped = u_t_history.view(t, -1).t()

                    # Initialize next state
                    x = torch.zeros_like(x_prev)

                    # Apply STU transformation
                    for i in range(20):
                        
                        # Get the projection of control history onto the i-th eigenvector
                        u_t_hist_stu = torch.matmul(u_t_history_reshaped, self.phi_stu[:t, i:i+1])

                        # Apply E_stu to transform the projection
                        for j in range(self.n):
                            x[j] += torch.sum(self.E_stu[i, :, j:j+1] * u_t_hist_stu)

                else:
                    x = self.A @ x_prev + self.B @ u_prev + w_t
                ### STU-PART ###
                            
                            
            # Update perturbation history (roll and add new perturbation)
            self.w_history = torch.roll(self.w_history, -1, dims=0)
            self.w_history[-1] = w_t
            
            # Calculate control using LQR + learned perturbation compensation
            u_nominal = -self.K @ x  # LQR component
            
            # Add the learned perturbation compensation using the most recent m perturbations
            recent_w = self.w_history[-self.m:]
            u_pert = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
            

            ##### MAIN STEP 
            for i in range(self.m): # for the last m perturbations
                if i < len(recent_w): # check if we have enough history
                    for j in range(self.n): # for each state
                        u_pert += self.E[:, j:j+1, i] * recent_w[i][j]

            ##### MAIN STEP END
            
            # Total control: nominal + perturbation compensation + bias
            u = u_nominal + u_pert + self.bias

            # Store the control in history
            if t < self.T: self.u_history[t] = u.detach().clone()
            
            self.u_trajectory.append(u.detach().clone())

            # Compute quadratic cost
            cost = x.T @ self.Q @ x + u.T @ self.R @ u
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
                    max_norm = 5.0 
                    if total_norm > max_norm:
                        self.E *= max_norm / total_norm

                    # Normalize E_stu
                    total_norm_stu = torch.norm(self.E_stu)
                    if total_norm_stu > max_norm:
                        self.E_stu *= max_norm / total_norm_stu
            

            self.e_history.append(self.E.detach().clone())

            # Save current state and control for next iteration
            x_prev = x.clone()
            u_prev = u.clone()

      
    def compute_proxy_loss(self):

        """
        Compute the proxy loss by simulating future states and controls

        """
        # Start with zero initial state for proxy simulation
        y = torch.zeros_like(self.A[:, 0:1])
        proxy_cost = 0.0

        # Create temporary control history for proxy simulation
        temp_u_history = self.u_history.clone()
        
        # Simulate dynamics for h steps ahead
        for h in range(self.h):
            # Get the perturbation window for this horizon step
            w_window = self.w_history[h:h+self.m]
            
            # Compute control: u = K @ y + E @ w_window + bias
            v_nominal = self.K @ y
            
            # Compute perturbation compensation similar to how we did in run()
            v_pert = torch.zeros_like(self.bias)
            n = self.A.shape[0]
            
            for i in range(self.m):
                if i < len(w_window):
                    for j in range(n):
                        v_pert += self.E[:, j:j+1, i] * w_window[i][j]
            
            v = -v_nominal + v_pert + self.bias

            # Store this control in temporary history
            last_idx = min(h + self.T - self.h, self.T - 1)
            temp_u_history[last_idx] = v

            # Get the relevant part of the temporary history
            relevant_history = temp_u_history[:last_idx+1]
            
            # Update state using STU approach if we have enough history
            if last_idx + 1 > 0:
                # Reshape control history for matrix operations
                u_history_reshaped = relevant_history.view(last_idx+1, -1).t()
                
                # Initialize next state
                y_next = torch.zeros_like(y)
                
                # Apply STU transformation
                for i in range(20):
                    input_stu = torch.matmul(u_history_reshaped, self.phi_stu[:last_idx+1, i:i+1])
                    for j in range(n):
                        y_next[j] += torch.sum(self.E_stu[i, :, j:j+1] * input_stu)
                
                y = y_next
                
            else:
                # Fall back to standard dynamics
                w_next = self.w_history[h + self.m]
                y = self.A @ y + self.B @ v + w_next
            
            # Compute cost at this horizon step
            step_cost = y.T @ self.Q @ y + v.T @ self.R @ v
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
