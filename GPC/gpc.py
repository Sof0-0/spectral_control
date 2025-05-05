import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from utils import lqr

class GradientPerturbationController(torch.nn.Module):
    def __init__(self, A, B, Q, R, h=5, eta=0.001, T=100, name="GPC", nl=False):

        """
        Gradient Perturbation Controller implementation in PyTorch
        
        Args:
            A: System dynamics matrix
            B: Control matrix
            Q: State cost matrix
            R: Control cost matrix
            h: History horizon 
            eta: Learning rate
            W_test: Test perturbation sequence
            name: Controller name
            nl: Whether to use nonlinear dynamics

        """
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
        self.E = torch.nn.Parameter(torch.ones(self.m_control, self.n, self.m))
        self.bias = torch.nn.Parameter(torch.zeros(self.m_control, 1))
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        self.register_buffer("w_history", torch.zeros(self.h + self.m, self.n, 1))

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
        
    def nonlinear_dynamics(self, x):
        return leaky_relu(x)
  
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
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x_prev)
                    x = self.A @ x_nl + self.B @ u_prev + w_t
                else:
                    x = self.A @ x_prev + self.B @ u_prev + w_t
            
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
            

            self.e_history.append(self.E.detach().clone())

            # Save current state and control for next iteration
            x_prev = x.clone()
            u_prev = u.clone()

            # if t == 30:
            #     print("X TRAJECTORY:", self.x_trajectory)
            #     print("U TRAJECTORY:", self.u_trajectory)
            #     print("E HISTORY:", self.e_history)
            # if t == 40:
            #     print("X TRAJECTORY:", self.x_trajectory)
            #     print("U TRAJECTORY:", self.u_trajectory)
            #     print("E HISTORY:", self.e_history)
            #if t == 40:
                #print("X TRAJECTORY:", self.x_trajectory)
                #print("U TRAJECTORY:", self.u_trajectory)
                #print("E HISTORY:", self.e_history)
    
    def compute_proxy_loss(self):

        """
        Compute the proxy loss by simulating future states and controls

        """
        # Start with zero initial state for proxy simulation
        y = torch.zeros_like(self.A[:, 0:1])
        proxy_cost = 0.0
        
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
            
            # Update state using the control
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
