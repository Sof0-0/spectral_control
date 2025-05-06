import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from FullO.utils import lqr, get_hankel_new

class OSC(torch.nn.Module):
    def __init__(self, A, B, Q, R, h=5, m=20, gamma=0.1, eta=0.001, T=100, name="OSC", nl=False):

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
        self.h = h  # spectral decomposition horizon
        self.m = m  # Hankel matrix size
        self.gamma = gamma  # stability parameter
        self.n, self.m_control = B.shape  # state and control dimensions
        self.eta = eta  
        self.T = T
        self.W_test = self._initialize_sinusoidal_disturbances()
        
        # Initialize Hankel matrix and compute spectral decomposition
        Z_m = get_hankel_new(self.m, self.gamma)
        eigvals, eigvecs = torch.linalg.eigh(Z_m)
        
        # Register top-h eigenvalues and eigenvectors
        self.register_buffer("sigma", eigvals[-h:].clone().detach().to(torch.float32))  # Top-h eigenvalues
        self.register_buffer("phi", eigvecs[:, -h:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        
        # E has shape (m_control, n, h) - maps spectral components to control
        #self.E = torch.nn.Parameter(torch.zeros(self.m_control, self.n, self.h))
        self.E = torch.nn.Parameter(torch.ones(self.m_control, self.n, self.h))
        self.bias = torch.nn.Parameter(torch.zeros(self.m_control, 1))
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        self.register_buffer("w_history", torch.zeros(self.m, self.n, 1))

        # Tracking variables for states and controls
        self.x_trajectory = []
        self.u_trajectory = []
        self.e_history = []
        self.loss_history = []

        
    def _initialize_sinusoidal_disturbances(self):
        """
        Initialize sinusoidal disturbances for testing
        """
        # Keep frequency = 3.0 and magnitude = 1.0 (noise example)
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
        
        # Weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)
        
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        # Initialize state and control
        x = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        
        x_prev = torch.zeros_like(x)
        u_prev = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)


        # Save initial E values
        self.e_history.append(self.E.detach().clone())
        #print("Initial E matrix norm:", torch.norm(self.E).item())
        
        for t in range(self.T):

            # Save current state
            self.x_trajectory.append(x.detach().clone())


            if self.w_test is not None and t < len(self.w_test): 
                w_t = self.w_test[t]  # Perturbation for this time step
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
            
            # Add the learned perturbation compensation using spectral decomposition
            recent_w = self.w_history[-self.m:]
            
            


            ##### MAIN STEP 
            # Compute control perturbation using spectral components

            u_pert = torch.zeros(self.m_control, 1, device=self.device)  # Initialize with proper dimensions

            for i in range(self.h):  # For each spectral component
                # First multiply each w by filters phi
                weighted_w = torch.zeros(self.n, 1, device=self.device)
                
                for k in range(min(self.m, len(recent_w))):
                    if k < len(recent_w):
                        # Apply phi filter to each w (phi[k,i] is the k-th element of the i-th eigenvector)
                        weighted_w += recent_w[k] * self.phi[k, i]
                
                # Then multiply the result by E (which maps state to control for each spectral component)
                # E[:,:,i] has shape (m_control, n)
                transformed = torch.matmul(self.E[:, :, i], weighted_w)
                
                # Finally multiply by the spectral factor sigma^(1/4)
                spectral_factor = torch.pow(self.sigma[i], 0.25)
                u_pert += spectral_factor * transformed

            ##### MAIN STEP END
            
            # Total control: nominal + perturbation compensation + bias
            u = u_nominal + u_pert + self.bias
            #u = u_pert + self.bias

            # Save control for analysis
            self.u_trajectory.append(u.detach().clone())
            
            
            # Compute quadratic cost
            cost = x.T @ self.Q @ x + u.T @ self.R @ u
            self.losses[t] = cost.item()
            
            # Skip gradient updates until we have enough history (warm-up phase)
            if t >= min(10, self.h + self.m):
                optimizer.zero_grad()
                
                # Compute proxy loss (forward simulation of cost over horizon)
                proxy_loss = self.compute_proxy_loss()
                self.loss_history.append(proxy_loss.item())

                # Debug: print proxy loss occasionally
                #if t % 10 == 0:
                    #print(f"Step {t}, Proxy Loss: {proxy_loss.item()}, E norm: {torch.norm(self.E).item()}")


                proxy_loss.backward()

                # Debug: print E gradients occasionally
                #if t % 10 == 0 and self.E.grad is not None:
                    #print(f"E grad norm: {torch.norm(self.E.grad).item()}")
                
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

            #if t % 20 == 0:
                #print(f"Step {t}/{self.T}, Cost: {cost.item()}, E norm: {torch.norm(self.E).item()}")

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
        y = torch.zeros(self.n, 1, device=self.device)
        proxy_cost = 0.0
        
        # Simulate dynamics for h steps ahead
        for h_step in range(self.h):
            # Compute control: u = K @ y + spectral perturbation compensation + bias
            v_nominal = self.K @ y
            
            # Compute spectral perturbation compensation
            v_pert = torch.zeros_like(self.bias)
            
            # Get the perturbation window for this horizon step
            w_window = self.w_history[h_step:h_step+self.m]
            #print("W window:", w_window)
            
            # Apply spectral control for each component
            for i in range(self.h):
                # First multiply each w by filters phi
                weighted_w = torch.zeros(self.n, 1, device=self.device)
                
                for k in range(min(self.m, len(w_window))):
                    if k < len(w_window):
                        # Apply phi filter to each w (phi[k,i] is the k-th element of the i-th eigenvector)
                        weighted_w += w_window[k] * self.phi[k, i]

                # Check if weighted_w is all zeros
                # if not torch.all(weighted_w == 0):
                #     print(f"weighted_w at step {h_step}, component {i} is not all zeros")
                #     print("weighted_w values:", weighted_w)
                
                # Then multiply the result by E
                transformed = torch.matmul(self.E[:, :, i], weighted_w)  # Shape: (m_control, 1)
        
                #if not torch.all(transformed == 0):
                    #print(f"transformed at step {h_step}, component {i} is not all zeros")
                    #print("transformed values:", transformed)


                # Finally multiply by the spectral factor
                spectral_factor = torch.pow(self.sigma[i], 0.25)
                v_pert += spectral_factor * transformed
                #print("V pert:", v_pert)
            
            v = -v_nominal + v_pert + self.bias
            #v = v_pert + self.bias
            
            # Update state using the control
            w_next = self.w_history[h_step + self.m] if h_step + self.m < self.w_history.size(0) else torch.zeros_like(y)
            #print("W next:", w_next)
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