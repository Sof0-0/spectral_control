import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu
from scipy.linalg import solve_discrete_are

from PO.utils import lqr


class LQG(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, Q_noise, R_noise, T=100, name="LQG", nl=False):
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

        # Compute LQR gain K
        P = solve_discrete_are(np.array(A.cpu()), np.array(B.cpu()), np.array(Q.cpu()), np.array(R.cpu()))
        P = torch.tensor(P, device=self.device, dtype=torch.float32)
        K = torch.inverse(self.B.t() @ P @ self.B + self.R) @ (self.B.t() @ P @ self.A)
        self.register_buffer("K", K)

        # Compute Kalman gain L
        kf_P = solve_discrete_are(np.array(A.cpu()).T, np.array(C.cpu()).T, np.array(Q_noise.cpu()) + 1e-1 * np.eye(self.d),np.array(R_noise.cpu()))
        #kf_P = solve_discrete_are(np.array(A.cpu()).T, np.array(C.cpu()).T, np.array(Q_noise.cpu()) + np.eye(self.n),np.array(R_noise.cpu()))

        kf_P = torch.tensor(kf_P, device=self.device, dtype=torch.float32)
        L = kf_P @ self.C.t() @ torch.inverse(self.C @ kf_P @ self.C.t() + self.R_noise)
        self.register_buffer("L", L)

        self.T = T
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)

    def nonlinear_dynamics(self, x):
        """Apply nonlinear transformation to state if nonlinear mode is enabled"""
        return leaky_relu(x)

    def run(self, initial_state=None, add_noise=False, use_control=True, num_trials=1):

        self.to(self.device)
        
        # Store costs for multiple trials
        all_costs = torch.zeros((num_trials, self.T), dtype=torch.float32, device=self.device)
        
        for trial in range(num_trials):

            
            # Initialize true state and estimated state
            if initial_state is not None: x = initial_state.to(self.device)
            else: x = torch.randn(self.d, 1, dtype=torch.float32, device=self.device)
            
            x_hat = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)
            costs = torch.zeros(self.T, dtype=torch.float32, device=self.device)
            
            for t in range(self.T):
               
                y_obs = self.C @ x  # Get observation
                
                # Calculate control using LQR based on estimated state
                u_t = -self.K @ x_hat if use_control else torch.zeros((self.m_control, 1), device=self.device)

                # Generate process noise if needed
                if add_noise:
                    noise_dist = torch.distributions.MultivariateNormal(
                        torch.zeros(self.d, device=self.device), 
                        self.Q_noise
                    ) #multivariate normal (Gaussian) distribution
                    w_t = noise_dist.sample().view(-1, 1) 
                else:
                    w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)
                
                # Update true state
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else:
                    x = self.A @ x + self.B @ u_t + w_t
                
                # Update state estimate using Kalman filter
                # x̂_{t+1} = A x̂_t + B u_t + L (y_t - C x̂_t)
                x_hat = self.A @ x_hat + self.B @ u_t + self.L @ (y_obs - self.C @ x_hat)
                # Compute quadratic cost
                cost = y_obs.t() @ self.Q_obs @ y_obs + u_t.t() @ self.R @ u_t
                costs[t] = cost.item()
            
            all_costs[trial, :] = costs
            
        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses
    
    def reset(self):
        """Reset all trajectories and losses"""
        self.x_trajectory = []
        self.x_hat_trajectory = []
        self.u_trajectory = []
        self.y_trajectory = []
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)


def plot_loss(controller, title):
    """Plot the controller's loss over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(controller.losses.cpu().numpy())
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



