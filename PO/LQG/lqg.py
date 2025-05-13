import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr

class LQG(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_noise, R, R_noise, h, eta=0.001, T=1000, name="LQG", nl=False):

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
        self.register_buffer("K", torch.tensor(lqr(A, B, Q, R), dtype=torch.float32)) # LQR feedback gain
        self.register_buffer("Q_noise", torch.tensor(Q_noise, dtype=torch.float32))
        self.register_buffer("R_noise", torch.tensor(R_noise, dtype=torch.float32))
        
        # Kalman filter gain L (steady-state)
        self.register_buffer("L", torch.tensor(self._compute_kalman_gain(A, C, Q_noise, R_noise), dtype=torch.float32))
        
        # Set controller parameters
        self.h = h  # number of past values to consider for disturbance compensation
        self.n, self.m_control = B.shape  # state and control dimensions
        self.p = C.shape[0]  # output dimension
        
        self.eta = eta  
        self.T = T
        #self.W_test = self._initialize_sinusoidal_disturbances()
        self.W_test = self._initialize_gaussian_disturbances()
        
        self.M = torch.nn.Parameter(torch.ones(self.m_control, self.p, self.h+1))
        self.bias = torch.nn.Parameter(torch.zeros(self.m_control, 1))
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        self.register_buffer("w_history", torch.zeros(self.h + self.h, self.n, 1))

        # Store control history
        self.max_history_len = T  # Maximum length of control history to keep
        self.register_buffer("u_history", torch.zeros(self.max_history_len, self.m_control, 1))

        # Store y_nat history for control computation
        self.register_buffer("y_nat_history", torch.zeros(self.h+1, self.p, 1))  # Store h+1 y_nat values

        # Store estimation error history for disturbance compensation
        self.register_buffer("error_history", torch.zeros(self.h+1, self.p, 1))

        # Tracking variables for states, estimates, and controls
        self.x_trajectory = []      # True states
        self.x_hat_trajectory = []  # Estimated states
        self.u_trajectory = []      # Control inputs
        self.y_trajectory = []      # Observations
        self.losses = None

    def _compute_kalman_gain(self, A, C, Q_noise, R_noise):
        """
        Compute the steady-state Kalman filter gain L
        Using DARE
        """
        # Convert to numpy for easier computation
        A_np = np.array(A)
        C_np = np.array(C)
        Q_np = np.array(Q_noise)
        R_np = np.array(R_noise)
        
        n = A_np.shape[0]
        P = np.eye(n)  # Initial covariance estimate
        
        # Iterate until convergence (simplified approach)
        for _ in range(100):
            # Kalman filter update equations for steady state
            # Prediction: P = A*P*A' + Q
            P_pred = A_np @ P @ A_np.T + Q_np
            
            # Kalman gain: L = P*C'*inv(C*P*C' + R)
            S = C_np @ P_pred @ C_np.T + R_np
            L = P_pred @ C_np.T @ np.linalg.inv(S)
            
            # Update: P = (I - L*C)*P_pred
            P = (np.eye(n) - L @ C_np) @ P_pred
            
        return L
    
    def _initialize_gaussian_disturbances(self):
        w = np.random.normal(loc=0.0, scale=1.0, size=(self.T, 1))
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def _initialize_sinusoidal_disturbances(self):
        """Initialize sinusoidal disturbances for testing"""
      
        magnitude = 1.0
        freq = 3.0

        t_range = torch.arange(self.T)
        sin_values = torch.sin(t_range * 2 * torch.pi * freq / self.T)
    
        w_matrix = magnitude * sin_values.repeat(self.n, 1).T
        
        # Convert to list of tensors
        W_test = [
            w_matrix[i].reshape(-1, 1)
            for i in range(w_matrix.shape[0])
        ]

        return W_test
        
    def nonlinear_dynamics(self, x):
        return leaky_relu(x)

    def compute_control_vectorized(self, error):
        """
        Compute control perturbation based on estimation errors
        Instead of using y_nat, we use the estimation error (y - C*x_hat)
        """
        # Initialize control perturbation
        u_pert = torch.zeros(self.m_control, 1, device=self.device)
        
        # For each time step in the history
        for i in range(self.h + 1):
            # If we have enough history
            if i < len(self.error_history):
                # Get the appropriate error value
                error_i = self.error_history[-(i+1)]  # Shape: [p, 1]
                
                # Apply the corresponding M matrix
                # M shape: [m_control, p, h+1]
                # For each control dimension and each output dimension
                for j in range(self.p):
                    u_pert += self.M[:, j:j+1, i] * error_i[j]
        
        return u_pert

    def run(self):
     
        self.to(self.device)
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)
    
        # Reset tracking variables
        self.x_trajectory = []
        self.x_hat_trajectory = []
        self.u_trajectory = []
        self.y_trajectory = []
        
        # Weight decay for stability
        optimizer = torch.optim.Adam(self.parameters(), lr=self.eta, weight_decay=1e-5)
        
        # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        # Initialize true state and estimated state
        x = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        x_hat = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
        x_hat_prev = torch.zeros_like(x_hat)
        y_obs = torch.zeros(self.p, 1, dtype=torch.float32, device=self.device)
        
        x_prev = torch.zeros_like(x)
        u_prev = torch.zeros(self.m_control, 1, dtype=torch.float32, device=self.device)
    
        for t in range(self.T):
            if self.w_test is not None and t < len(self.w_test): 
                w_t = self.w_test[t]  # Perturbation for this time step
            else:
                if t > 0:  w_t = x - self.A @ x_prev - self.B @ u_prev
                else: w_t = torch.zeros(self.n, 1, dtype=torch.float32, device=self.device)
            
            # True state update
            if t > 0: 
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x_prev)
                    x = self.A @ x_nl + self.B @ u_prev + w_t
                else: x = self.A @ x_prev + self.B @ u_prev + w_t
            
            y_obs = self.C @ x  # True observation
            
            self.w_history = torch.roll(self.w_history, -1, dims=0)
            self.w_history[-1] = w_t
            
            # Kalman filter update 
            # x̂_{t+1} = (A - LC)x̂_t + Bu_t + Ly_t
            if t > 0: x_hat = (self.A - self.L @ self.C) @ x_hat_prev + self.B @ u_prev + self.L @ y_obs
            
            # Calculate control using LQR based on estimated state
            u_lqr = -self.K @ x_hat
            
    
            # Update control history
            self.u_history = torch.roll(self.u_history, -1, dims=0)
            self.u_history[-1] = u_lqr
    
            # Compute quadratic cost
            cost = x.T @ self.Q @ x + u_lqr.T @ self.R @ u_lqr
            self.losses[t] = cost.item()

    
            x_prev = x.clone()
            x_hat_prev = x_hat.clone()
            u_prev = u_lqr.clone()



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

def plot_loss_sliding(controller, title, window_size=50):
    """
    Plot the moving average of the controller's loss over time.

    Parameters:
    - controller: controller object with .losses attribute
    - title: title for the plot
    - window_size: window size for computing moving average
    """
    losses = controller.losses.cpu().numpy()
    T = len(losses)

    if T < window_size:
        raise ValueError("Window size should be smaller than the length of the loss sequence.")

    # Compute moving average using convolution
    moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(window_size - 1, T), moving_avg)
    plt.xlabel('Time Step')
    plt.ylabel(f'{window_size}-Step Average Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


