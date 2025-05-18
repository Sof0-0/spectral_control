import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.functional import relu, leaky_relu

from PO.utils import lqr, get_hankel_new

class DSC(torch.nn.Module):
    def __init__(self, A, B, C, Q, Q_obs, R, h, h_tilde, m, m_tilde,gamma, Q_noise=None, R_noise=None,  eta=0.0001, T=100, name="DSC", nl=False):

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
        self.n = B.shape[1]  # control input dimension
        self.p = C.shape[0]          # observation dimension
        
        
        # Set controller parameters
        # CHECK THESE ARE MATRICES DIMS
        self.h = h      # filter dimension parameter (h1 in the formula)
        self.h_tilde = h     # filter dimension parameter (h2 in the formula), using same value as h
        self.m = m      # number of row eigenvectors to use
        self.m_tilde = m     # number of past y_nat to consider (m1 in the formula)
       
     
        self.eta = 0.9  # was 0.001
        self.gamma = gamma

        #### NOISE PARAMS ####
        self.noise_mode = "sinusoid"  # Options: "gaussian", "sinusoid"
        self.sin_freq = 0.1  # Frequency of the sinusoid
        self.sin_amplitude = 0.5  # Amplitude of the sinusoid
        self.sin_phase = torch.rand(self.d, device=self.device) * 2 * np.pi  # Random phase per dimension
         #### NOISE PARAMS ####


        # Initialize M matrices (controller parameters to be learned)
        # self.M = torch.nn.ParameterList([
        #     torch.nn.Parameter(torch.ones(self.m_control, self.p, device=self.device) * 0.02)
        #     for _ in range(h + 1)  # M_0 to M_h
        # ])
        
        #ANAND VERSION
        self.M = torch.nn.Parameter(torch.ones(self.h_tilde+1, self.h+1, self.n, self.p, device=self.device) * 0.1)

        #Initialize M_bar (additional controller parameter for DSC)
        self.M_bar = torch.nn.Parameter(torch.ones(self.h+1, self.n, self.p, device=self.device) * 0.1)
        
        
        #Set up optimizer for M matrices
        self.optimizer = torch.optim.SGD([self.M] + [self.M_bar], lr=self.eta) # ANAND VERSION


        ## FILTERS ######
        #VERSION WITH ANAND
        #Initialize Hankel matrix and compute spectral decomposition
        #Compute Hankel matrix for columns
        Z_m = get_hankel_new(self.m+1, self.gamma)
        eigvals_m, eigvecs_m = torch.linalg.eigh(Z_m)
        
        # Register eigenvalues and eigenvectors for columns
        self.register_buffer("sigma", eigvals_m[-(self.h+1):].clone().detach().to(torch.float32))  # Top-h eigenvalues
        self.register_buffer("phi", eigvecs_m[:,-(self.h+1):].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        
        # Compute Hankel matrix for rows
        Z_m_tilde = get_hankel_new(self.m_tilde, self.gamma)
        eigvals_m_tilde, eigvecs_m_tilde = torch.linalg.eigh(Z_m_tilde)
        
        # Register eigenvalues and eigenvectors for rows
        self.register_buffer("lambda_e", eigvals_m_tilde[-self.h_tilde:].clone().detach().to(torch.float32))  # Top-H eigenvalues
        self.register_buffer("phi_tilde", eigvecs_m_tilde[:, -self.h_tilde:].clone().detach().to(torch.float32))  # Corresponding eigenvectors



    


         # MINE
        # Initialize M matrices (controller parameters to be learned)
        # self.M = torch.nn.ParameterList([
        #     torch.nn.Parameter(torch.ones(self.n, self.p, device=self.device) * 0.01)
        #     for _ in range(h + 1)  # M_0 to M_h
        # ])
        
        # # Initialize M_bar (additional controller parameter for DSC)
        # self.M_bar = torch.nn.Parameter(torch.ones(self.n, self.p, device=self.device) * 0.01)

        # #MINE
        # Z_m = get_hankel_new(self.m, self.gamma)
        # eigvals_m, eigvecs_m = torch.linalg.eigh(Z_m)
        
        # # Register eigenvalues and eigenvectors for columns
        # self.register_buffer("sigma", eigvals_m[-self.h:].clone().detach().to(torch.float32))  # Top-h eigenvalues
        # self.register_buffer("phi", eigvecs_m[:,-self.h:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        
        # # Compute Hankel matrix for rows
        # Z_m_tilde = get_hankel_new(self.m_tilde, self.gamma)
        # eigvals_m_tilde, eigvecs_m_tilde = torch.linalg.eigh(Z_m_tilde)
        
        # # Register eigenvalues and eigenvectors for rows
        # self.register_buffer("lambda_e", eigvals_m_tilde[-self.h_tilde:].clone().detach().to(torch.float32))  # Top-H eigenvalues
        # self.register_buffer("phi_tilde", eigvecs_m_tilde[:, -self.h_tilde:].clone().detach().to(torch.float32))  # Corresponding eigenvectors
        
        # self.optimizer = torch.optim.SGD(list(self.M.parameters()) + [self.M_bar], lr=self.eta)
        # # Learning rate schedule: start low, increase, then decrease
        def lr_lambda(epoch):
            if epoch < 10: return 0.1  # Start with lower learning rate
            elif epoch < 50: return 1.0  # Full learning rate for main training period
            else: return max(0.1, 1.0 - (epoch - 50) / 100)  # Gradual decrease
        
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        #### FILTERS ######



        self.T = T
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)


        
    def nonlinear_dynamics(self, x):
        """Apply nonlinear dynamics if nl flag is True"""
        return leaky_relu(x)

    def compute_loss(self, y_obs, u_t):
        """
        Standard quadratic cost function: y^T Q_obs y + u^T R u
        """
        return y_obs.t() @ self.Q_obs @ y_obs + u_t.t() @ self.R @ u_t

    def compute_proxy_loss(self, y_nat, u_t):
 
        # Proxy loss using natural observation (y^nat) instead of actual observation (y)
        return y_nat.t() @ self.Q_obs @ y_nat + u_t.t() @ self.R @ u_t

    def update_M_matrices_new(self, y_nat_history, u_history):
        """
        Update M matrices using gradient descent:
        M^{t+1}_{0:h} ← Π_K [M^t_{0:h} - η∇ℓ_t(M^t_{0:h})]
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Create tensor versions of y_nat_history for gradient computation
        #y_nat_tensors = [y.clone().detach().requires_grad_(True) for y in y_nat_history]
        
        # Compute the control using current M matrices
        #u_t = self.compute_control_new(y_nat_tensors)
        y_nat_current = y_nat_history[-1].detach().clone()
        
        # Compute the proxy loss using natural observation (y^nat)
        # This is the key difference from the standard LQR approach
        proxy_loss = self.compute_rollout_loss(y_nat_current, u_history, y_nat_history)
        
        # Backpropagate
        proxy_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()

        # with torch.no_grad():
        #     total_norm = torch.norm(self.M)
        #     total_norm2 = torch.norm(self.M_bar)
        #     max_norm = 1000.0 
        #     if total_norm > max_norm:
        #         self.M *= max_norm / total_norm
        #     if total_norm2 > max_norm:
        #         self.M_bar *= max_norm / total_norm
        
        # Return loss value for tracking
        return proxy_loss.item()

    def compute_natural_observation(self, y_history, u_history):
        """
        Compute y^nat_t = y_t - C ∑_{i=0}^{t} A^i B u_{t-i}
        This removes the effects of control from the observation
        """
        t = len(y_history) - 1
        
        # Start with the current observation
        y_nat = y_history[-1].clone()
        
        # Subtract the effect of previous controls
        for i in range(min(t+1, len(u_history))):
            if i >= len(u_history):
                break
                
            # Compute A^i
            A_i = torch.eye(self.d, device=self.device)
            for _ in range(i):
                A_i = self.A @ A_i
                
            # Subtract C * A^i * B * u_{t-i}
            y_nat = y_nat - self.C @ A_i @ self.B @ u_history[-(i+1)]
            
        return y_nat

    def compute_predicted_state(self, y_nat_current, u_history):
        """
        Compute predicted state by adding the effect of past controls to the natural observation
        Similar to: final_state = y_nat[0] + jnp.tensordot(G, us, axes=([0, 2], [0, 1]))
        """
        # Start with current natural observation
        predicted_state = y_nat_current.clone()
        
        # Construct the G matrix (similar to DRC's G)
        # G represents the effect of past controls on the current state
        G = []
        A_power = torch.eye(self.d, device=self.device)
        for i in range(min(len(u_history), self.h)):
            G.append(self.C @ A_power @ self.B)
            A_power = A_power @ self.A
        
        # Add effect of past controls
        for i in range(min(len(u_history), self.h)):
            if i >= len(u_history): break
            
            predicted_state = predicted_state + G[i] @ u_history[-(i+1)]
        
        return predicted_state

    def compute_action_from_state_new(self, state, y_nat_history):

        return self.compute_control(y_nat_history)

    def compute_control_mine(self, y_nat_history):
        #Ensure we have enough history
        if len(y_nat_history) < max(self.m, self.m_tilde) + 1:
            # If not enough history, fall back to basic control
            u_t = torch.zeros(self.n, 1, device=self.device)
            if len(y_nat_history) > 0:
                u_t = self.M_bar @ y_nat_history[-1]
            return u_t
            
        # First term: M_bar y_t^nat
        u_t = self.M_bar @ y_nat_history[-1]
        
        # Second term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^(1/4) [phi_i]_j M_i y_{t-j}^nat
        for i in range(self.h_tilde):
            lambda_i_pow = self.lambda_e[i] ** 0.25
            for j in range(min(self.m_tilde, len(y_nat_history)-1)):
                if j+1 < len(y_nat_history):
                    phi_i_j = self.phi_tilde[j, i]
                    u_t = u_t + lambda_i_pow * phi_i_j * (self.M[min(i+1, len(self.M)-1)] @ y_nat_history[-(j+1)])
        
        # Third term: sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^(1/4) [phi_l]_k M_bar y_{t-k}^nat
        for l in range(self.h):
            sigma_l_pow = self.sigma[l] ** 0.25
            for k in range(min(self.m, len(y_nat_history))):
                if k < len(y_nat_history):
                    phi_l_k = self.phi[k, l]
                    u_t = u_t + sigma_l_pow * phi_l_k * (self.M_bar @ y_nat_history[-(k+1)])
        
        # Fourth term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l * lambda_i)^(1/4) [phi_l]_k [phi_i]_j M_i y_{t-j-k}^nat
        for i in range(self.h_tilde):
            for j in range(min(self.m_tilde, len(y_nat_history))):
                for l in range(self.h):
                    for k in range(min(self.m, len(y_nat_history))):
                        if j+k+1 < len(y_nat_history):
                            combined_pow = (self.sigma[l] * self.lambda_e[i]) ** 0.25
                            phi_l_k = self.phi[k, l]
                            phi_i_j = self.phi_tilde[j, i]
                            u_t = u_t + combined_pow * phi_l_k * phi_i_j * (self.M[min(i+1, len(self.M)-1)] @ y_nat_history[-(j+k+1)])
            
        return u_t

    def compute_action_prediction_mine(self, state, y_nat_history):
     
        return self.compute_control_mine(y_nat_history)
        


    def compute_rollout_loss(self, y_nat, u_history, y_nat_history):
        """
        Compute the proxy loss for GRC matching the JAX implementation
        """
        # Predict the state by adding the effect of past controls
        predicted_state = self.compute_predicted_state(y_nat, u_history)
        
        
        # Compute the action for this predicted state
        predicted_action = self.compute_action_from_state_new(predicted_state, y_nat_history)
        # HERE 1
        #predicted_action = self.compute_action_prediction_mine(predicted_state, y_nat_history)
        #print(predicted_action.shape)
        # Compute the loss using the predicted state and action
        return predicted_state.t() @ self.Q_obs @ predicted_state + predicted_action.t() @ self.R @ predicted_action


    def compute_control_new(self, y_nat_history):
        """
        Compute control action based on natural output history according to the formula:
        
        u_t = M_0^t y_t^nat + 
              sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^{1/4} [phi_i]_j M_i^t y_{t-j}^nat +
              sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^{1/4} [phi_l]_k M_{0l}^t y_{t-k}^nat +
              sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l lambda_i)^{1/4} [phi_l]_k [phi_i]_j M_{il}^t y_{t-j-k}^nat
        """

        # VERSION WITH ANAND
        #print("HERE")
        #Ensure we have enough history
        if len(y_nat_history) < (self.m + self.m_tilde + 1):
            # If not enough history, fall back to basic control
            u_t = torch.zeros(self.n, 1, device=self.device)
            if len(y_nat_history) > 0:
                u_t = self.M_bar[0,:,:] @ y_nat_history[-1]
            return u_t
            
        # First term: M_bar y_t^nat
        u_t = self.M_bar[0,:,:] @ y_nat_history[-1]
        #print("HERE2")
        
        # Second term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^(1/4) [phi_i]_j M_i y_{t-j}^nat
        for i in range(self.h_tilde):
            lambda_i_pow = self.lambda_e[i] ** 0.25
            for j in range(self.m_tilde):
                phi_i_j = self.phi_tilde[j, i]
                u_t = u_t + lambda_i_pow * phi_i_j * (self.M_bar[i+1, :, :] @ y_nat_history[-(j+2)])
        

        # Third term: sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^(1/4) [phi_l]_k M_bar y_{t-k}^nat
        for l in range(self.h+1):
            sigma_l_pow = self.sigma[l] ** 0.25
            for k in range(self.m+1):
                phi_l_k = self.phi[k, l]
                u_t = u_t + sigma_l_pow * phi_l_k * (self.M[0, l, :, :] @ y_nat_history[-(k+1)])
        
        # Fourth term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l * lambda_i)^(1/4) [phi_l]_k [phi_i]_j M_i y_{t-j-k}^nat
        for i in range(self.h_tilde):
            for j in range(self.m_tilde):
                for l in range(self.h+1):
                    for k in range(self.m+1):
                        combined_pow = (self.sigma[l] * self.lambda_e[i]) ** 0.25
                        phi_l_k = self.phi[k, l]
                        phi_i_j = self.phi_tilde[j, i]
                        u_t = u_t + combined_pow * phi_l_k * phi_i_j * (self.M[i+1, l, :, :] @ y_nat_history[-(j+k+2)])
            
        return u_t
    

    def run(self, initial_state=None, add_noise=False, use_control=True, num_trials=1):
        """Run the controller for T time steps"""
        self.to(self.device)

        # Store costs for multiple trials
        all_costs = torch.zeros((num_trials, self.T), dtype=torch.float32, device=self.device)

        
        for trial in range(num_trials):
            # Initialize state
            if initial_state is not None:  x = initial_state.to(self.device)
            else: x = torch.randn(self.d, 1, dtype=torch.float32, device=self.device)
                

            # Initialize histories for observations and controls
            y_history = []
            y_nat_history = []
            u_history = []

            costs = torch.zeros(self.T, dtype=torch.float32, device=self.device)

            # Initial observation
            y_obs = self.C @ x
            y_history.append(y_obs)
            y_nat_history.append(y_obs)  # Initially, y_nat = y since no control has been applied

            for t in range(self.T):

                # Compute control if enabled
                if use_control and t > 0:  # Skip first step since we need history
                    # HERE 2
                    u_t = self.compute_control_new(y_nat_history)
                    #print("UT", u_t)
                    #print(y_nat_history[-1])
                else: u_t = torch.zeros((self.n, 1), device=self.device)

                # Add control to history
                u_history.append(u_t.detach())
                print(u_history)

                # Generate process noise if needed
                if add_noise:
                    if self.noise_mode == "gaussian":
                        noise_dist = torch.distributions.MultivariateNormal(
                            torch.zeros(self.d, device=self.device), 
                            self.Q_noise
                        ) #multivariate normal (Gaussian) distribution
                        w_t = noise_dist.sample().view(-1, 1) 
                    elif self.noise_mode == "sinusoid":
                        t_tensor = torch.tensor([t], dtype=torch.float32, device=self.device)  # current timestep
                        sinusoid = self.sin_amplitude * torch.sin(2 * np.pi * self.sin_freq * t_tensor + self.sin_phase)
                        w_t = sinusoid.view(-1, 1)

                else:
                    w_t = torch.zeros(self.d, 1, dtype=torch.float32, device=self.device)

                # Update true state
                if self.nl:
                    x_nl = self.nonlinear_dynamics(x)
                    x = self.A @ x_nl + self.B @ u_t + w_t
                else:
                    x = self.A @ x + self.B @ u_t + w_t

                # Get new observation
                y_obs = self.C @ x
                y_history.append(y_obs.detach())
                
                # Compute natural observation
                y_nat = self.compute_natural_observation(y_history, u_history)
                y_nat_history.append(y_nat.detach())

                # Compute actual quadratic cost for this step (for evaluation)
                actual_cost = self.compute_loss(y_obs, u_t)
                costs[t] = actual_cost.item()
                

                # Update controller parameters using proxy loss
                if use_control and t >= 30:
                    self.update_M_matrices_new(y_nat_history, u_history)

            all_costs[trial, :] = costs
           

        # Store average costs if multiple trials
        if num_trials > 1: self.losses = torch.mean(all_costs, dim=0)
        else: self.losses = all_costs[0]
            
        return self.losses
    
    def reset(self):
        """Reset all trajectories and losses"""
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.device)



    """
        Compute control using the DSC formula:
        u_t = M_bar y_t^nat + 
              sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^(1/4) [phi_i]_j M_i y_{t-j}^nat +
              sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^(1/4) [phi_l]_k M_bar y_{t-k}^nat +
              sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l * lambda_i)^(1/4) [phi_l]_k [phi_i]_j M_i y_{t-j-k}^nat
        """
        # Ensure we have enough history
        # if len(y_nat_history) < max(self.m, self.m_tilde) + 1:
        #     # If not enough history, fall back to basic control
        #     u_t = torch.zeros(self.n, 1, device=self.device)
        #     if len(y_nat_history) > 0:
        #         u_t = self.M_bar @ y_nat_history[-1]
        #     return u_t
            
        # # First term: M_bar y_t^nat
        # u_t = self.M_bar @ y_nat_history[-1]
        
        # # Second term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} lambda_i^(1/4) [phi_i]_j M_i y_{t-j}^nat
        # for i in range(self.h_tilde):
        #     lambda_i_pow = self.lambda_e[i] ** 0.25
        #     for j in range(min(self.m_tilde, len(y_nat_history)-1)):
        #         if j+1 < len(y_nat_history):
        #             phi_i_j = self.phi_tilde[j, i]
        #             u_t = u_t + lambda_i_pow * phi_i_j * (self.M[min(i+1, len(self.M)-1)] @ y_nat_history[-(j+1)])
        
        # # Third term: sum_{l=0}^{h} sum_{k=0}^{m} sigma_l^(1/4) [phi_l]_k M_bar y_{t-k}^nat
        # for l in range(self.h):
        #     sigma_l_pow = self.sigma[l] ** 0.25
        #     for k in range(min(self.m, len(y_nat_history))):
        #         if k < len(y_nat_history):
        #             phi_l_k = self.phi[k, l]
        #             u_t = u_t + sigma_l_pow * phi_l_k * (self.M_bar @ y_nat_history[-(k+1)])
        
        # # Fourth term: sum_{i=1}^{h_tilde} sum_{j=1}^{m_tilde} sum_{l=0}^{h} sum_{k=0}^{m} (sigma_l * lambda_i)^(1/4) [phi_l]_k [phi_i]_j M_i y_{t-j-k}^nat
        # for i in range(self.h_tilde):
        #     for j in range(min(self.m_tilde, len(y_nat_history))):
        #         for l in range(self.h):
        #             for k in range(min(self.m, len(y_nat_history))):
        #                 if j+k+1 < len(y_nat_history):
        #                     combined_pow = (self.sigma[l] * self.lambda_e[i]) ** 0.25
        #                     phi_l_k = self.phi[k, l]
        #                     phi_i_j = self.phi_tilde[j, i]
        #                     u_t = u_t + combined_pow * phi_l_k * phi_i_j * (self.M[min(i+1, len(self.M)-1)] @ y_nat_history[-(j+k+1)])
            
        # return u_t