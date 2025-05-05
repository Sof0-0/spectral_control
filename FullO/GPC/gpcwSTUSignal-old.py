import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import relu, leaky_relu 

from utils import lqr, get_hankel, project_l2_ball

class GPCwSTU(torch.nn.Module):
    def __init__(self, A, B, Q, R, h, eta, T, name, nl=False):
        super().__init__()
        self.name = namem
        self.nl = nl
        self.T = T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device
        self.register_buffer("A",torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B",torch.tensor(B, dtype=torch.float32))
        self.register_buffer("Q",torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("R",torch.tensor(R, dtype=torch.float32))
        self.h = h  # memory
        self.eta = eta
        self.register_buffer("K", lqr(A, B, Q, R))
        

        self.d = A.shape[0]  # State dimension
        self.n = B.shape[1]  # Control dimension

        self.M = torch.nn.Parameter(torch.zeros(h, self.n, self.d)) # 0.009
        self.M_stu  = torch.nn.Parameter(torch.randn(20, self.d, self.n) * 0.09)  # 0.01
        self.register_buffer("x", torch.zeros(self.d, 1, dtype=torch.float32))

         # STU filters:
        Z_T = get_hankel(self.T) 
        eigvals, eigvecs = torch.linalg.eigh(Z_T)  # Compute eigenpairs
        self.register_buffer("sigma_stu", eigvals[-20:].clone().detach().to(torch.float32))# Top-k eigenvalues
        self.register_buffer("phi_stu", eigvecs[:, -20:].clone().detach().to(torch.float32)) # Corresponding eigenvectors
        

        
        # Sinusoid perturbation
        t_values = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float32, device=self.A.device)

        w_t_history_all_rand_sin = [
            torch.sin(t_values[i])  * torch.ones((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device)
            for i in range(T)
        ]

        # Gaussian perturbation
        w_t_history_all_rand_gaussian = [torch.randn((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device)  for i in range(T)]

        # Uniform perturbation  
        w_t_history_all_rand_uniform = [(2 * torch.rand((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device) - 1) for i in range(T)]

        
    def nonlinear_dynamics(self, x):
        return leaky_relu(x)  # Example nonlinearity
  
    def run(self, T=100):
        self.to(self.device)
        # print(f"self.A shape: {self.A.shape}") 
        # print(f"self.B shape: {self.B.shape}") 
        # print(f"self.Q shape: {self.Q.shape}") 
        # print(f"self.R shape: {self.R.shape}") 
        # print(f"self.K shape: {self.K.shape}") 
        # print(f"self.M shape: {self.M.shape}") 
        #optimizer = torch.optim.Adagrad(self.parameters(), lr=self.eta)  # Adagrad optimizer
        self.losses = torch.zeros(T, dtype=torch.float32, device=self.A.device)

        # each w_t is (1,1) since B is (n,1)
        
        u_t_history = torch.zeros((self.n, T), dtype=torch.float32, device=self.A.device)  # Store past controls (history now lives in T) 

        for t in range(T):
            
            # Get noise and compute controls:
            # M is (3,2,4) w (4,1)
            controlled_noise = sum(self.M[i, :, :] @ self.w_t_history_all_rand_sin[i] for i in range(self.h))

            # print(f"self.x shape: {self.x.shape}")  # (n,1)
            # print(f"self.K shape: {self.K.shape}")  # (m,n)
            # print(f"controlled_noise shape: {controlled_noise.shape}")  # (m,1)
            # print(f"controlled noise: {controlled_noise}") 
            #print(controlled_noise)
            u_orig = -self.K @ self.x + controlled_noise #  - K_t * x_t + sum of (M_i * w_t-i)

            #radius_u = 1e11  # Define an appropriate radius
            #u_orig = project_l2_ball((-self.K @ self.x + controlled_noise), radius_u)
            
            #u_orig = torch.clamp(u_orig, min=-1e15, max=1e15)
            
            # Udpate the history and add the new item
            u_t_history       = u_t_history.roll(1,-1)
            u_t_history[:, 0] = u_orig.detach()[:,0]
            #print("UT HISTORY", u_t_history)

            
            # STU signal STEP:
            x_next = sum(
                self.M_stu[i] @ (u_t_history @ self.phi_stu[:, i:i+1]) for i in range(20)
    
            )

            
            # x_next = self.A @ self.x + self.B @ u_orig + w_t_history[t % self.h] PREVIOUS
            #new_w_t = x_next - self.A @ self.x - self.B @ u_orig
            #radius_w = 1e11  # Define an appropriate radius
            #new_w_t = project_l2_ball((x_next - self.A @ self.x - self.B @ u_orig), radius_w)
            #new_w_t = torch.clamp(new_w_t, min=-1e15, max=1e15)  # Prevent explosion
            
            # print(f"u shape: {u.shape}")  # (m,1)
            # Observe next state and compute new disturbance:
            ####################################
            # WORKING VERSION:
            # x_next = self.A @ self.x + self.B @ u + w_t_history[t % self.h]  # get the next state (check the % self.h)
            # # print(f"x_next shape: {x_next.shape}")  # (n,1)

            # new_w_t = x_next - self.A @ self.x - self.B @ u        # compute the new perturbation
            # print(f"new_w_t shape: {new_w_t.shape}")  # (n,1)
            ####################################

            ####################################
            # WORKING VERSION:
            # Introduce nonlinearity in state transition
            # if self.nl:
            #     x_nl = self.nonlinear_dynamics(self.x)
            #     x_next = self.A @ x_nl + self.B @ u + w_t_history[t % self.h]
            #     new_w_t = x_next - self.A @ x_nl - self.B @ u
            # else:
            #     x_next = self.A @ self.x + self.B @ u + w_t_history[t % self.h] 
            #     new_w_t = x_next - self.A @ self.x - self.B @ u
                
            ####################################
            
            # Update w_t history
            # self.w_t_history.pop(0)  
            # self.w_t_history.append(new_w_t.detach()) 
            # #print("W HIST", self.w_t_history)

            # Compute cost and construct loss:
            cost = self.x.T @ self.Q @ self.x + u_orig.T @ self.R @ u_orig 
            # print(f"cost shape: {cost.shape}")  # (1,1)


            #optimizer.zero_grad()    # Reset gradients
            cost.backward()
            #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max(1.0, torch.norm(cost).item() / 10))  # Adjust max_norm as needed

            #optimizer.step()        # gradient step
            # with torch.no_grad(): 
            #     self.M *= 0.1  # Scale M by 1/2
            #     self.M_stu *= 0.1


            #print("NORM OF M", torch.norm(self.M))
            #print("NORM OF M_STU", torch.norm(self.M_stu))
            with torch.no_grad():
                self.M -= self.eta * self.M.grad  # Gradient step
            #     norm = torch.norm(self.M, p=2)
            #     if norm > 0.6: self.M *= (0.6 / norm)
                for i in range(self.h):
                     norm = torch.norm(self.M[i, :, :], p=2)  # Compute L2 norm
                     if norm > (1-0.1)**t:
                         self.M[i, :, :] *= (((1-0.1)**t) / norm)  # Scale down to satisfy constraint
                self.M.grad.zero_()  # Manually reset gradients
                
            self.losses[t] = cost.item()  # Store cost in the tensor
            self.x = x_next.detach()
    
    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Time Step')
        plt.ylabel('Loss')
        plt.title('Gradient Perturbation Controller Loss Over Time')
        plt.show()
