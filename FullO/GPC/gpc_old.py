import torch
import torch.nn as nn
import torch.optim as optim

from numbers import Real
from typing import Callable

from utils import lqr


def quad_loss(x: torch.Tensor, u: torch.Tensor) -> Real:
    return torch.sum(x.T @ x + u.T @ u)


class GPC_OLD(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor = None,
        R: torch.Tensor = None,
        K: torch.Tensor = None,
        start_time: int = 0,
        cost_fn: Callable[[torch.Tensor, torch.Tensor], Real] = None,
        H: int = 3,
        HH: int = 2,
        lr_scale: Real = 0.005,
        decay: bool = True,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (torch.Tensor): system dynamics
            B (torch.Tensor): system dynamics
            Q (torch.Tensor): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (torch.Tensor): cost matrices (i.e. cost = x^TQx + u^TRu)
            K (torch.Tensor): Starting policy (optional). Defaults to LQR gain.
            start_time (int): 
            cost_fn (Callable[[torch.Tensor, torch.Tensor], Real]): loss function
            H (int): history of the controller
            HH (int): history of the system
            lr_scale (Real): learning rate scale
            decay (bool): whether to decay learning rate
        """
        super(GPC_OLD, self).__init__()

        cost_fn = quad_loss

        d_state, d_action = B.shape  # State & Action Dimensions

        self.A, self.B = A, B  # System Dynamics

        self.t = 0  # Time Counter (for decaying learning rate)

        self.H, self.HH = H, HH

        self.lr_scale, self.decay = lr_scale, decay

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        # TODO: need to address problem of LQR with PyTorch
        self.K = K if K is not None else lqr(A, B, Q, R)

        self.M = torch.zeros((H, d_action, d_state), dtype=torch.float32)

        # Past H + HH noises ordered increasing in time
        self.noise_history = torch.zeros((H + HH, d_state, 1), dtype=torch.float32)

        # past state and past action
        self.state, self.action = torch.zeros((d_state, 1), dtype=torch.float32), torch.zeros((d_action, 1), dtype=torch.float32)

        self.cost_fn = cost_fn



    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (torch.Tensor): current state

        Returns:
            torch.Tensor: action to take
        """
        action = self.get_action(state)
        self.update(state, action)
        return action

    def update(self, state: torch.Tensor, u: torch.Tensor) -> None:
        """
        Description: update agent internal state.

        Args:
            state (torch.Tensor): current state
            u (torch.Tensor): current action

        Returns:
            None
        """
        noise = state - self.A @ self.state - self.B @ u
        self.noise_history = torch.roll(self.noise_history, shifts=-1, dims=0)
        self.noise_history[0] = noise

        delta_M, delta_bias = self._compute_gradients(self.M, self.noise_history)

        lr = self.lr_scale
        lr *= (1 / (self.t + 1)) if self.decay else 1
        self.M -= lr * delta_M
        self.bias -= lr * delta_bias

        # update state
        self.state = state

        self.t += 1

    def _compute_gradients(self, M: torch.Tensor, noise_history: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients for the policy loss.

        Args:
            M (torch.Tensor): Model parameter
            noise_history (torch.Tensor): History of noise

        Returns:
            torch.Tensor: gradients
        """
        # Example of a simple gradient computation. Replace with actual policy loss computation
        action = -self.K @ self.state + torch.tensordot(M, noise_history, dims=([0, 2], [0, 1]))
        loss = self.cost_fn(self.state, action)
        loss.backward()
        return M.grad, None  # Placeholder for delta_bias

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Description: get action from state.

        Args:
            state (torch.Tensor): current state

        Returns:
            torch.Tensor: action
        """
        return -self.K @ state + torch.tensordot(self.M, self.noise_history, dims=([0, 2], [0, 1]))
    

# GPC (The one presented in the paper)

# import numpy as np
# import scipy as scp
# import matplotlib.pyplot as plt
# import torch
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.nn.functional import relu, leaky_relu
# from scipy.linalg import solve_discrete_are as dare

# from utils import lqr

# class GradientPerturbationController(torch.nn.Module):
#     def __init__(self, A, B, Q, R, h, eta, W_test, name, nl=False):
#         super().__init__()
#         self.name = name
#         self.nl = nl
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device
#         #self.register_buffer("W_test",torch.tensor(W_test, dtype=torch.float32))
        
#         self.register_buffer("A",torch.tensor(A, dtype=torch.float32))
#         self.register_buffer("B",torch.tensor(B, dtype=torch.float32))
#         self.register_buffer("Q",torch.tensor(Q, dtype=torch.float32))
#         self.register_buffer("R",torch.tensor(R, dtype=torch.float32))
#         self.h = h  # memory
#         self.eta = eta
#         self.register_buffer("K", lqr(A, B, Q, R))
#         #self.M = torch.nn.Parameter(torch.zeros(h, B.shape[1], A.shape[0]))
#         self.M = torch.nn.Parameter(torch.zeros(h, B.shape[1], A.shape[0]))
#         self.bias = torch.nn.Parameter(torch.zeros(h, B.shape[1], 1))
#         self.register_buffer("x", torch.zeros(A.shape[0], 1, dtype=torch.float32))

#         self.w_test = torch.stack(W_test).to(self.device)
        
#     def nonlinear_dynamics(self, x):
#         return relu(x)  # Example nonlinearity

  
#     def run(self, T):
#         self.cuda()
#         self.losses = torch.zeros(T, dtype=torch.float32, device=self.A.device)
#         optimizer = torch.optim.Adagrad(self.parameters(), lr=self.eta)  #
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Decays LR by 10% every 50 steps

#         #print(self.w_test)

#         # Sinusoid perturbation
#         t_values = torch.linspace(0, 2 * torch.pi, T, dtype=torch.float32, device=self.A.device)

#         w_t_history_all_rand_sin = [
#             torch.sin(t_values[i])  * torch.ones((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device)
#             for i in range(T)
#         ]
#         #print("W HIST SIN", w_t_history_all_rand_sin)


#         # Gaussian perturbation
#         w_t_history_all_rand_gaussian = [torch.randn((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device) for i in range(T)]

#         # Uniform perturbation  
#         w_t_history_all_rand_uniform = [(2 * torch.rand((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device) - 1) for i in range(T)]

#         # Constant perturbation
#         w_t_history_all_rand_const = [0.2 * torch.ones((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device)
#     for _ in range(T)]

#         # Zero perturbation
#         w_t_history_all_zeros = [torch.zeros((self.B.shape[0], 1), dtype=torch.float32, device=self.A.device) for i in range(T)]
        


#         for t in range(1, T):

#             if self.M.grad is not None: self.M.grad.zero_()   # Reset gradients
#             #optimizer.zero_grad()
           
#             if t<self.h: indices = list(range(t))       # dynamically learn history
#             else:   indices = list(range(t- self.h, t)) # access last h elements for history

#             #print("INDICES", indices)
                
#             #indices = list(range(t-1, t-self.h-1, -1)) (WORKING)
#             #if t > 4: indices = list(range(4 + t, t - 5, -1))  # Ensure valid index range (MAYBE)
                
#             #variable = t + 4
#             #print("x INSIDE", self.x)
#             #print("WHOLE HISTORY", self.w_test)
#             # Get noise and compute controls:
#             # M is (3,2,4) w (4,1)
#             #controlled_noise = sum(self.M[i, :, :] @ self.w_test[t-i] for i in range(self.h))
#                #print("SHAPE", self.M[i,:,:].shape)
#             #controlled_noise = sum(self.M[i, :, :] @ self.w_test[-i:] for i in range(self.h))
#             # controlled_noise = torch.sum(torch.stack([self.M[i, :, :] @ self.w_test[-i] for i in range(self.h)]), dim=0)
#             # print("PRE-COMPUTE", controlled_noise)
#             #print("NEW ITERATION:", t)
#             # for i in range(self.h):
#             #         #print("HISTORY [t-i]:", self.w_test[variable-i] )
#             #         controlled_noise = sum(self.M[i, :, :] @ self.w_test[i:2i] for i in range(self.h))
#             #controlled_noise = sum(self.M[i, :, :] @ self.w_test[-i] for i in range(self.h))
#             index_length = len(indices)
#             # for i in range(index_length):
#             #     # print("W TEST BEFORE ACCSESS", self.w_test)
#             #     # print("W TEST i (LOOKBACK)", self.w_test[t-i])
#             #     # print("HISTORY SHAPE", self.w_test[i].shape)
#             #     selected_w_test = self.w_test[indices]  # Select rows from w_test
#             #     #print("SELECT w_i", selected_w_test[i])
#             #     print("AAOOAOAOAOAOAOOST", selected_w_test)
#             #     print("I GO THROUGH NOISE:", selected_w_test[index_length-i-1])
#             #     past_noise = sum(self.M[i, :, :] @ selected_w_test[index_length-i-1])  # Adjust for matrix multiplication
               
#                 #past_noise2 = sum(self.M[i, :, :] @ self.w_test[i])

#             past_noise = sum(self.M[i, :, :] @ self.w_test[indices][index_length-i-1] for i in range(index_length))
#             # past_noise = 0 
#             # for i in range(self.h):
#             #     selected_w_i = self.w_test[indices][i]  # Extract the i-th element
#             #     #print(f"Selected w_test[{i}]:\n", selected_w_i)  # Print each selected w_test[i]

#             #     past_noise += self.M[i, :, :] @ selected_w_i  # Perform matrix multiplication
#             # #past_noise2 = sum(self.M[i, :, :] @ self.w_test[i] for i in range(self.h))
            
#             #     print("Past noise", past_noise)

#             #print("SELF T-i", self.w_test[t-i])
#                 #print("M MATRIX_i", self.M[i, :,:])
#             #print("M matrix:", self.M)
#             #print("PAST NOISE", past_noise)
                

#             # Extract the relevant past values using convolution-style slicing
#             # Perform batched matrix multiplication
                
#             u = - self.K @ self.x + past_noise
#             #print("CONTROL", u)
            
#             #print(controlled_noise)
#             #print(w_t_history.shape)
#             #print(controlled_noise.shape)

#             # print(f"self.x shape: {self.x.shape}")  # (n,1)
#             # print(f"self.K shape: {self.K.shape}")  # (m,n)
#             # print(f"controlled_noise shape: {controlled_noise.shape}")  # (m,1)
#             # print(f"controlled noise: {controlled_noise}") 
#             #print(controlled_noise)
#             #u = - self.K @ self.x + controlled_noise #  - K_t * x_t + sum of (M_i * w_t-i)
           
#             # print(f"u shape: {u.shape}")  # (m,1)
#             # Observe next state and compute new disturbance:
#             ####################################
#             # WORKING VERSION:
#             # x_next = self.A @ self.x + self.B @ u + w_t_history[t % self.h]  # get the next state (check the % self.h)
#             # # print(f"x_next shape: {x_next.shape}")  # (n,1)

#             # new_w_t = x_next - self.A @ self.x - self.B @ u        # compute the new perturbation
#             # print(f"new_w_t shape: {new_w_t.shape}")  # (n,1)
#             ####################################

#             ####################################
#             # TESTING SECTION:
#             # Introduce nonlinearity in state transition
#             if self.nl:
#                 x_nl = self.nonlinear_dynamics(self.x)
#                 x_next = self.A @ x_nl + self.B @ u + self.w_test[t]
#                 #new_w_t = x_next - self.A @ x_nl - self.B @ u
#             else:

#                 #x_next = self.A @ self.x + self.B @ u + self.w_test[t % self.h] 
#                 #print("W IN STATE UPDATE", self.w_test[t])
#                 x_next = self.A @ self.x + self.B @ u + self.w_test[t]
#                 #print("X NEXT", x_next)
#                 #new_w_t = x_test - self.A @ self.x - self.B @ u
#                 #print("NEW W T SHAPE",new_w_t.shape)
#                # print("W T HISTORY SHAPE",self.w_test.shape)

            
#             ####################################

#             #self.w_test = torch.roll(self.w_test, shifts=-1, dims=0)  # Shift left
#             #self.w_test[-1] = new_w_t.detach()  # Insert new disturbance at the last position
#             # Update w_t history
#             # self.w_test.pop(0)  
#             # self.w_test.append(new_w_t.detach()
            
#             #self.w_test = self.w_test.roll(1,-1)
#             #print("W HISTORY ROLLED", self.w_test)
#             #self.w_test[:, 0] = new_w_t.detach().view(1, 2)

#             # Compute cost and construct loss:
#             cost = self.x.T @ self.Q @ self.x + u.T @ self.R @ u 
        
#             #print(f"Cost shape: {cost.shape}") # cost has to be [1,1]
#             #print(f"cost shape: {cost.shape}")  # (1,1)
    
#             #self.M.grad.zero_() 
#             #optimizer.zero_grad()    # Reset gradients
#             #if t >= 10:  # Start learning M after the 10th ste
            
                
#             cost.backward()  # Compute gradients
#             self.losses[t] = cost.item()  # Store cost in the tensor
#             #optimizer.step()

#             #self.eta = self.eta * (1 / (t + 1))  # decay in the learning rate

#             #optimizer.step()
#             scheduler.step()
            
#             with torch.no_grad():  # Disable gradient tracking for manual update
#                 self.M -= self.eta * self.M.grad # Gradient step

            
#                 for i in range(self.h):  
#                     norm = torch.norm((self.M), p=2)  
                    
#                     if norm > ((1 - 0.1) ** t):
#                         self.M[i, :, :] *= ((1 - 0.1) ** t) / norm  # Projection step


#                 #self.M.grad = None
                
              

            
#             # with torch.no_grad():
#             # #     norm = torch.norm(self.M, p=2)
#             # #     if norm > 0.6: self.M *= (0.6 / norm)
#             #     for i in range(self.h):
#             #          norm = torch.norm(self.M[i, :, :], p=2)  # Compute L2 norm
#             #          if norm > (1-0.2)**t:
#             #              self.M[i, :, :] *= (((1-0.2)**t) / norm)  # Scale down to satisfy constraint
#             #    print("Learned matrix M ITEM NORM:", torch.norm(self.M[i, :, :]))
#             # with torch.no_grad(): self.M *= 0.999  # Scale M by 1/2

#             #print("M INSIDE", t, self.M)
            
            

           
#             self.x = x_next.detach()
#             #print("Learned matrix M NORM:", torch.norm(self.M))
    
#     def plot_loss(self):
#         plt.plot(self.losses)
#         plt.xlabel('Time Step')
#         plt.ylabel('Loss')
#         plt.title('Gradient Perturbation Controller Loss Over Time')
#         plt.show()

