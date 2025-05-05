import matplotlib as plt
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import lqr, get_hankel, get_hankel_new
from torch.nn.functional import leaky_relu, relu, tanh


"""
New Algorithm:

1. Z now lives in k x k, so phi_s are k x h (if the history stays top h)
2. W is in d x k with each w_t_new in d x 1
3. Small gamma that calculates each eigenvalue in the Z_k matrix

"""

class OSFCwNF(torch.nn.Module):
    def __init__(self, A, B, Q, R, h, eta, gamma, m, T, name, nl=False, flag=False):
        super().__init__()
        self.name = name
        self.nl = nl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device
        self.register_buffer("A",torch.tensor(A, dtype=torch.float32))
        self.register_buffer("B",torch.tensor(B, dtype=torch.float32))
        self.register_buffer("Q",torch.tensor(Q, dtype=torch.float32))
        self.register_buffer("R",torch.tensor(R, dtype=torch.float32))
        self.h = h           # Filter parameter (amount we pick - history / top h eigenpairs)
        self.m = m           # Dimension of the new Hankel Matrix Z_m (order of log T)
        self.gamma = gamma   # (very small)
        self.flag = flag
        self.eta = eta  # Step size
        self.register_buffer("K", lqr(A, B, Q, R))
        self.T = T

        self.d = A.shape[0]  # State dimension
        self.n = B.shape[1]  # Control dimension

        # Compute top-k eigenpairs of Z_T (for now, assume random for testing)

        Z_m = get_hankel_new(self.m, self.gamma)  

        # For the new filters:
        #Z_new = get_hankel_new(self.T)  # has to be the dimension k, where k is defined in the Lemma A4
        #print("Z_T", Z_T)

        eigvals, eigvecs = torch.linalg.eigh(Z_m)  # Compute eigenpairs
        self.register_buffer("sigma", eigvals[-h:].clone().detach().to(torch.float32))  # Top-k eigenvalues
        self.register_buffer("phi", eigvecs[:, -h:].clone().detach().to(torch.float32))  # Corresponding eigenvectors

        #print(self.phi)
        #print(f"filters shape: {self.phi.shape}") (T x h)

        self.M = torch.nn.Parameter(torch.zeros(self.h, self.n, self.d))  # Initialize M (lives in h x n x d)
        # Initialize the first entry with random values
        # M_init = torch.randn(1, B.shape[1], A.shape[0]) * 0.01
        # zeros = torch.zeros(h - 1, B.shape[1], A.shape[0])
        # self.M = torch.nn.Parameter(torch.cat([M_init, zeros], dim=0))

    
        self.register_buffer("x", torch.randn(self.d, 1, dtype=torch.float32))


        # Sinusoid perturbation
        t_values = torch.linspace(0, 2 * torch.pi, self.m, dtype=torch.float32, device=self.device) 
        
        sinusoid = torch.sin(t_values) # (m, 1)
        self.w_t_history_all_rand_sin = sinusoid.unsqueeze(0).expand(self.d, -1) #(d,m)

        # Constant perturbation
        constant_value = 0.2  # You can change this to any constant
        self.w_t_history_all_const = torch.full((self.d, self.m), constant_value, dtype=torch.float32, device=self.device)


        # Gaussian perturbation
        self.w_t_history_all_rand_gaussian = torch.randn((self.d, self.m), dtype=torch.float32, device=self.device)

        # Zero perturbation
        self.w_t_history_all_zeros = torch.zeros((self.d, self.m), dtype=torch.float32, device=self.device)



        # Uniform perturbation  
        # self.w_t_history_all_rand_uniform = (2 * torch.rand((self.d, self.m), dtype=torch.float32, device=self.A.device) - 1)

        # Initialize test perturbations
        self.W_test = self._initialize_sinusoidal_disturbances()
        
        # Store the test perturbation sequence
        self.w_test = [w.to(self.device) for w in self.W_test] if self.W_test is not None else None
        
        # Initialize disturbance history buffer
        self.register_buffer("w_history", torch.zeros(self.m, self.d, 1))


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
        return relu(x)
        #return tanh(x)  # Example nonlinearity

    def run(self):
        self.to(self.device)
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.A.device)
        #optimizer = torch.optim.Adagrad(self.parameters(), lr=self.eta)  # Adagrad optimizer

        #w_t is in d x (T+1) where d is the dimension of the state x with each w_t being (d x 1)(10 x 101)
        #w_t_history = torch.zeros((self.d, self.m), dtype=torch.float32, device=self.A.device)  # W new lives in d x m

        #w_t_history = [0.1 * torch.randn((self.d, 1), dtype=torch.float32) for _ in range(T)]
        #print(f"w_t_history: {w_t_history}") 
        
        for t in range(self.T):

            #print("size of w_t_history", len(w_t_history))
            # Construct w̃_{t:1} = [w_t, ..., w_1, 0, ..., 0] ∈ ℝ^{d×(T+1)}
            #w_tilde = torch.cat(w_t_history, dim=1) # concatinate past disturbances along time dim (keep w_tilde)
            #print(f"w_tilde shape: {w_tilde.shape}") 
            #print(f"w_tilde: {w_tilde}") 
            # gamma = 0.9  # Decay factor
            # w_tilde_dec = torch.cat([(gamma ** i) * w_t for i, w_t in enumerate(w_t_history)], dim=1)           
            #print(w_t_history) # currently gives all 0s
            w_t = self.w_test[t % len(self.w_test)] if t < len(self.w_test) else torch.zeros_like(self.w_test[0])
            # # Compute control u_t = u_{t-1} + \sum_{i=1}^h  M_i^t  \tilde{W}_{t-1:1} \phi_i
            controlled_noise = sum(
            #     # currently has phi[:, i:i+1] in (100 x 1) and w_tilde in (10 x 101)
            #     # currently has M[i] as [4 x 10], which is right [control, state]

            #     # See if we could approx u_t(k) with K(x_t - x_t-1) + u^K_t-1 (where u^K_t-1 could be the best one),
            #     # since we need the previous u_t-1 for learning

            #     # stationary (since M_i does not change with t)
            #     # non-stationary we would have M^t_i that will be affecting W and filter

            #     # w_tilde is 0 throughout, how do we get the weights? 
                # Previously had w_tilde instead of w_t_history
                (self.sigma[i] ** 0.25) * self.M[i] @ (self.w_test @ self.phi[:, i:i+1]) for i in range(self.h) # keep the index i from the history iteration
             )
            
            # Debugging loop:
            # controlled_noise = torch.zeros_like(self.M[0] @ (w_t_history @ self.phi[:, 0:1]))
            # for i in range(self.h):
            #     controlled_noise += self.M[i] @ (w_t_history @ self.phi[:, i:i+1])
                # print("M[i] shape: ", self.M[i].shape)
                # print("M[i]: ", self.M[i])
                

            # print(f"controlled noise: {controlled_noise}") 

            # TEMP DEBUGGING LOOP:
            # noise_final = torch.zeros_like(self.M[0] @ (w_tilde @ self.phi[:, :1]))
            # for i in range(self.h):
            #     temp = self.M[i] @ (w_tilde @ self.phi[:, i:i+1])
            #     print("Self M[i].shape:", self.M[i].shape) # M[i] is 4 x 10 (n x d)
            #     print("w_tilde:", w_tilde.shape)           # W_tilde is 10 x 100
            #     print("Self phi[:, :1]", self.phi[:, :1].shape) # is 100 x 1
            #     noise_final += temp

            # TEMP DEBUGGING LOOP^^^^^^^^^^^

            # New rationale:
            # stationary: u_t ^ M = (1-gamma) u^M_t-1 + summation(x) can capture all linear policies for A' = A / (1-gamma) where |A'| <=1


            if self.flag: u_stat = controlled_noise # previously had dependency on u_{t-1]}
            else: u_stat = -self.K @ self.x + controlled_noise
                
            
            if self.nl:
                x_nl = self.nonlinear_dynamics(self.x)
                x_next = self.A @ x_nl + self.B @ u_stat + w_t
                #new_w_t = x_next - self.A @ x_nl - self.B @ u_stat
                #print("Here:", new_w_t)
            else:
                x_next = self.A @ self.x + self.B @ u_stat + w_t
                #new_w_t = x_next - self.A @ self.x - self.B @ u_stat
                #print("Not here:", new_w_t)

            #print("New w_t shape", new_w_t.shape)
            # x_next = self.A @ self.x + self.B @ u_stat + w_t_history[:, 0:1]  # get the next state (check the % self.h)
            #x_next = self.A @ self.x + self.B @ u + w_tilde  # get the next state (check the % self.h)
            #print(f"x_next - A @ x: {x_next - self.A @ self.x}")  # (n,1) --- 0
            #print(f"B @ u: {self.B @ u}")  # (n,1) --- 0 

            # new_w_t = x_next - self.A @ self.x - self.B @ u_stat        # compute the new perturbation
            #print(f"new_w_t: {new_w_t}")  # new_w_t shape is (n,1)

            # Update w_t history
            # self.w_t_history = self.w_t_history.roll(1,-1)
            # self.w_t_history[:, 0] = new_w_t.detach()[:,0]
            #print(w_t_history)

            # Compute cost
            # Compute cost and construct loss:
            cost = self.x.T @ self.Q @ self.x + u_stat.T @ self.R @ u_stat # LQR cost? ----- convex/quadratic? (diff for GPC)
            #print(f"cost shape: {cost.shape}")  # (1,1)


            #optimizer.zero_grad()    # Reset gradients
            cost.backward()
            #print("M gradient", torch.norm(self.M.grad))
            #optimizer.step()        # gradient step

            with torch.no_grad():
                self.M -= self.eta * self.M.grad  # Gradient step
            
                for i in range(self.h):
                    norm = torch.norm(self.M[i, :, :], p=2)  # Compute L2 norm
                    
                    if norm > (np.sqrt(2/self.gamma)):
                        self.M[i, :, :] *= (np.sqrt(2/self.gamma) / norm)  # Scale down to satisfy constraint
                self.M.grad.zero_()
            # #     norm = torch.norm(self.M, p=2)
            # #     if norm > 0.6: self.M *= (0.6 / norm)
            #     for i in range(self.h):
            #         norm = torch.norm(self.M[i, :, :], p=2)  # Compute L2 norm
            #         if norm > (1 / np.sqrt(self.gamma):
            #             self.M[i, :, :] *= (((1 / np.sqrt(self.gamma)) / norm)  # Scale down to satisfy constraint
            #         print("Learned matrix M ITEM NORM:", torch.norm(self.M[i, :, :]))

            #with torch.no_grad(): self.M *= 0.5  # Scale M by 1/2
            self.losses[t] = cost.item()  # Store cost in the tensor

            # Projection step if ||M||_F >= RM
            # frob_norm = torch.norm(self.M, p='fro')
            # if frob_norm >= self.RM:
            #     self.M.data = (self.RM / frob_norm) * self.M.data  # Frobenius norm projection

            # Update state
            self.x = x_next.detach()
            #print(cost)
            #print("Learned matrix M", torch.norm(self.M))