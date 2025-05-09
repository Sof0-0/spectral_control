import matplotlib as plt
import torch
import matplotlib.pyplot as plt
from utils import  lqr, get_hankel, get_hankel_new, project_l2_ball
from torch.nn.functional import leaky_relu


class OSFCNFwSTU(torch.nn.Module):
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

    

        # OSFC filters: 
        Z_m = get_hankel_new(self.m, self.gamma)  
        
        eigvals, eigvecs = torch.linalg.eigh(Z_m)  # Compute eigenpairs
        self.register_buffer("sigma", eigvals[-h:].clone().detach().to(torch.float32))  # Top-k eigenvalues
        self.register_buffer("phi", eigvecs[:, -h:].clone().detach().to(torch.float32))  # Corresponding eigenvectors


        # STU filters:
        self.register_buffer("sigma_stu", eigvals[-20:].clone().detach().to(torch.float32))# Top-k eigenvalues
        self.register_buffer("phi_stu", eigvecs[:, -20:].clone().detach().to(torch.float32)) # Corresponding eigenvectors
        
        #print(f"sigma stu shape: {self.sigma_stu.shape}") # should be 20 (right)
        #print(f"filters stu shape: {self.phi_stu.shape}") #  they are 100 x 20 rn

        
        self.M      = torch.nn.Parameter(torch.randn(self.h, self.n, self.d) * 0.02)    # Initialize M (lives in h x n x d)
        self.M_stu  = torch.nn.Parameter(torch.randn(20, self.d, self.n) * 0.1)        # M for STU with fixed 20 filters  (20 x n x d)

        # Removing scalar in M_stu causes more harm than removing one in M
        
        self.register_buffer("x", torch.randn(self.d, 1, dtype=torch.float32))

        self.w_t_history = torch.zeros((self.d, self.m), dtype=torch.float32, device=self.device)
        self.w_t_history[:, 0] = torch.randn((self.d,), dtype=torch.float32)

        self.u_t_history = torch.zeros((self.n, self.m), dtype=torch.float32, device=self.device)
        

    def nonlinear_dynamics(self, x):
        return leaky_relu(x)  # Example nonlinearity
    

    def run(self):
        
        self.to(self.device)
        self.losses = torch.zeros(self.T, dtype=torch.float32, device=self.A.device)
        #optimizer = torch.optim.Adagrad(self.parameters(), lr=self.eta)  # Adagrad optimizer

        #w_t is in d x (T+1) where d is the dimension of the state x with each w_t being (d x 1)(10 x 101)
       # Only set the first input to be Gaussian
        #alternating_values = torch.tensor([(-1) ** i for i in range(self.d)], dtype=torch.float32) * 1e-25
        #self.w_t_history[:, 0] = alternating_values 
         # Create a tensor of indices and apply torch.sin()
        #self.w_t_history[:, 0] = torch.sin(torch.arange(self.d, dtype=torch.float32, device=self.device))

        # Init u_history:
        #self.u_t_history = torch.zeros((self.n, self.T), dtype=torch.float32, device=self.device)  # Store past controls (history now lives in T) 

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
                 (self.sigma[i] ** 0.25) * self.M[i] @ (self.w_t_history @ self.phi[:, i:i+1]) for i in range(self.h) # keep the index i from the history iteration
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


            #print(w_tilde)
            # get the u_t-1 in the update here
            #u = -self.K @ self.x + controlled_noise # previously had dependency on u_{t-1]}

            # STEP 1: get the original computed control (CONFIRMED)
            if self.flag: u_stat =  controlled_noise # Assuming no LQR K term here (check if - sign is needed) (40 x 1)
            else: u_stat =  -self.K @ self.x + controlled_noise
            #radius_u = 1e15  # Define an appropriate radius

            # if self.flag: u_stat = project_l2_ball(controlled_noise, radius_u) # previously had dependency on u_{t-1]}
            # else: u_stat = project_l2_ball(-self.K @ self.x + controlled_noise, radius_u)
                
            ####################################
            # WORKING VERSION:
            # if self.nl:
            #     x_nl = self.nonlinear_dynamics(self.x)
            #     x_next = self.A @ x_nl + self.B @ u_stat + w_t_history[:, 0:1]
            #     new_w_t = x_next - self.A @ x_nl - self.B @ u_stat
            # else:
            #     x_next = self.A @ self.x + self.B @ u_stat + w_t_history[:, 0:1]
            #     new_w_t = x_next - self.A @ self.x - self.B @ u_stat
            ####################################

            ####################################
            # TESTING VERSION:

            
            # STEP 2: Update x_next to be the equation (2) from the SPECTRAL STATE MODELS paper as follows:
            # 1. Add computed u_orig to the history u_history (has to initialize the history given that u_orig is n x 1):
           

            # Udpate the history and add the new item
            self.u_t_history       = self.u_t_history.roll(1,-1)
            self.u_t_history[:, 0] = u_stat.detach()[:,0]
            #print(u_orig)
        
            # 2. Get the next x state as follows (initialize new matrix M and filters phi - fixed to amount 20): 
            # x_next = sum (i = 1 to 20) of M_phi_i < (u_orig) * phi_i >

            # M is (40 x 100) @ item in u_history which is in (40 x 1)
            x_next = sum(
                self.M_stu[i] @ (self.u_t_history @ self.phi_stu[:, i:i+1]) for i in range(20)
                #(u_t_history @ self.phi_stu[:, i:i+1]) for i in range(20)
            )
            #print("TEST", x_next.shape)

            ####################################
            # Below need to pass x_next to be output from the STU rather than LDS signal
            # x_next = self.A @ self.x + self.B @ u + w_t_history[t % self.h]  # get the next state (check the % self.h)
            # print(f"x_next shape: {x_next.shape}")  # (n,1)


            new_w_t = x_next - self.A @ self.x - self.B @ u_stat        # compute the new perturbation
            # radius_w = 1e15  # Define an appropriate radius
            # new_w_t = project_l2_ball((x_next - self.A @ self.x - self.B @ u_stat), radius_w)
            #print(f"new_w_t shape: {new_w_t.shape}")  # (n,1)
            ####################################
            
            # x_next = self.A @ self.x + self.B @ u_stat + w_t_history[:, 0:1]  # get the next state (check the % self.h)
            #x_next = self.A @ self.x + self.B @ u + w_tilde  # get the next state (check the % self.h)
            #print(f"x_next - A @ x: {x_next - self.A @ self.x}")  # (n,1) --- 0
            #print(f"B @ u: {self.B @ u}")  # (n,1) --- 0 

            # new_w_t = x_next - self.A @ self.x - self.B @ u_stat        # compute the new perturbation
            #print(f"new_w_t: {new_w_t}")  # new_w_t shape is (n,1)

            # Update w_t history
            self.w_t_history = self.w_t_history.roll(1,-1)
            self.w_t_history[:, 0] = new_w_t.detach()[:,0]

            # Compute cost
            # Compute cost and construct loss:
            cost = self.x.T @ self.Q @ self.x + u_stat.T @ self.R @ u_stat # LQR cost? ----- convex/quadratic? (diff for GPC)
            #print(f"cost shape: {cost.shape}")  # (1,1)


            optimizer.zero_grad()    # Reset gradients
            cost.backward()

            with torch.no_grad():
                self.M -= self.eta * self.M.grad  # Gradient step
                self.M.grad.zero_()  # Manually reset gradients
            #print("M gradient", torch.norm(self.M.grad))
           # optimizer.step()        # gradient step
            #print("NORM OF M", torch.norm(self.M))
            
            # with torch.no_grad(): 
            #     self.M *= 0.4  # Scale M by 1/2
            #     self.M_stu *= 0.4
                 
            self.losses[t] = cost.item()  # Store cost in the tensor

            # Projection step if ||M||_F >= RM
            # frob_norm = torch.norm(self.M, p='fro')
            # if frob_norm >= self.RM:
            #     self.M.data = (self.RM / frob_norm) * self.M.data  # Frobenius norm projection

            # Update state
            self.x = x_next.detach()
