import numpy as np
import torch
import os
import scipy.linalg as scp_lin
import matplotlib.pyplot as plt
import torch.nn.functional as F

def project_l2_ball(x, radius):
    norm_x = torch.norm(x, p=2)
    if norm_x > radius:
        return x * (radius / norm_x)
    return x

# def make_diagonalizable_matrix(n):
#     """ Generate a diagonalizable matrix A. """

#     # Negative eigenvalues:
#     #signs = np.random.choice([-1, 1], size=n) # comment out signs if only testing for positive
#     #D = np.diag(signs * np.random.uniform(0.98, 1.00, n))  # Diagonal with values close to 1

    
#     D = np.diag(np.random.uniform(0.99, 1.01, n))  # Diagonal with values close to 1
#     P = np.random.randn(n, n)
#     while np.linalg.cond(P) > n:  # Ensure P is well-conditioned
#         P = np.random.randn(n, n)
#     A = P @ D @ np.linalg.inv(P)  # A = P D P^-1
#     return A

def make_diagonalizable_matrix(n):
    """ Generate a diagonalizable matrix A with all eigenvalues exactly 0.99. """
    
    # Create diagonal matrix with all eigenvalues = 0.99
    D = np.diag(np.ones(n) * 0.98)  # All diagonal elements set to 0.99
    
    # Generate a random well-conditioned matrix for similarity transformation
    P = np.random.randn(n, n)
    while np.linalg.cond(P) > n:  # Ensure P is well-conditioned
        P = np.random.randn(n, n)
    
    # Create similar matrix A with same eigenvalues as D
    A = P @ D @ np.linalg.inv(P)  # A = P D P^-1
    return A

def make_diagonalizable_matrix_complex(n):
    """ Generate a diagonalizable matrix A with some complex eigenvalues. """
    assert n % 2 == 0, "For complex eigenvalues, use an even n to ensure conjugate pairs."
    
    real_vals = np.random.uniform(-1.0, 1.0, n // 2)  # Half real eigenvalues
    complex_parts = np.random.uniform(-1.0, 1.0, n // 2)  # Imaginary components

    # Construct complex eigenvalues as conjugate pairs
    complex_eigenvalues = real_vals + 1j * complex_parts
    all_eigenvalues = np.concatenate([complex_eigenvalues, np.conjugate(complex_eigenvalues)])

    D = np.diag(all_eigenvalues)  # Diagonal matrix with complex eigenvalues
    P = np.random.randn(n, n) + 1j * np.random.randn(n, n)  # Complex change-of-basis matrix

    while np.linalg.cond(P) > n:  # Ensure P is well-conditioned
        P = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    A = P @ D @ np.linalg.inv(P)  # A = P D P^-1
    return A.real  # Return only the real part to ensure a real-valued matrix

def get_hankel(T):
    entries = torch.arange(1, T + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z

def get_hankel_new(m, gamma):
     # New Hankel Matrix for the filters (lives in m by m)
    Z_m = torch.zeros((m, m), dtype=torch.float64)
    for i in range(m):
        for j in range(m):
            Z_m[i, j] = (1 - gamma) ** (i + j - 1) / (i + j - 1) if (i + j - 1) != 0 else 0
    return Z_m


def make_positive_semidefinite(n):
    """ Generate a positive semidefinite matrix Q (or R). """
    M = np.random.randn(n, n)
    return M @ M.T  # Ensures positive semidefiniteness


# LQR to solve K after (A,B) is observed
def lqr(A, B, Q, R):
    S = scp_lin.solve_discrete_are(A, B, Q, R)  # DARE solution
    #S = torch.tensor(S, dtype=torch.float32)
    K = torch.tensor(np.linalg.inv(R + B.T @ S @ B) @ (B.T @ S @ A), dtype=torch.float32)  # linear map from state to control
    
    return K

def plot_loss(model, title, save_path=None):
    plt.plot(model.losses.cpu().numpy())
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title(title)


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save
    
    plt.show()

def compare_losses(models, title="Loss Comparison", save_path=None):
    """
    Plots the loss curves of multiple models.
    
    Parameters:
    - models: list of model objects, each having 'losses' (tensor) and 'name' (string) attributes.
    - title: str, the title of the plot.
    """
    plt.figure(figsize=(8, 5))

    # Define colors and linestyles for distinction
    colors = ["blue", "red", "green", "orange", "purple", "brown", "cyan", "magenta"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for i, model in enumerate(models):
        color = colors[i % len(colors)]  # Cycle through colors
        linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
        plt.plot(model.losses.cpu().numpy(), label=f"{model.name} Loss", linestyle=linestyle, color=color)

    # Labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for better visualization

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save

    
    plt.show()
