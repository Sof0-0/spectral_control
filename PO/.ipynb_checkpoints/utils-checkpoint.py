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

def make_diagonalizable_matrix(n):
    """ Generate a diagonalizable matrix A. """

    # Negative eigenvalues:
    #signs = np.random.choice([-1, 1], size=n) # comment out signs if only testing for positive
    #D = np.diag(signs * np.random.uniform(0.98, 1.00, n))  # Diagonal with values close to 1

    
    D = np.diag(np.random.uniform(0.99, 1.01, n))  # Diagonal with values close to 1
    P = np.random.randn(n, n)
    while np.linalg.cond(P) > n:  # Ensure P is well-conditioned
        P = np.random.randn(n, n)
    A = P @ D @ np.linalg.inv(P)  # A = P D P^-1
    return A

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

def run_multiple_runs(controller_class, num_runs=50, T=100, seed_base=0, **controller_kwargs):
    all_losses = []

    for i in range(num_runs):
        torch.manual_seed(seed_base + i)
        np.random.seed(seed_base + i)

        controller = controller_class(T=T, **controller_kwargs)
        controller.run()
        all_losses.append(controller.losses.cpu().numpy())

    all_losses = np.stack(all_losses)  # Shape: [num_runs, T]
    #print("LOSS", all_losses)
    return all_losses

# def plot_loss_with_95ci(loss_matrix, title):
#     """
#     loss_matrix: np.ndarray of shape (num_runs, T)
#     """
#     mean_loss = loss_matrix.mean(axis=0)
#     std_loss = loss_matrix.std(axis=0)
#     n = loss_matrix.shape[0]

#     ci95 = 1.96 * (std_loss / np.sqrt(n))
#     print("CI", ci95)
#     timesteps = np.arange(loss_matrix.shape[1])

#     plt.figure(figsize=(10, 6))
#     plt.plot(timesteps, mean_loss, label='Mean Loss')
#     plt.fill_between(timesteps, mean_loss - ci95, mean_loss + ci95, alpha=0.3, label='95% CI')

#     plt.xlabel('Time Step')
#     plt.ylabel('Loss')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def plot_runs_with_mean(loss_matrix, title):
    """
    Plot all runs' losses as light lines and overlay the mean loss in bold.
    
    Parameters:
        loss_matrix (np.ndarray): shape (num_runs, T)
        title (str): plot title
    """
    num_runs, T = loss_matrix.shape
    timesteps = np.arange(T)

    plt.figure(figsize=(10, 6))

    # Plot each run's loss in a light color
    for i in range(num_runs):
        plt.plot(timesteps, loss_matrix[i], color='gray', alpha=0.2, linewidth=1)
        print("LOSS MATRIX", loss_matrix[i])

    # Compute and plot mean loss
    mean_loss = loss_matrix.mean(axis=0)
    plt.plot(timesteps, mean_loss, color='blue', linewidth=2.5, label='Mean Loss')

    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


