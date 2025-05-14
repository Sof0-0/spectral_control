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

    
#     D = np.diag(np.random.uniform(0.79, 0.81, n))  # Diagonal with values close to 1
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


def plot_loss_comparison(controllers, labels, title, window_size=10, save_path=None):
    """
    Plot the moving average of multiple controllers' losses over time.

    Parameters:
    - controllers: list of controller objects with .losses attribute
    - labels: list of labels for the legend
    - title: title for the plot
    - window_size: window size for computing moving average
    """
    plt.figure(figsize=(10, 6))
    
    for controller, label in zip(controllers, labels):
        losses = controller.losses.cpu().numpy()
        T = len(losses)
        
        if T < window_size:
            raise ValueError("Window size should be smaller than the length of the loss sequence.")
        
        # Compute moving average using convolution
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size - 1, T), moving_avg, label=label)
    
    plt.xlabel('Time Step')
    plt.ylabel(f'{window_size}-Step Average Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save

    plt.show()


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

# def plot_loss_sliding(controller, title, window_size):
#     """
#     Plot the moving average of the controller's loss over time.

#     Parameters:
#     - controller: controller object with .losses attribute
#     - title: title for the plot
#     - window_size: window size for computing moving average
#     """
#     losses = controller.losses.cpu().numpy()
#     T = len(losses)

#     if T < window_size:
#         raise ValueError("Window size should be smaller than the length of the loss sequence.")

#     # Compute moving average using convolution
#     moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

#     plt.figure(figsize=(10, 6))
#     plt.plot(np.arange(window_size - 1, T), moving_avg)
#     plt.xlabel('Time Step')
#     plt.ylabel(f'{window_size}-Step Average Loss')
#     plt.title(title)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def plot_loss_sliding(controllers, title, window_size, labels=None, colors=None, figsize=(12, 7), alpha=0.8, save_path=None):

    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert single controller to list for uniform processing
    if not isinstance(controllers, list):
        controllers = [controllers]
    
    # Set default labels if not provided
    if labels is None:
        labels = [f"Controller {i+1}" for i in range(len(controllers))]
    elif not isinstance(labels, list):
        labels = [labels]  # Convert single label to list
    
    # Ensure labels and controllers have the same length
    if len(labels) != len(controllers):
        raise ValueError("Number of labels must match number of controllers")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, controller in enumerate(controllers):
        losses = controller.losses.cpu().numpy()
        T = len(losses)
        
        if T < window_size:
            print(f"Warning: Window size ({window_size}) larger than loss sequence length ({T}) for {labels[i]}. Skipping.")
            continue
        
        # Compute moving average using convolution
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        
        # Plot with color if specified
        if colors and i < len(colors):
            ax.plot(np.arange(window_size - 1, T), moving_avg, label=labels[i], alpha=alpha, color=colors[i])
        else:
            ax.plot(np.arange(window_size - 1, T), moving_avg, label=labels[i], alpha=alpha)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'{window_size}-Step Average Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save
    
    plt.tight_layout()
    
    return fig, ax

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

def run_multiple_runs(controller_class, num_runs=10, T=100, seed_base=0, **controller_kwargs):
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


def run_multiple_controllers(controller_configs, num_runs=10, seed_base=0):

    results = {}
    
    # Run each controller multiple times with its specific parameters
    for config in controller_configs:
        controller_class = config['class']
        name = config['name']
        params = config.get('params', {})
        
        print(f"Running {name}...")
        
        # Get the time steps T from params if specified
        T = params.get('T', 500)
        
        all_losses = []
        for i in range(num_runs):
            torch.manual_seed(seed_base + i)
            np.random.seed(seed_base + i)
            
            # Create controller with its specific parameters
            controller = controller_class(**params)
            controller.run()
            all_losses.append(controller.losses.cpu().numpy())
            
        all_losses = np.stack(all_losses)  # Shape: [num_runs, T]
        results[name] = all_losses
    
    return results


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
        #print("LOSS MATRIX", loss_matrix[i])

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

def plot_multiple_controllers(results_dict, title="Comparison of Controllers", colors=None, save_path=None):

    plt.figure(figsize=(12, 7))
    
    if colors is None:
        # Default color cycle
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (name, loss_matrix) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        num_runs, T = loss_matrix.shape
        timesteps = np.arange(T)
        
        # Plot individual runs with low alpha
        for j in range(num_runs):
            plt.plot(timesteps, loss_matrix[j], color=color, alpha=0.05, linewidth=0.5)
        
        # Plot mean with bold line
        mean_loss = loss_matrix.mean(axis=0)
        plt.plot(timesteps, mean_loss, color=color, linewidth=2.5, label=f'{name} (Mean)')
    
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multiple_sliding_losses(results_dict, window_size=50, title="Sliding Window Loss Comparison", colors=None, save_path=None):
    """
    Plot sliding window average losses from multiple controllers on the same graph.
    
    Parameters:
    results_dict (dict): Dictionary mapping controller names to loss matrices (shape [num_runs, T])
    window_size (int): Window size for computing moving averages
    title (str): Plot title
    colors (list): Optional list of colors for each controller
    """
    plt.figure(figsize=(12, 7))
    
    if colors is None:
        # Default color cycle
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (name, loss_matrix) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        
        # Compute mean loss across all runs for this controller
        mean_loss = loss_matrix.mean(axis=0)
        T = len(mean_loss)
        
        if T < window_size:
            print(f"Warning: Window size {window_size} is larger than time steps {T} for {name}. Skipping.")
            continue
        
        # Compute moving average of mean loss
        moving_avg = np.convolve(mean_loss, np.ones(window_size)/window_size, mode='valid')
        
        # Plot the sliding window average
        plt.plot(
            np.arange(window_size - 1, T), 
            moving_avg, 
            color=color, 
            linewidth=2.5, 
            label=f'{name}'
        )
    
    plt.xlabel('Time Step')
    plt.ylabel(f'{window_size}-Step Average Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save

    plt.tight_layout()
    plt.show()

def run_multiple_models_with_params(controller_configs, num_runs=10, seed_base=0):
    """
    Run multiple different controllers with their own specific parameters
    
    Parameters:
    controller_configs (list): List of tuples (name, controller_class, params_dict)
    num_runs (int): Number of runs per controller
    seed_base (int): Base seed for reproducibility
    
    Returns:
    dict: Dictionary of {name: loss_matrix} where loss_matrix has shape [num_runs, T]
    """
    all_results = {}
    
    for name, controller_class, params in controller_configs:
        print(f"Running {name}...")
        all_losses = []
        
        for i in range(num_runs):
            torch.manual_seed(seed_base + i)
            np.random.seed(seed_base + i)
            
            # Create a copy of the parameters to avoid modifying the original
            run_params = params.copy()
            
            # Extract T from params if available
            T = run_params.get('T', 100)  # Default to 100 if not specified
            
            controller = controller_class(**run_params)
            controller.run()
            all_losses.append(controller.losses.cpu().numpy())
            
        all_results[name] = np.stack(all_losses)  # Shape: [num_runs, T]
    
    return all_results

def plot_loss_comparison_with_ci(controllers, labels, title, window_size=10, confidence=0.95, save_path=None):
    """
    Plot the moving average of multiple controllers' losses over time with confidence intervals.
    
    Parameters:
    - controllers: list of controller objects with .losses attribute storing a tensor of shape [num_trials, T]
                  where each row is a trial and each column is a time step
    - labels: list of labels for the legend
    - title: title for the plot
    - window_size: window size for computing moving average
    - confidence: confidence level for the interval (default: 0.95 for 95% CI)
    - save_path: path to save the figure (optional)
    """
    
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(controllers)))
    
    for i, (controller, label) in enumerate(zip(controllers, labels)):
        # Get all trials data
        # We assume losses is of shape [num_trials, T]
        # If it's not, reshape it accordingly
        if len(controller.losses.shape) == 1:
            # If losses is just a vector, we need to reshape it to [1, T]
            losses_np = controller.losses.cpu().numpy().reshape(1, -1)
        else:
            losses_np = controller.losses.cpu().numpy()
        
        num_trials, T = losses_np.shape
        
        if T < window_size:
            raise ValueError("Window size should be smaller than the length of the loss sequence.")
        
        # Apply moving average to each trial
        smoothed_losses = np.zeros((num_trials, T - window_size + 1))
        for j in range(num_trials):
            smoothed_losses[j] = np.convolve(losses_np[j], np.ones(window_size)/window_size, mode='valid')
        
        # Calculate mean and confidence intervals
        mean_losses = np.mean(smoothed_losses, axis=0)
        
        # Calculate the confidence interval
        if num_trials > 1:
            # Use t-distribution when we have multiple trials
            t_value = stats.t.ppf((1 + confidence) / 2, num_trials - 1)
            std_err = stats.sem(smoothed_losses, axis=0)
            margin_error = t_value * std_err
            lower_bound = mean_losses - margin_error
            upper_bound = mean_losses + margin_error
        else:
            # If only one trial, we can't compute CI
            lower_bound = mean_losses
            upper_bound = mean_losses
        
        # Plotting
        x_values = np.arange(window_size - 1, T)
        plt.plot(x_values, mean_losses, label=label, color=colors[i], linewidth=2)
        
        if num_trials > 1:
            plt.fill_between(x_values, lower_bound, upper_bound, color=colors[i], alpha=0.2)
    
    plt.xlabel('Time Step')
    plt.ylabel(f'{window_size}-Step Average Loss')
    plt.title(f"{title}\n(with {confidence*100:.0f}% Confidence Intervals, {num_trials} trials)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return mean_losses, lower_bound, upper_bound




