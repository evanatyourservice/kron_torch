#%%
import matplotlib.pyplot as plt
import torch
import random

class AdamStats:
    def __init__(self, beta1=0.5, beta2=0.8, bias_correction=False):
        self.mean = 0
        self.var = 0
        self.count = 0
        self.m1 = 0  # First moment (mean)
        self.m2 = 0  # Second moment
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0  # Add timestep counter for bias correction
        self.bias_correction = bias_correction

    def update(self, grad):
        # Update running statistics
        count = self.count
        new_count = count + 1
        delta = grad - self.mean
        self.mean += delta / new_count
        delta2 = grad - self.mean
        self.var += delta * delta2
        self.count = new_count

        # Update Adam moments with timestep counter
        self.t += 1
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * grad
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * grad * grad

    def get_stats_and_reset(self):
        if self.count == 0:
            return None, None, None, None
        std = torch.sqrt(self.var / self.count)
        mean = self.mean
        
        # Apply bias corrections only if enabled
        if self.bias_correction:
            m1_corrected = self.m1 / (1 - self.beta1**self.t)
            m2_corrected = self.m2 / (1 - self.beta2**self.t)
        else:
            m1_corrected = self.m1
            m2_corrected = self.m2
        
        # Reset statistics
        self.mean = 0
        self.var = 0
        self.count = 0
        # Note: we don't reset t as it needs to continue for proper bias correction
        
        return mean, std, m1_corrected, m2_corrected

def update_precond_newton_math_(Q, v, h, step, tiny):
    """Update the preconditioner P = Q'*Q with (v, h), adjusted for EMA gradients."""
    a = Q.mm(h)
    b = torch.linalg.solve_triangular(Q.t(), v, upper=False)
    grad = torch.triu(a @ a.t() - b @ b.t())
    mu = step / (torch.sum(a**2 + b**2) + tiny)
    Q.sub_(mu * grad @ Q)

def run_experiment(beta1, beta2=0.9, num_iterations=10000):
    Q = torch.eye(N)
    stats = AdamStats(beta1=beta1, beta2=beta2,bias_correction=False)
    avg_energies = []

    for i in range(num_iterations):
        # Use Cauchy distribution like in second example
        g = A @ torch.distributions.Cauchy(0, 1).sample((N, 1)) + b
        g = g / torch.linalg.norm(g)  # normalize gradient
        
        stats.update(g)
        mean, std, m1, m2 = stats.get_stats_and_reset()
        
        if mean is not None:
            update_precond_newton_math_(
                Q,
                torch.randn_like(m1),
                m1 + alpha * torch.sqrt(torch.clamp(m2 - m1 * m1, 0)) * torch.randn_like(m1),
                0.1 * (1 - i / num_iterations),
                0.0,
            )

        avg_energy = torch.mean((Q.T @ Q @ g) ** 2)
        avg_energies.append(avg_energy.item())
    
    return Q, avg_energies

# Setup experiment
N = 100
# Define matrices A, b, and C
A = torch.randn(N, N)  # Random matrix for demonstration
A = A @ A.T  # Make it symmetric positive definite
b = torch.randn(N, 1)  # Random vector
C = torch.eye(N)  # Identity matrix for covariance

num_iterations = 10000
betas = [0.0, 0.5, 0.9, 0.95]
alpha = 1

# Setup plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Run experiments for different betas
for idx, beta in enumerate(betas):
    Q, avg_energies = run_experiment(beta)
    
    # Plot energy
    axes[idx].semilogy(avg_energies)
    axes[idx].set_title(f"β = {beta}")
    axes[idx].set_xlabel("Iterations")
    axes[idx].set_ylabel("Energy")
    
    print(f"Condition number for β={beta}: {torch.linalg.cond(Q)}")
    print(f"Trace of Q: {torch.trace(Q)}")

plt.tight_layout()
plt.show()

# Plot covariance matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, beta in enumerate(betas):
    axes[idx].imshow(Q.T @ Q @ C @ Q.T @ Q)
    axes[idx].set_title(f"Covariance Matrix (β = {beta})")

plt.tight_layout()
plt.show()
# %%
