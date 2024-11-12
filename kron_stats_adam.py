#%%
import matplotlib.pyplot as plt
import torch
import random

class AdamStats:
    def __init__(self, beta=0.5):
        self.mean = 0
        self.var = 0
        self.count = 0
        self.m1 = 0  # First moment (mean)
        self.m2 = 0  # Second moment
        self.beta = beta

    def update(self, grad):
        # Update running statistics
        count = self.count
        new_count = count + 1
        delta = grad - self.mean
        self.mean += delta / new_count
        delta2 = grad - self.mean
        self.var += delta * delta2
        self.count = new_count

        # Update Adam moments
        self.m1 = self.beta * self.m1 + (1 - self.beta) * grad
        self.m2 = self.beta * self.m2 + (1 - self.beta) * grad * grad

    def get_stats_and_reset(self):
        if self.count == 0:
            return None, None, None, None
        std = torch.sqrt(self.var / self.count)
        mean = self.mean
        m1, m2 = self.m1, self.m2
        
        # Reset statistics
        self.mean = 0
        self.var = 0
        self.count = 0
        # Note: we don't reset m1 and m2 as they are running estimates
        
        return mean, std, m1, m2

def update_precond_newton_math_(Q, v, h, step, tiny):
    """Update the preconditioner P = Q'*Q with (v, h), adjusted for EMA gradients."""
    a = Q.mm(h)
    b = torch.linalg.solve_triangular(Q.t(), v, upper=False)
    grad = torch.triu(a @ a.t() - b @ b.t())
    mu = step / (torch.sum(a**2 + b**2) + tiny)
    Q.sub_(mu * grad @ Q)


N = 100
num_iterations = 100000
precond_update_prob = 0.1

A = torch.triu(torch.rand(N, N))
b = torch.randn(N, 1)
C = A @ A.T + b @ b.T  # the true covariance matrix of gradient
beta = 0.9
Q = torch.eye(N)
stats = AdamStats(beta=beta)
avg_energies = []

for i in range(num_iterations):
    g = A @ torch.randn(N, 1) + b  # gradient
    
    # Update statistics when not updating preconditioner
    if random.random() >= precond_update_prob:
        stats.update(g)
    
    if random.random() < precond_update_prob:
        mean, std, m1, m2 = stats.get_stats_and_reset() if stats.count > 0 else (g, torch.zeros_like(g), g, g*g)
        
        # Use Adam-style update with sqrt clamp like in kron_adam_
        h = m1 + torch.sqrt(torch.clamp(m2 - m1 * m1, 0)) * torch.randn_like(g)
        
        update_precond_newton_math_(
            Q,
            torch.randn_like(h),
            h,
            0.1 * (1 - i / num_iterations),
            0.0,
        )

    to_precondition = g  # or could use m1 if you want to precondition the Adam estimate
    avg_energy = torch.mean((Q.T @ Q @ to_precondition) ** 2)
    avg_energies.append(avg_energy.item())

print(torch.linalg.cond(Q))
#%%
plt.semilogy(avg_energies)
plt.show()

plt.imshow(Q.T @ Q @ C @ Q.T @ Q)
plt.show()
# %%
