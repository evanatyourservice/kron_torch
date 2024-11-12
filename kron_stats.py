#%%
import matplotlib.pyplot as plt
import torch
import random

class GradStats:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.count = 0

    def update(self, grad):
        count = self.count
        new_count = count + 1
        delta = grad - self.mean
        self.mean += delta / new_count
        delta2 = grad - self.mean
        self.var += delta * delta2
        self.count = new_count

    def get_stats_and_reset(self):
        if self.count == 0:
            return None, None
        std = torch.sqrt(self.var / self.count)
        mean = self.mean
        self.mean = 0
        self.var = 0
        self.count = 0
        return mean, std

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
beta = 0.5
Q = torch.eye(N)
stats = GradStats()
avg_energies = []

for i in range(num_iterations):
    g = A @ torch.randn(N, 1) + b  # gradient
    
    # Update statistics when not updating preconditioner
    if random.random() >= precond_update_prob:
        stats.update(g)
    
    if random.random() < precond_update_prob:
        mean, std = stats.get_stats_and_reset() if stats.count > 0 else (g, torch.zeros_like(g))
        fake_grad = mean + std * torch.randn_like(g) if mean is not None else g
        
        update_precond_newton_math_(
            Q,
            torch.randn_like(fake_grad),
            fake_grad,
            0.1 * (1 - i / num_iterations),
            0.0,
        )

    to_precondition = g
    avg_energy = torch.mean((Q.T @ Q @ to_precondition) ** 2)
    avg_energies.append(avg_energy.item())

print(torch.linalg.cond(Q))
#%%
plt.semilogy(avg_energies)
plt.show()

plt.imshow(Q.T @ Q @ C @ Q.T @ Q)
plt.show()
# %%
