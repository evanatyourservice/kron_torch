#%%
import matplotlib.pyplot as plt
import torch
import random

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
m1, m2 = 0, 0
avg_energies = []
for i in range(num_iterations):
    g = A @ torch.randn(N, 1) + b  # gradient
    m1 = beta * m1 + (1 - beta) * g  # momentums in adam; I do not do /(1-beta**(i+1))
    m2 = beta * m2 + (1 - beta) * g * g

    if random.random() < precond_update_prob:
        update_precond_newton_math_(
            Q,
            torch.randn_like(m1),
            m1 + torch.sqrt(torch.clamp(m2 - m1 * m1, 0)) * torch.randn_like(m1),
            0.1 * (1 - i / num_iterations),
            0.0,
        )

    to_precondition = m1
    avg_energy = torch.mean((Q.T @ Q @ to_precondition) ** 2)
    avg_energies.append(avg_energy.item())

print(torch.linalg.cond(Q))
#%%
plt.semilogy(avg_energies)
plt.show()

plt.imshow(Q.T @ Q @ C @ Q.T @ Q)
plt.show()
# %%
