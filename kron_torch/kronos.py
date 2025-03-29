"""Kronos (Kron One-Sided) optimizer for PyTorch."""

from typing import Optional, Any
import numpy as np
import torch
from torch import Tensor


class ProbScheduler:
    """Scheduler for annealing preconditioner update probability.
    
    Implements an exponential anneal with a flat start.
    """
    
    def __init__(self, max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
        self.max_prob = torch.tensor(max_prob, dtype=torch.float32)
        self.min_prob = torch.tensor(min_prob, dtype=torch.float32)
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self.flat_start = torch.tensor(flat_start, dtype=torch.float32)
        self._compiled = False
        try:
            self._compiled_schedule = torch.compile(self._schedule_fn)
            self._compiled = True
        except Exception:
            pass
    
    def _schedule_fn(self, n):
        """Exponential anneal with flat start."""
        prob = self.max_prob * torch.exp(-self.decay * (n - self.flat_start))
        prob.clamp_(min=self.min_prob, max=self.max_prob)
        return prob
    
    def __call__(self, n):
        """Call schedule function, using compiled version if available."""
        if self._compiled:
            return self._compiled_schedule(n)
        else:
            return self._schedule_fn(n)
    
    def __reduce__(self):
        """Enable proper pickling by serializing only the parameters."""
        return (self.__class__, (
            self.max_prob.item(),
            self.min_prob.item(),
            self.decay.item(),
            self.flat_start.item()
        ))


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """
    return ProbScheduler(max_prob, min_prob, decay, flat_start)


class Kronos(torch.optim.Optimizer):
    """PSGD Kronos (Kron One-Sided) optimizer.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        b1: Momentum
        weight_decay: Weight decay
        preconditioner_update_probability: Prob of updating preconditioner (default: anneals 1.0->0.03 by 4000 steps)
        precond_lr: Preconditioner learning rate (default: 0.1)
        dtype: Data type for params/grads
    """

    def __init__(
        self,
        params: list[torch.Tensor | dict[str, Any]],
        lr: float = 0.0003,
        b1: float = 0.9,
        weight_decay: float = 0.0,
        preconditioner_update_probability: Optional[ProbScheduler] = None,
        precond_lr: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        self.current_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        params = [*params]
        kron_param_groups = []
        adam_param_groups = []
        if isinstance(params[0], dict):
            for group in params:
                kron_params = []
                adam_params = []
                for p in group["params"]:
                    if p.ndim < 2 or max(p.shape) == np.prod(p.shape):
                        adam_params.append(p)
                    else:
                        kron_params.append(p)
                
                if kron_params:
                    kron_param_groups.append({
                        "params": kron_params,
                        **{k: v for k, v in group.items() if k != "params"}
                    })
                if adam_params:
                    adam_param_groups.append({
                        "params": adam_params,
                        **{k: v for k, v in group.items() if k != "params"}
                    })
        else:
            kron_params = []
            adam_params = []
            for p in params:
                if p.ndim < 2 or max(p.shape) == np.prod(p.shape):
                    adam_params.append(p)
                else:
                    kron_params.append(p)
            
            if kron_params:
                kron_param_groups.append({"params": kron_params})
            if adam_params:
                adam_param_groups.append({"params": adam_params})

        if adam_param_groups:
            self._adam = torch.optim.Adam(
                adam_param_groups,
                lr=lr * 3.0,
                betas=(0.9, 0.99),
                fused=True,
            )
        else:
            self._adam = None

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            precond_lr=precond_lr,
            dtype=dtype,
        )
        
        super().__init__(kron_param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device=self.current_device)
        self._prob_step = torch.tensor(0, dtype=torch.int32)
        self._update_counter = torch.tensor(0, dtype=torch.int32)
        self.dtype = dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_prob = self.defaults["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(self._prob_step.to(dtype=torch.float32))
            self._prob_step += 1
        self._update_counter += 1
        do_update = self._update_counter >= 1 / update_prob
        if do_update:
            self._update_counter = torch.tensor(0, dtype=torch.int32)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.to(self.dtype)
                state = self.state[p]

                if g.dim() > 2:
                    if "merged_shape" not in state:
                        shape1 = [np.prod(g.shape[:-1]), g.shape[-1]]
                        shape2 = [g.shape[0], np.prod(g.shape[1:])]
                        shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                        state["merged_shape"] = shape
                    g = g.view(*state["merged_shape"])

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["Q"] = torch.eye(min(g.shape), dtype=self.dtype, device=g.device)
                    state["step"] = torch.tensor(0, dtype=torch.int32, device=self.current_device)
                state["step"] += 1

                g = _update_momentum(
                    state["momentum_buffer"],
                    g,
                    torch.tensor(group["b1"], dtype=self.dtype, device=self.current_device),
                    state["step"]
                )

                if do_update:
                    state["Q"] = _update_preconditioner(
                        g,
                        state["Q"],
                        torch.tensor(group["precond_lr"], dtype=self.dtype, device=self.current_device),
                    )
                
                g = _precondition(g, state["Q"])
                
                _apply_updates(
                    g, 
                    p, 
                    p.shape, 
                    torch.tensor(group["weight_decay"], dtype=self.dtype, device=self.current_device), 
                    torch.tensor(group["lr"], dtype=self.dtype, device=self.current_device)
                )

        # adam for 1D params
        if self._adam is not None:
            self._adam.step()
        
        return loss

    def state_dict(self):
        """Return the state of the optimizer as a dict."""
        state_dict = super().state_dict()
        if self._adam is not None:
            state_dict['adam_state'] = self._adam.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the optimizer state."""
        if 'adam_state' in state_dict:
            adam_state = state_dict.pop('adam_state')
            if self._adam is not None:
                self._adam.load_state_dict(adam_state)
        super().load_state_dict(state_dict)


@torch.compile
def _update_momentum(momentum_buffer, grad, beta, step):
    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
    return momentum_buffer.div(1 - beta**step)


def _lb(A: Tensor, max_abs: Tensor) -> Tensor:
    """Cheap lower bound for the spectral norm ||A||₂ using power-iteration-like steps.


    Summary:

    Provides a cheap estimate of the largest singular value (spectral norm) of `A` using power-iteration-like steps.
    Used within `_update_preconditioner` to adaptively normalize the step size of the preconditioner update.


    Implementation Steps (analogous to power iteration on `A A^T`):

    1. Normalize `A` by its max absolute element to prevent under/overflow
    2. Find the column of `A` with the largest L2 norm (likely rich in the dominant right singular vector) and use its transpose as the initial vector `x`.
    3. Compute `y = x @ A` (analogous to one step of applying `A^T`).
    4. Normalize `y`.
    5. Compute `z = y @ A.T` (analogous to one step of applying `A`).
    6. The L2 norm `||z||` approximates the largest singular value of the normalized `A`.
    7. Rescale the result by `max_abs`.


    Note:

    Numerical results on random matrices with a wide range of distributions and sizes suggest
    `norm(A) <= sqrt(2) * norm_lower_bound(A)`, which looks to be a very tight lower bound.
    """
    # Normalize `A` by its max absolute element to prevent under/overflow
    A /= max_abs

    # Calculate squared L2 norm of each column
    a0 = torch.einsum("ij,ij->j", A, A)

    # Select column with maximum norm, transpose it (implicitly), use as initial vector `x` (shape [num_rows])
    # (rich in the dominant right singular vector)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()

    # Compute `y = x^T @ A` (shape [num_cols])
    # This step emphasizes components related to the dominant left singular vector.
    y = torch.einsum("i,ij->j", x, A)

    # Avoiding expression `norm(x * A^T) / norm(x)` as `x * A^T` could under/overflow...
    # Normalize the intermediate vector
    y /= y.norm()

    # Compute `z = y @ A.T` (shape [num_rows])
    # This step emphasizes components related to the dominant right singular vector.
    z = torch.einsum("j,kj->k", y, A)

    # Calculate the L2 norm of `z`.
    # This approximates the largest singular value (spectral norm) of the normalized matrix `A`.
    norm_estimate = z.norm()

    # Rescale the estimate by max_abs to get the estimate for the original matrix A
    norm_estimate *= max_abs
    return norm_estimate


@torch.compile
def _update_preconditioner(G: Tensor, Q: Tensor, lr: Tensor) -> Tensor:
    """Updates the upper triangular preconditioner factor Q using relative gradient descent.


    Gist: Whitening Preconditioner

    This function maintains a Cholesky-like factor `Q` of the preconditioner (`P = Q^T Q`), and updates it by checking
    how well it whitens gradients versus how much it amplifies noise.


    Summary: Technical Details

    Criterion 3:
    This function implements the core preconditioner update logic based on Criterion 3 from Li's PSGD papers 
    (e.g., Eq. 31 in Li 2015, Eq. 2 in Li 2024). The goal is to find an upper triangular matrix `Q` such that 
    the implicit symmetric positive definite preconditioner `P = Q^T Q` minimizes the criterion:
        `E[ g^T P g + v^T P^{-1} v ]`
    where `g` is the stochastic gradient (`G`) and `v` is auxiliary Gaussian noise (`V`). `Q` is analogous 
    to the upper Cholesky factor of `P`.

    Intuitive Meaning:
    Intuitively, this criterion encourages `P` to transform the gradients such that their covariance 
    `P E[gg^T] P` approaches the identity matrix `I = E[vv^T]`, effectively "whitening" the gradients, 
    while balancing the amplification of noise introduced by the `P^{-1}` term. `P` is driven to normalize 
    curvature and inherently damp stochastic noise without requiring explicit damping factors or Hessian 
    definiteness assumptions, making it suitable for non-convex optimization (Prop. 3 in Li 2015; 
    Sec. III.A/C in Li 2018, Prop. 5 in Li 2024).

    Update Method:
    The update follows a relative (natural) gradient descent step on the Lie group of invertible upper 
    triangular matrices (Sec. 4.1 in Li 2018; Cardoso & Laheld, 1996). The update form is 
    (Eq. 43 in Li 2015; Eq. 19 in Li 2018):
        `Q_new = Q_old - step_size * Grad_relative @ Q_old`
    where `Grad_relative` is the gradient restricted to the upper triangular structure (Lie algebra), 
    calculated as `torch.triu(AhA - BBh)`.


    Implementation Steps:
    
    1.  **Transform Noise:** See how the inverse of the current shaping factor (`Q^{-1}`) affects random noise (`V`). (Calculates `Bh = V @ Q^{-1}`)
        This helps us understand how our current preconditioner would amplify random noise in the system.
    
    2.  **Estimate Noise Effect:** Calculate how much the inverse shaping factor amplifies this random noise. (Calculates `BBh = Bh^T @ Bh`)
        `BBh` represents the noise amplification effect of our preconditioner. Ideally this would be close to the identity matrix.
    
    3.  **Transform Gradient:** Apply the transpose of the current shaping factor (`Q^T`) to the gradient (`G`). (Calculates `A = G @ Q^T`)
        This transforms our current gradient using the preconditioner to see how it reshapes the gradient space.
    
    4.  **Estimate Gradient Effect:** Calculate the "energy" or spread of this transformed gradient. (Calculates `AhA = A^T @ A`)
        This represents how well the preconditioner is currently whitening the gradient. If whitening is perfect, `AhA` would approximate the identity matrix.
    
    5.  **Find Adjustment Direction:** Compare the gradient's spread (`AhA`) with the noise amplification (`BBh`). The difference (`AhA - BBh`) tells us how `Q` should change to better balance whitening the gradient vs. amplifying noise.
        When `AhA - BBh` approaches zero, `P` is approaching optimal whitening where `P E[gg^T] P ≈ I`.
    
    6.  **Refine Direction:** Keep only the essential (upper triangular) part of the adjustment direction, as required by the underlying mathematical structure (Lie group). (Calculates `Grad = torch.triu(AhA - BBh)`)
    
    7.  **Calculate Safe Step Size:** Estimate the "strength" of the required adjustment (`AhA + BBh`) to determine how big a step we can safely take (`normalizer`).
        This ensures we take an appropriate-sized step, like an adaptive learning rate that prevents too large or too small updates.
    
    8.  **Update Preconditioner Factor:** Apply the refined adjustment direction (`Grad`) to the current shaping factor (`Q`), scaled by the learning rate (`lr`) and the safe step size (`1 / normalizer`). (Updates `Q -= step_size * Grad @ Q`)
        This drives `P = Q^T Q` toward transforming the gradient covariance to approximate the identity matrix, effectively whitening the gradients.


    Args:
        G: The current (potentially momentum-adjusted) gradient tensor, reshaped
           to 2D. Represents the stochastic gradient `g` (or `δĝ`).
        Q: The current upper triangular preconditioner factor.
        lr: The learning rate for updating the preconditioner `Q`.

    Returns:
        The updated upper triangular preconditioner factor `Q`.
    """
    m, n = G.shape
    if m < n:
        G = G.T
        m, n = G.shape

    # Generate V ~ N(0,I) as the auxiliary noise variable `v` from Criterion 3.
    # `V` helps estimate the gradient contribution from the `P^{-1}` term (like a Hutchinson trace estimator)
    # and balances gradient scaling vs. noise suppression.
    V = torch.randn_like(G, dtype=torch.float32)

    # Add small scaled noise for numerical stability if G is very small.
    # This prevents potential issues when G approaches zero.
    eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=G.dtype, device=G.device).sqrt()
    G += eps * G.abs().mean() * V.to(dtype=G.dtype)

    # --- Calculate terms related to the relative gradient of Criterion 3 ---

    # 1. Calculate Bh = V @ Q^{-1} using triangular solve (numerically stable).
    #    Represents the noise `v` transformed by the inverse factor `Q^{-1}`.
    Bh = torch.linalg.solve_triangular(Q.float(), V, upper=True, left=False).to(dtype=G.dtype)
    # This helps us understand how our current preconditioner would amplify random noise in the system.

    # 2. Calculate BBh = Bh^T @ Bh = Q^{-T} V^T V Q^{-1}.
    #    Estimates the gradient contribution from the `v^T P^{-1} v` term in the criterion.
    #    Approximates `m * Q^{-T} Q^{-1}` (scaled empirical covariance of transformed noise).
    BBh = Bh.T @ Bh
    # `BBh` represents the noise amplification effect of our preconditioner. Ideally this would be close to the identity matrix.

    # 3. Calculate A = G @ Q^T, intermediate transformation of the gradient.
    A = G @ Q.T
    # This transforms our current gradient using the preconditioner to see how it reshapes the gradient space.

    # 4. Calculate AhA = A^T @ A = Q G^T G Q^T.
    #    Estimates the gradient contribution from the `g^T P g` term in the criterion.
    #    Approximates `Q E[gg^T] Q^T` (empirical covariance of gradient transformed by Q).
    AhA = A.T @ A
    # This represents how well the preconditioner is currently whitening the gradient.
    # If whitening is perfect, `AhA` would approximate the identity matrix.

    # --- Compute Relative Gradient and Perform Update Step ---

    # 5. Calculate the raw relative gradient direction: `AhA - BBh`.
    #    This difference drives the update towards making `P E[gg^T] P = I` to satisfy Criterion 3 (structure matches Eq. 36 in Li 2024).
    #    When `AhA - BBh` approaches zero, `P` is approaching optimal whitening where `P E[gg^T] P ≈ I`.
    # 6. Project the raw difference onto the Lie algebra of upper triangular matrices using `torch.triu`.
    #    This yields the relative gradient `Grad` for the Lie group update (Eq. 42 in Li 2015; Eq. 42 in Li 2024).
    Grad = torch.triu(AhA - BBh)  # Grad_relative

    # 7. Calculate an adaptive step size `normalizer`. This factor scales the learning rate.
    #    It's based on a cheap spectral norm estimate (`_lb`) of `AhA + BBh`, which relates
    #    to the local Hessian/curvature of the fitting criterion (Eq. 34-35 in Li 2024).
    #    This relates to the normalization in Eq. 37-38 in Li 2024, which involves terms
    #    like `||Qtht||² + ||Q⁻ᵀvt||²`, providing stability and a practical heuristic
    #    to scale the update based on local properties.
    A_norm = AhA + BBh
    max_abs = A_norm.norm(float("inf"))
    normalizer = torch.where(max_abs > 0, _lb(A_norm, max_abs), max_abs)
    # This ensures we take an appropriate-sized step, like an adaptive learning rate that prevents too large or too small updates.

    # 8. Perform the relative gradient descent step on the Lie group (Eq. 43 in Li 2015):
    #    `Q_{t+1} = Q_t - (lr / normalizer) * Grad @ Q_t`
    Q -= lr / normalizer * Grad @ Q
    # This drives `P = Q^T Q` toward transforming the gradient covariance to approximate the identity matrix, effectively whitening the gradients.

    return Q


@torch.compile
def _precondition(G: Tensor, Q: Tensor):
    m, n = G.shape
    if m < n:
        return torch.einsum("ji,jk,kl->il", Q, Q, G)
    else:
        return torch.einsum("ij,kj,kl->il", G, Q, Q)


@torch.compile
def _apply_updates(g: Tensor, p: Tensor, original_shape: tuple, weight_decay: float, lr: float) -> None:
    g = g / (g.square().mean().sqrt() + 1e-12)
    g = g.div(2.0).tanh_().mul_(2.0)
    g = g.view(original_shape)
    if weight_decay > 0:
        g.add_(p, alpha=weight_decay)
    p.add_(g.to(p.dtype), alpha=-lr)
