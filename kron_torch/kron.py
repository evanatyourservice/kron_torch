import string
import random
import numpy as np
import torch


torch._dynamo.config.cache_size_limit = 1_000_000

try:
    torch.backends.opt_einsum.strategy = "dynamic-programming"
except AttributeError:
    # opt_einsum backend is not available, so we'll skip setting the strategy
    pass


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=250
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        n = torch.tensor(n, dtype=torch.float32)
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs
            to have triangular preconditioners.
        memory_save_mode: (string, optional), None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: (bool), whether to send momentum into preconditioner
            update instead of raw gradients.
        mu_dtype (torch.dtype, optional): Dtype of the momentum accumulator.
        precond_dtype (torch.dtype, optional): Dtype of the preconditioner.
        mean_type (str): Type of mean to use for statistics ('arithmetic', 'geometric', 'harmonic', or 'cesaro').
        verbose (bool): Whether to print energy after preconditioning.
    """

    def __init__(
        self,
        params,
        lr=0.01,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        mu_dtype=None,
        precond_dtype=None,
        mean_type='geometric',
        verbose=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter: {b1}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            precond_lr=0.1,  # precond lr hardcoded to 0.1
            precond_init_scale=1.0,  # precond init scale hardcoded to 1.0
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            mean_type=mean_type,
            verbose=verbose,
        )
        super(Kron, self).__init__(params, defaults)

        self._tiny = torch.finfo(torch.bfloat16).tiny
        self._prob_step = 0
        self.rng = random.Random(5318008)
        self.grad_mean = {}
        self.grad_var = {}
        self.grad_log_sum = {}
        self.grad_sq_log_sum = {}
        self.grad_reciprocal_sum = {}
        self.grad_reciprocal_sq_sum = {}
        self.grad_sum = {}
        self.grad_sq_sum = {}
        self.partial_sums = {}
        self.partial_sq_sums = {}
        self.grad_count = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_momentum_size = 0
        total_momentum_mb = 0
        total_precond_size = 0
        total_precond_mb = 0

        # update preconditioners all together
        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(self._prob_step)
        do_update = self.rng.random() < update_prob
        self._prob_step += 1

        balance = self.rng.random() < 0.01 and do_update

        for group in self.param_groups:
            mu_dtype = group.get("mu_dtype")
            precond_dtype = group.get("precond_dtype", torch.float32)


            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=mu_dtype or p.dtype
                    )
                    state["Q"], state["exprs"] = init_Q_exprs(
                        p,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                        dtype=precond_dtype,
                    )

                    # Print sizes
                    momentum_size = state["momentum_buffer"].numel()
                    momentum_mb = (
                        momentum_size
                        * state["momentum_buffer"].element_size()
                        / (2**20)
                    )
                    total_momentum_size += momentum_size
                    total_momentum_mb += momentum_mb

                    precond_size = sum(q.numel() for q in state["Q"])
                    precond_mb = sum(
                        q.numel() * q.element_size() for q in state["Q"]
                    ) / (2**20)
                    total_precond_size += precond_size
                    total_precond_mb += precond_mb

                    # Initialize statistics for all mean types
                    self.grad_mean[p] = torch.zeros_like(grad)
                    self.grad_var[p] = torch.zeros_like(grad)
                    self.grad_log_sum[p] = torch.zeros_like(grad)
                    self.grad_sq_log_sum[p] = torch.zeros_like(grad)
                    self.grad_reciprocal_sum[p] = torch.zeros_like(grad)
                    self.grad_reciprocal_sq_sum[p] = torch.zeros_like(grad)
                    self.grad_sum[p] = torch.zeros_like(grad)
                    self.grad_sq_sum[p] = torch.zeros_like(grad)
                    self.partial_sums[p] = torch.zeros_like(grad)
                    self.partial_sq_sums[p] = torch.zeros_like(grad)
                    self.grad_count[p] = 0

                state["step"] += 1

                # balance preconditioners about every 100 updates
                if grad.dim() > 1 and balance:
                    _balance_Q(state["Q"])

                # Update gradient statistics when not updating preconditioner
                if not do_update:
                    if group['mean_type'] == 'cesaro':
                        # Update running sum
                        self.grad_sum[p] += grad
                        self.grad_sq_sum[p] += grad * grad
                        # Update partial sums (Cesàro means)
                        count = self.grad_count[p] + 1
                        self.partial_sums[p] = self.grad_sum[p] / count
                        self.partial_sq_sums[p] = self.grad_sq_sum[p] / count
                    elif group['mean_type'] == 'geometric':
                        grad_abs = grad.abs() + self._tiny
                        self.grad_log_sum[p] += grad_abs.log()
                        self.grad_sq_log_sum[p] += (grad_abs * grad_abs).log()
                    elif group['mean_type'] == 'harmonic':
                        grad_abs = grad.abs() + self._tiny
                        self.grad_reciprocal_sum[p] += 1.0 / grad_abs
                        self.grad_reciprocal_sq_sum[p] += 1.0 / (grad_abs * grad_abs)
                    else:  # arithmetic
                        count = self.grad_count[p]
                        new_count = count + 1
                        delta = grad - self.grad_mean[p]
                        self.grad_mean[p] += delta / new_count
                        delta2 = grad - self.grad_mean[p]
                        self.grad_var[p] += delta * delta2
                    self.grad_count[p] += 1

                # Update preconditioner using collected statistics
                if do_update and self.grad_count[p] > 0:
                    if group['mean_type'] == 'cesaro':
                        count = self.grad_count[p]
                        cesaro_mean = self.partial_sums[p]
                        cesaro_var = self.partial_sq_sums[p] - cesaro_mean * cesaro_mean
                        cesaro_std = torch.sqrt(torch.clamp(cesaro_var, min=self._tiny))
                        fake_grad = cesaro_mean + cesaro_std * torch.randn_like(grad, dtype=precond_dtype)
                        # Reset statistics
                        self.grad_sum[p].zero_()
                        self.grad_sq_sum[p].zero_()
                        self.partial_sums[p].zero_()
                        self.partial_sq_sums[p].zero_()
                    elif group['mean_type'] == 'geometric':
                        count = self.grad_count[p]
                        geo_mean = (self.grad_log_sum[p] / count).exp() * grad.sign()
                        geo_std = ((self.grad_sq_log_sum[p] / count).exp()).sqrt()
                        fake_grad = geo_mean + geo_std * torch.randn_like(grad, dtype=precond_dtype)
                        # Reset statistics
                        self.grad_log_sum[p].zero_()
                        self.grad_sq_log_sum[p].zero_()
                    elif group['mean_type'] == 'harmonic':
                        count = self.grad_count[p]
                        harm_mean = count / self.grad_reciprocal_sum[p] * grad.sign()
                        harm_std = torch.sqrt(
                            abs(count / self.grad_reciprocal_sq_sum[p] - 
                                (count / self.grad_reciprocal_sum[p])**2)
                        )
                        fake_grad = harm_mean + harm_std * torch.randn_like(grad, dtype=precond_dtype)
                        # Reset statistics
                        self.grad_reciprocal_sum[p].zero_()
                        self.grad_reciprocal_sq_sum[p].zero_()
                    else:  # arithmetic
                        mean = self.grad_mean[p]
                        std = torch.sqrt(self.grad_var[p] / self.grad_count[p] + self._tiny)
                        fake_grad = mean + std * torch.randn_like(grad, dtype=precond_dtype)
                        # Reset statistics
                        self.grad_mean[p].zero_()
                        self.grad_var[p].zero_()
                    
                    _update_precond(
                        state["Q"],
                        state["exprs"],
                        torch.randn_like(grad, dtype=precond_dtype),
                        fake_grad,
                        group["precond_lr"],
                        self._tiny,
                    )
                    self.grad_count[p] = 0

                # Apply momentum to raw gradients first
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["b1"]).add_(grad)

                # Then precondition the momentum buffer
                pre_grad = _precond_grad(
                    state["Q"], state["exprs"], momentum_buffer
                ).to(dtype=p.dtype, non_blocking=True)

                # Apply trust region
                trust_region_fn = lambda x: 0.1 * torch.sign(x) * torch.log(
                    torch.abs(x) + 1
                ) + 0.9 * torch.tanh(x)
                pre_grad = torch.clip(
                    trust_region_fn(pre_grad / 1.5) * 1.5, min=-2, max=2
                )

                # Apply weight decay and update parameters
                if group["weight_decay"] != 0 and p.dim() >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])
                p.add_(pre_grad, alpha=-group["lr"])

        if total_momentum_size > 0:
            print(
                f"PSGD Momentum buffer size: {total_momentum_size} "
                f"elements, {total_momentum_mb:.2f} MB"
            )
            print(
                f"PSGD Preconditioners size: {total_precond_size} "
                f"elements, {total_precond_mb:.2f} MB"
            )

        # Calculate mean energy at the end of step if verbose
        if any(group["verbose"] for group in self.param_groups):
            all_energies = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None and len(self.state[p]) > 0:
                        pre_grad = _precond_grad(
                            self.state[p]["Q"], 
                            self.state[p]["exprs"], 
                            self.state[p]["momentum_buffer"]
                        ).to(dtype=p.dtype, non_blocking=True)
                        energy = torch.sum(pre_grad**2).item()
                        all_energies.append(energy)
            
            if all_energies:
                mean_energy = sum(all_energies) / len(all_energies)
                print(f"Mean preconditioned gradient energy: {mean_energy:.6f}")

        return loss


def init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'one_diag', 'all_diag']"
            )

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile(fullgraph=True, dynamic=False)
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.prod() ** (1 / len(Q_in))
    norms = geometric_mean / norms
    for i, q in enumerate(Q_in):
        q.mul_(norms[i])


def _lb(A, max_abs):
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)
    if value0 > value1:
        x = A[:, i].conj() @ A
        return max_abs * torch.linalg.vector_norm(
            (x / torch.linalg.vector_norm(x)) @ A.H
        )
    else:
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(
            A.H @ (x / torch.linalg.vector_norm(x))
        )


def _norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)


def _solve_triangular_right(X, A):
    """X @ inv(A)"""
    orig_dtype = X.dtype
    X = X.to(dtype=torch.float32, non_blocking=True)
    A = A.to(dtype=torch.float32, non_blocking=True)
    return torch.linalg.solve_triangular(A, X[None, :], upper=True, left=False).to(
        dtype=orig_dtype, non_blocking=True
    )[0]


@torch.compile(fullgraph=True, dynamic=False)
def _calc_A_and_conjB(exprA, G, Q, V):
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


@torch.compile(fullgraph=True, dynamic=False)
def _q_terms(exprGs, A, conjB):
    terms = []
    for exprG in exprGs:
        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)
        terms.append((term1, term2))
    return terms


def _update_precond(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    A, conjB = _calc_A_and_conjB(exprA, G, Q, V)

    terms = _q_terms(exprGs, A, conjB)

    for q, (term1, term2) in zip(Q, terms):
        tmp = term1 - term2
        tmp *= step
        if q.dim() < 2:
            tmp *= q
            tmp /= (term1 + term2).norm(float("inf")) + tiny
            q.sub_(tmp)
        else:
            tmp = torch.triu(tmp)
            tmp /= _norm_lower_bound(term1 + term2) + tiny
            tmp @= q
            q.sub_(tmp)


@torch.compile(fullgraph=True, dynamic=False)
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)
