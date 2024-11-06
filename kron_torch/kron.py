import random
import string

import numpy as np
import torch

torch._dynamo.config.cache_size_limit = 1_000_000

try:
    torch.backends.opt_einsum.strategy = "dynamic-programming"
except AttributeError:
    # opt_einsum backend is not available, so we'll skip setting the strategy
    pass

_mode = "default"


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=250):
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
        prob = torch.minimum(torch.maximum(max_prob * torch.exp(-decay * (n - flat_start)), torch.tensor(min_prob)),
                             torch.tensor(max_prob), )
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
        mu_dtype (torch.dtype, optional): Dtype of the momentum accumulator.
        precond_dtype (torch.dtype, optional): Dtype of the preconditioner.
        trust_region_scale (float): Trust region on preconditioned grads. Normally this
            doesn't need to be changed but if things seem unstable you can try reducing
            this to 1.5.
    """

    def __init__(self, params, lr=0.001, b1=0.9, weight_decay=0.0, preconditioner_update_probability=0.5,
                 max_size_triangular=2048, min_ndim_triangular=2, memory_save_mode=None, mu_dtype=None,
                 precond_dtype=None, trust_region_scale=2.0, warmup_steps=0, efficient: bool = False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter: {b1}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        defaults = dict(lr=lr, b1=b1, weight_decay=weight_decay,
                        preconditioner_update_probability=preconditioner_update_probability,
                        max_size_triangular=max_size_triangular, min_ndim_triangular=min_ndim_triangular,
                        memory_save_mode=memory_save_mode, precond_lr=0.1,  # precond lr hardcoded to 0.1
                        precond_init_scale=1.0,  # precond init scale hardcoded to 1.0
                        mu_dtype=mu_dtype, precond_dtype=precond_dtype, trust_region_scale=trust_region_scale,
                        warmup_steps=warmup_steps, efficient=efficient)
        super(Kron, self).__init__(params, defaults)

        self._tiny = torch.finfo(torch.bfloat16).tiny
        self._prob_step = 0
        self.rng = random.Random(0x81293218)  # seed chosen by fair dice roll (guaranteed to be random)

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
            precond_dtype = group.get("precond_dtype", torch.float32)
            mu_dtype = group.get("mu_dtype")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=mu_dtype or p.dtype)
                    state["Q"], state["exprs"] = init_Q_exprs(p, group["precond_init_scale"],
                                                              group["max_size_triangular"],
                                                              group["min_ndim_triangular"], group["memory_save_mode"],
                                                              dtype=precond_dtype, )

                    # Print sizes
                    momentum_size = state["momentum_buffer"].numel()
                    momentum_mb = (momentum_size * state["momentum_buffer"].element_size() / (2 ** 20))
                    total_momentum_size += momentum_size
                    total_momentum_mb += momentum_mb

                    precond_size = sum(q.numel() for q in state["Q"])
                    precond_mb = sum(q.numel() * q.element_size() for q in state["Q"]) / (2 ** 20)
                    total_precond_size += precond_size
                    total_precond_mb += precond_mb

                state["step"] += 1

                # Update momentum buffer
                beta = group["b1"]
                bias_correction = 1 - beta ** state["step"]
                debiased = (1 - beta) / bias_correction

                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(grad, debiased)
                momentum_buffer = momentum_buffer.to(dtype=precond_dtype, non_blocking=True)

                if group['efficient']:
                    grouped_fn(state["Q"], state["exprs"], momentum_buffer.to(dtype=precond_dtype, non_blocking=True),
                               group["precond_lr"], group["trust_region_scale"], group['weight_decay'], group['lr'],
                               group['warmup_steps'], state['step'], balance, do_update)
                    continue

                # balance preconditioners about every 100 updates
                if grad.dim() > 1 and balance:
                    _balance_Q(state["Q"])

                # Update preconditioner
                if do_update:
                    _update_precond(state["Q"], state["exprs"], momentum_buffer, group["precond_lr"])

                # Precondition gradients
                _epilogue(p, state["Q"], state["exprs"], momentum_buffer, group["trust_region_scale"],
                          group['weight_decay'], group['lr'], group['warmup_steps'], state['step'])

        if total_momentum_size > 0:
            print(f"PSGD Momentum buffer size: {total_momentum_size} "
                  f"elements, {total_momentum_mb:.2f} MB")
            print(f"PSGD Preconditioners size: {total_precond_size} "
                  f"elements, {total_precond_mb:.2f} MB")

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
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

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
            raise ValueError(f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                             "[None, 'one_diag', 'all_diag']")

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d):
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
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

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
                piece2 = "".join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
                subscripts = (piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26])
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P)

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile(fullgraph=False, mode=_mode)
def grouped_fn(Q, exprs, momentum_buffer, precond_lr, trust_region_scale, weight_decay, lr, warmup_steps, step, balance,
               do_update):
    if momentum_buffer.dim() > 1 and balance:
        _balance_Q(Q)
    if do_update:
        _update_precond(Q, exprs, momentum_buffer, precond_lr)
    _epilogue(momentum_buffer, Q, exprs, momentum_buffer, trust_region_scale, weight_decay, lr, warmup_steps, step)


@torch.compile(fullgraph=True, dynamic=True, mode=_mode)
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float('inf')) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    for i, q in enumerate(Q_in):
        q.mul_(norms[i])


def _lb(A, max_abs):
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)

    ah = A.H
    comp = value0 > value1
    x = torch.where(comp, A[:, i], A[j])
    x = x.conj()
    at = torch.where(comp, A, A.T)
    if x.dim() > 1:
        x = torch.where(comp, x, x.T)
    x = x @ at
    x /= torch.linalg.vector_norm(x)
    x = x @ torch.where(comp, ah, ah.T)
    x = torch.linalg.vector_norm(x)
    x *= max_abs
    return x


def _norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = A.norm(float('inf'))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)


def _solve_triangular_right(X, A):
    """X @ inv(A)"""
    orig_dtype = X.dtype
    X = X.to(dtype=torch.float32, non_blocking=True)
    A = A.to(dtype=torch.float32, non_blocking=True)    
    unsqueeze = X.dim() == 1
    if unsqueeze:
        X = X.unsqueeze(0)
    out = torch.linalg.solve_triangular(A, X, upper=True, left=False).to(
        dtype=orig_dtype, non_blocking=True
    )
    if not unsqueeze:
        return out
    return out[0]


@torch.compile(fullgraph=True, dynamic=True, mode=_mode)
def _calc_A_and_conjB(exprA, G, Q, V):
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        if q.dim() < 2:
            conjB = conjB / q
        else:
            conjB = _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


@torch.compile(fullgraph=True, dynamic=True, mode=_mode)
def _q_terms(exprGs, A, conjB):
    terms = []
    for exprG in exprGs:
        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)
        terms.append((term1, term2))
    return terms


def _update_precond(Q, exprs, G, step):
    tiny = torch.finfo(torch.bfloat16).tiny
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs
    V = torch.randn_like(G, dtype=G.dtype)

    A, conjB = _calc_A_and_conjB(exprA, G, Q, V)

    terms = _q_terms(exprGs, A, conjB)

    for q, (term1, term2) in zip(Q, terms):
        tmp = term1 - term2
        tmp *= step
        if q.dim() < 2:
            tmp *= q
            tmp /= (term1 + term2).norm(float('inf')) + tiny
            q.sub_(tmp)
        else:
            tmp = torch.triu(tmp)
            tmp /= _norm_lower_bound(term1 + term2) + tiny
            tmp @= q
            q.sub_(tmp)


@torch.compile(fullgraph=True, dynamic=True, mode=_mode)
def _epilogue(p, Q, exprs, G, trust_region_scale, weight_decay, lr, warmup_steps, step):
    """Precondition gradient G with preconditioner Q."""
    grad = torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)
    grad.div_(trust_region_scale)
    grad.tanh_()
    grad.mul_(trust_region_scale)

    if weight_decay != 0 and p.dim() >= 2:
        grad.add_(p, alpha=weight_decay)
    p.add_(grad, alpha=-lr * min(step / warmup_steps, 1))
