import string
import torch

import torch_tensorrt 
torch._dynamo.config.cache_size_limit = 100_000_000
torch._dynamo.config.capture_scalar_outputs = False


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=200
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
        prob = torch.minimum(
            torch.maximum(
                max_prob * torch.exp(-decay * (n - flat_start)), torch.tensor(min_prob)
            ),
            torch.tensor(max_prob),
        )
        return prob

    return _schedule


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 0.001).
        b1 (float, optional): Momentum parameter (default: 0.9).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int, optional): Max size for dim's preconditioner to be
            triangular (default: 8192).
        max_skew_triangular (float, optional): Max skew for dim's preconditioner to be
            triangular (default: inf).
        min_ndim_triangular (int, optional): Minimum number of dimensions a layer needs
            to have triangular preconditioners (default: 2).
        mu_dtype (torch.dtype, optional): Dtype of the momentum accumulator. Defaults
            to the same dtype as the parameters.
        precond_dtype (torch.dtype, optional): Dtype of the preconditioner (default: None).
    """

    def __init__(
        self,
        params,
        lr=0.001,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        max_skew_triangular=float("inf"),
        min_ndim_triangular=2,
        mu_dtype=None,
        precond_dtype=None,
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
            max_skew_triangular=max_skew_triangular,
            min_ndim_triangular=min_ndim_triangular,
            precond_lr=0.1,  # precond lr hardcoded to 0.1
            precond_init_scale=1.0,  # precond init scale hardcoded to 1.0
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
        )
        super(Kron, self).__init__(params, defaults)

        self._global_clip = (
            sum(
                p.numel()
                for group in self.param_groups
                for p in group["params"]
                if p.requires_grad
            )
            ** 0.5
        )
        self._element_clip = 1.0
        self._tiny = 1e-30
        self._prob_step = 0

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
        device = self.param_groups[0]["params"][0].device
        do_update = torch.rand([], device=device) < update_prob
        self._prob_step += 1

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
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=mu_dtype or p.dtype
                    )
                    state["Q"], state["exprs"] = init_Q_exprs(
                        p,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["max_skew_triangular"],
                        group["min_ndim_triangular"],
                        dtype=precond_dtype,
                    )

                    # Calculate sizes
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

                state["step"] += 1

                # Update momentum buffer
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["b1"]).add_(grad, alpha=1 - group["b1"])

                # balance preconditioners about every 100 precond updates
                if grad.dim() > 1 and torch.rand([]) < 0.01 and do_update:
                    balance_Q(state["Q"])

                # Update preconditioner
                if do_update:
                    update_precond_kron_math_(
                        state["Q"],
                        state["exprs"],
                        torch.randn_like(momentum_buffer, dtype=precond_dtype),
                        momentum_buffer.to(dtype=precond_dtype, non_blocking=True),
                        group["precond_lr"],
                        self._tiny,
                    )

                # Precondition gradients
                pre_grad = precond_grad_kron_math(
                    state["Q"],
                    state["exprs"],
                    momentum_buffer.to(dtype=precond_dtype, non_blocking=True),
                ).to(dtype=p.dtype, non_blocking=True)

                # Apply trust region
                torch.nn.utils.clip_grad_norm_([pre_grad], self._global_clip)
                pre_grad.clamp_(-self._element_clip, self._element_clip)

                # Apply weight decay and update parameters
                if group["weight_decay"] != 0 and p.dim() >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])
                p.add_(pre_grad, alpha=-group["lr"])

                # Restore momentum buffer dtype (if needed)
                if mu_dtype is not None:
                    momentum_buffer.to(dtype=mu_dtype, non_blocking=True)

        if total_momentum_size > 0:
            print(
                f"PSGD Momentum buffer size: {total_momentum_size} elements, {total_momentum_mb:.2f} MB"
            )
            print(
                f"PSGD Preconditioners size: {total_precond_size} elements, {total_precond_mb:.2f} MB"
            )

        return loss


def init_Q_exprs(t, scale, max_size, max_skew, min_ndim_triangular, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->,"
        exprP = ",,->,"
        exprGs = [",->"]
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))
        if len(shape) == 1:
            beta_size = 1  # 2nd largest size
        else:
            beta_size = sorted(list(shape))[-2]

        Q = []
        exprGs = []
        piece1A, piece2A, piece3A = (
            [],
            "",
            "",
        )  # used for getting the subscripts for exprA
        piece1P, piece2P, piece3P, piece4P = (
            [],
            [],
            "",
            "",
        )  # used for getting the subscripts for exprP
        for i, size in enumerate(shape):
            if (
                size == 1
                or size > max_size
                or size > max_skew * beta_size
                or len(shape) < min_ndim_triangular
            ):
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]

                piece1 = "".join(
                    [
                        (letters[j + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

                piece1 = "".join(
                    [
                        (letters[j + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[j + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )
                exprGs.append(subscripts)

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile(fullgraph=True,backend='cudagraphs')
def balance_Q(Q_in):
    norms = torch.tensor([torch.max(torch.abs(q)) for q in Q_in])
    geometric_mean = norms.prod() ** (1 / len(Q_in))
    for i, q in enumerate(Q_in):
        q.mul_(geometric_mean / norms[i])

# @torch.compile(fullgraph=True,mode="max-autotune")
# def balance_Q(Q_in):
#     norms = [torch.max(torch.abs(q)) for q in Q_in]
#     geometric_mean = (torch.cumprod(torch.stack(norms), dim=0)[-1]) ** (1 / len(Q_in))
#     for i, q in enumerate(Q_in):
#         q.mul_(geometric_mean / norms[i])


@torch.compiler.disable
def norm_lower_bound(A):
    """Returns a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions
        and sizes suggest, norm(A) <= sqrt(2) * norm_lower_bound(A).
    Looks to be a very tight lower bound.
    """
    max_abs = torch.max(
        torch.abs(A)
    )  # used to normalize A to avoid numerically under- or over-flow
    if max_abs > 0:
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
    else:  # must have A=0
        return max_abs
 




# @torch.compile(fullgraph=True, mode="max-autotune")
# def solve_triangular_right(X, A):
#     # return X @ inv(A)
#     if X.dim() > 1:
#         X = X[None, :]
#     orig_dtype = X.dtype
#     X = X.to(dtype=torch.float32, non_blocking=True)
#     A = A.to(dtype=torch.float32, non_blocking=True)
#     out = torch.linalg.solve_triangular(A, X, upper=True, left=False).to(
#         dtype=orig_dtype, non_blocking=True
#     )
#     if X.dim() > 1:
#         return out[0]
#     return out

# torch._dynamo.list_backends()
# @torch.compile(fullgraph=True, mode="max-autotune")
# @torch.compile(fullgraph=True, backend='cudagraphs',options={"epilogue_fusion": False,"max_autotune":True,"triton.cudagraphs":True,"shape_padding":False})
# @torch.compile(fullgraph=True, backend='cudagraphs',options={"epilogue_fusion": False,"max_autotune":True,"triton.cudagraphs":True,"shape_padding":False})
@torch.compile(fullgraph=True, backend='cudagraphs',options={"epilogue_fusion": False,"max_autotune":True,"triton.cudagraphs":True,"shape_padding":False})
def solve_triangular_right(X, A):
    """X @ inv(A)"""
    orig_dtype = X.dtype
    X = X.to(dtype=torch.float32, non_blocking=True)
    A = A.to(dtype=torch.float32, non_blocking=True)
    return torch.linalg.solve_triangular(A, X[None, :], upper=True, left=False).to(
        dtype=orig_dtype, non_blocking=True
    )[0]

# @torch.compile(options={"epilogue_fusion": True,"max_autotune":True,"triton.cudagraphs":True,"shape_padding":True},fullgraph=True)
@torch.compile(fullgraph=True, mode="max-autotune")
def calc_A_and_conjB(Q, G, V, exprA):
    order = G.dim()
    A = torch.einsum(exprA, *Q, G)

    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    
    return A, conjB


@torch.compile(backend='onnxrt')
def update_precond_kron_math_(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with (vector, hess-vector-product)
    pair (V, G). V is optional, and we can set it to None if it is integrated out
    (NOT recommend).
    """
    exprA, exprGs, _ = exprs

    # Use the combined function
    A, conjB = calc_A_and_conjB(Q, G, V, exprA)

    for i, q in enumerate(Q):
        # Expanded calc_term_1_and_2
        term1 = torch.einsum(exprGs[i], A, A.conj())
        term2 = torch.einsum(exprGs[i], conjB.conj(), conjB)

        if q.dim() < 2:
            q.sub_(
                step
                / (torch.max(torch.abs(term1 + term2)) + tiny)
                * (term1 - term2)
                * q
            )
        else:
            # Expanded calc_triu_q
            triu_q = torch.triu(term1 - term2) @ q
            
            q.sub_(
                step
                / (norm_lower_bound(term1 + term2) + tiny)
                * triu_q
            )

# @torch.compile(options={"epilogue_fusion": True,"max_autotune":True,"triton.cudagraphs":True,"shape_padding":True},fullgraph=True)

@torch.compile(fullgraph=True,backend='inductor',options={"epilogue_fusion": True,"max_autotune":True,"triton.cudagraphs":True,"shape_padding":True})
def precond_grad_kron_math(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    # the last expr is exprP
    return torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)