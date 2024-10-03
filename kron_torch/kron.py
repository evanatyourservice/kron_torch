import torch
import opt_einsum


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
        return prob.item()

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
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
        )
        super(Kron, self).__init__(params, defaults)

        self._global_clip_norm = (
            sum(
                p.numel()
                for group in self.param_groups
                for p in group["params"]
                if p.requires_grad
            )
            ** 0.5
        )
        self._element_wise_clip = 1.0
        self._tiny = 1e-30
        self._precond_init_scale = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            precond_dtype = group.get("precond_dtype", torch.float32)
            mu_dtype = group.get("mu_dtype")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(
                        p, dtype=mu_dtype or p.dtype
                    )
                    state["Q"], state["exprs"] = init_Q_exprs(
                        p,
                        self._precond_init_scale,
                        group["max_size_triangular"],
                        group["max_skew_triangular"],
                        group["min_ndim_triangular"],
                        dtype=precond_dtype,
                    )

                state["step"] += 1

                # Update momentum buffer
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["b1"]).add_(grad, alpha=1 - group["b1"])

                # Update preconditioner
                update_prob = group["preconditioner_update_probability"]
                if callable(update_prob):
                    update_prob = update_prob(state["step"])

                if torch.rand([], device=p.device) < update_prob:
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
                torch.nn.utils.clip_grad_norm_([pre_grad], self._global_clip_norm)
                pre_grad.clamp_(-self._element_wise_clip, self._element_wise_clip)

                # Apply weight decay and update parameters
                if group["weight_decay"] != 0 and p.dim() >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])
                p.add_(pre_grad, alpha=-group["lr"])

                # Restore momentum buffer dtype (if needed)
                if mu_dtype is not None:
                    momentum_buffer.to(dtype=mu_dtype, non_blocking=True, copy=False)

        return loss

    def _init_Qs_exprs(self, momentum_buffer_with_grad, group, precond_dtype):
        Qs_exprs = [
            init_Q_exprs(
                m,
                self._precond_init_scale,
                group["max_size_triangular"],
                group["max_skew_triangular"],
                group["min_ndim_triangular"],
                dtype=precond_dtype,
            )
            for m in momentum_buffer_with_grad
        ]

        # Print preconditioner sizes
        Qs_n_elements = sum(sum(q.numel() for q in Q[0]) for Q in Qs_exprs)
        Qs_size_MB = sum(
            sum(q.numel() * q.element_size() / (2**20) for q in Q[0]) for Q in Qs_exprs
        )
        print(
            f"PSGD Preconditioners size: {Qs_n_elements} elements, {Qs_size_MB:.2f} MB"
        )

        return Qs_exprs


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
            # We must have norm(x) > 0 since norm(x) >= value0 > value1 >= 0
            # Also, avoid expression norm(x*A^H)/norm(x) as x*A^H could under/over flow
            return max_abs * torch.linalg.vector_norm(
                (x / torch.linalg.vector_norm(x)) @ A.H
            )
        else:
            x = A @ A[j].conj()
            # normx = torch.linalg.vector_norm(x)
            # if normx > 0:
            #     # Again, avoid expression norm(A^H*x)/norm(x) as A^H*x could under/over flow
            #     return max_abs * torch.linalg.vector_norm(A.H @ (x / normx))
            # else:  # A = 0
            #     return normx
            return max_abs * torch.linalg.vector_norm(
                A.H @ (x / torch.linalg.vector_norm(x))
            )
    else:  # must have A=0
        return max_abs


def init_Q_exprs(t, scale, max_size, max_skew, min_ndim_triangular, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable contraction expressions for updating Q and preconditioning gradient.
    """
    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape)
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape)]
    else:  # tensor
        if len(shape) > 26:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters; "
                "Replace 26 with larger numbers!"
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

                piece1A.append(opt_einsum.get_symbol(i))
                piece2A = piece2A + opt_einsum.get_symbol(i)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)

                piece1 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 26)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i + 26)
                )
                exprGs.append(
                    opt_einsum.contract_expression(subscripts, t.shape, t.shape)
                )
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A = piece2A + opt_einsum.get_symbol(i + 26)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                a, b, c = (
                    opt_einsum.get_symbol(i),
                    opt_einsum.get_symbol(i + 26),
                    opt_einsum.get_symbol(i + 805),
                )
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

                piece1 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 26)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 805)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1
                    + ","
                    + piece2
                    + "->"
                    + opt_einsum.get_symbol(i + 26)
                    + opt_einsum.get_symbol(i + 805)
                )
                exprGs.append(
                    opt_einsum.contract_expression(subscripts, t.shape, t.shape)
                )

        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], t.shape
        )

        subscripts = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )
        exprP = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


def update_precond_kron_math_(Q, exprs, V, G, step, tiny):
    """
    Update Kronecker product preconditioner Q with (vector, hess-vector-product)
    pair (V, G). V is optional, and we can set it to None if it is integrated out
    (NOT recommend).
    """

    def triangular_inv(A):
        # return inv(A); used only when V is None, i.e., integrating out V
        I = torch.eye(A.shape[0], dtype=torch.float32, device=A.device)
        orig_dtype = A.dtype
        A = A.to(dtype=torch.float32)
        return torch.linalg.solve_triangular(A, I, upper=True).to(dtype=orig_dtype)

    def solve_triangular_right(X, A):
        # return X @ inv(A)
        if X.dim() > 1:
            orig_dtype = X.dtype
            X = X.to(dtype=torch.float32)
            A = A.to(dtype=torch.float32)
            return torch.linalg.solve_triangular(A, X, upper=True, left=False).to(
                dtype=orig_dtype
            )
        else:  # torch.linalg.solve_triangular complains if X.dim() < 2! So insert None
            orig_dtype = X.dtype
            X = X.to(dtype=torch.float32)
            A = A.to(dtype=torch.float32)
            return torch.linalg.solve_triangular(A, X[None, :], upper=True, left=False)[
                0
            ].to(dtype=orig_dtype)

    order = G.dim()  # order of tensor
    if order > 1 and torch.rand([]) < 0.01:
        # balance the dynamic range of Q if there are more than one factors
        norms = [torch.max(torch.abs(q)) for q in Q]
        gmean = (torch.cumprod(torch.stack(norms), dim=0)[-1]) ** (
            1 / order
        )  # geometric mean
        for i, q in enumerate(Q):
            q.mul_(gmean / norms[i])

    exprA, exprGs, _ = exprs

    A = exprA(*Q, G)
    if V is not None:
        invQhinvQ, trace_invQhinvQ = None, None
        p = list(range(order))
        conjB = torch.permute(
            V.conj(), p[1:] + p[:1]
        )  # permute dims like [0,1,2,3,4] -> [1,2,3,4,0]
        for i, q in enumerate(Q):
            conjB = conjB / q if q.dim() < 2 else solve_triangular_right(conjB, q)
            if i < order - 1:  # transpose dims like
                # [1,2,3,4,0]->[0,2,3,4,1]->[0,1,3,4,2]->[0,1,2,4,3]->[0,1,2,3,4]
                conjB = torch.transpose(conjB, i, order - 1)
    else:  # V is integrated out, and no need to form conjB
        conjB = None
        invQ = [1 / q if q.dim() < 2 else triangular_inv(q) for q in Q]
        invQhinvQ = [q.conj() * q if q.dim() < 2 else q.H @ q for q in invQ]
        trace_invQhinvQ = [
            torch.sum(q) if q.dim() < 2 else torch.trace(q) for q in invQhinvQ
        ]

    for i, q in enumerate(Q):
        term1 = exprGs[i](A, A.conj())
        if conjB is not None:
            term2 = exprGs[i](conjB.conj(), conjB)
        else:  # V is integrated out
            term2 = 1.0
            for j, trace in enumerate(trace_invQhinvQ):
                term2 = term2 * (trace if i != j else invQhinvQ[i])

        if q.dim() < 2:  # q is a diagonal matrix or scalar
            q.sub_(
                step
                / (torch.max(torch.abs(term1 + term2)) + tiny)
                * (term1 - term2)
                * q
            )
        else:
            q.sub_(
                step
                / (norm_lower_bound(term1 + term2) + tiny)
                * torch.triu(term1 - term2)
                @ q
            )


def precond_grad_kron_math(Q, exprs, G):
    """
    Precondition gradient G with preconditioner Q.
    """
    # the last expr is exprP
    return exprs[-1](*[q.conj() for q in Q], *Q, G)
