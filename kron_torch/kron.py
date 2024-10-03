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

        self._params_with_grad = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        total_params = sum(p.numel() for p in self._params_with_grad)

        self._global_clip_norm = total_params**0.5
        self._element_wise_clip = 1.0
        self._tiny = 1e-30
        self._Qs_exprs = None
        self._precond_init_scale = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            precond_dtype = group.get("precond_dtype", torch.float32)
            mu_dtype = group.get("mu_dtype")

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = [
                    torch.zeros_like(p, dtype=mu_dtype or p.dtype)
                    for p in group["params"]
                ]

                # Print momentum buffer sizes
                mu_n_elements = sum(m.numel() for m in group["momentum_buffer"])
                mu_size_MB = sum(
                    m.numel() * m.element_size() / (2**20)
                    for m in group["momentum_buffer"]
                )
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

            b1 = group["b1"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            # Update momentum buffers
            params_with_grad = []
            momentum_buffer_with_grad = []
            for p, m in zip(group["params"], group["momentum_buffer"]):
                if p.grad is not None:
                    params_with_grad.append(p)
                    momentum_buffer_with_grad.append(m)
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)

            if self._Qs_exprs is None:
                self._init_Qs_exprs(momentum_buffer_with_grad, group, precond_dtype)

            # Update preconditioner
            update_prob = group["preconditioner_update_probability"]
            if callable(update_prob):
                update_prob = update_prob(group["step"])

            if torch.rand([], device=p.device) < update_prob:
                self._update_preconditioner(
                    momentum_buffer_with_grad, group, precond_dtype
                )

            # Precondition gradients
            pre_grads = self._precondition_gradients(
                momentum_buffer_with_grad, params_with_grad, group, precond_dtype
            )

            # Apply trust region
            self._apply_trust_region(pre_grads)

            # Apply weight decay and update parameters
            for param, g in zip(params_with_grad, pre_grads):
                if weight_decay != 0 and param.dim() >= 2:
                    g.add_(param, alpha=weight_decay)
                param.add_(g, alpha=-lr)

            # Restore momentum buffer dtype (if needed)
            if mu_dtype is not None:
                for m in group["momentum_buffer"]:
                    m.to(dtype=mu_dtype, non_blocking=True, copy=False)

        return loss

    def _init_Qs_exprs(self, momentum_buffer_with_grad, group, precond_dtype):
        self._Qs_exprs = [
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
        Qs_n_elements = sum(sum(q.numel() for q in Q[0]) for Q in self._Qs_exprs)
        Qs_size_MB = sum(
            sum(q.numel() * q.element_size() / (2**20) for q in Q[0])
            for Q in self._Qs_exprs
        )
        print(
            f"PSGD Preconditioners size: {Qs_n_elements} elements, {Qs_size_MB:.2f} MB"
        )

    def _update_preconditioner(self, momentum_buffer_with_grad, group, precond_dtype):
        for Q_exprs, m in zip(self._Qs_exprs, momentum_buffer_with_grad):
            update_precond_kron_math_(
                *Q_exprs,
                torch.randn_like(m, dtype=precond_dtype),
                m.to(dtype=precond_dtype, non_blocking=True),
                group["precond_lr"],
                self._tiny,
            )

    def _precondition_gradients(
        self, momentum_buffer_with_grad, params_with_grad, group, precond_dtype
    ):
        return [
            precond_grad_kron_math(
                *Q_exprs, m.to(dtype=precond_dtype, non_blocking=True)
            ).to(dtype=p.dtype, non_blocking=True)
            for Q_exprs, m, p in zip(
                self._Qs_exprs, momentum_buffer_with_grad, params_with_grad
            )
        ]

    def _apply_trust_region(self, pre_grads):
        # Global gradient clipping
        torch.nn.utils.clip_grad_norm_(pre_grads, self._global_clip_norm)
        # Element-wise gradient clipping
        for g in pre_grads:
            g.clamp_(-self._element_wise_clip, self._element_wise_clip)


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
