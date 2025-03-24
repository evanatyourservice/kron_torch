"""Distributed PSGD Kron for GPU clusters."""

import string
import random
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist


class ProbScheduler:
    """Scheduler for annealing preconditioner update probability.
    
    Implements an exponential anneal with a flat start.
    """
    
    def __init__(self, max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
        self.max_prob = torch.tensor(max_prob, dtype=torch.float32)
        self.min_prob = torch.tensor(min_prob, dtype=torch.float32)
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self.flat_start = torch.tensor(flat_start, dtype=torch.float32)
        # Make compiled function optional to avoid pickling issues
        self._compiled = False
        try:
            self._compiled_schedule = torch.compile(self._schedule_fn)
            self._compiled = True
        except Exception:
            # Fallback to non-compiled version if compilation fails
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


class Kron(torch.optim.Optimizer):
    """PSGD Kron optimizer with layer-wise pipeline parallelism.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        b1: Momentum
        weight_decay: Weight decay
        preconditioner_update_probability: Prob of updating preconditioner (default: anneals 1.0->0.03 by 4000 steps)
        max_size_triangular: Max size for triangular preconditioner
        min_ndim_triangular: Min dims needed for triangular preconditioners
        memory_save_mode: Memory saving mode:
            None: All triangular preconditioners
            'smart_one_diag': Large outlier dims use diagonal
            'one_diag': Largest dim uses diagonal
            'all_diag': All diagonal preconditioners
        precond_lr: Preconditioner learning rate (default: 0.1)
        precond_init_scale: Initial preconditioner scale (default: 1.0)
        merge_dims: Whether to combine dims to make grad tensor a matrix
        partition_grads: Whether to partition gradients
        block_size: Partition size for gradients
        clip_update_rms: Clip update RMS at 1.1
        dtype: Data type for params/grads
        rank: Worker rank for pipeline
        world_size: Total workers for pipeline
    """

    def __init__(
        self,
        params,
        lr=0.0003,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        precond_lr=0.1,
        precond_init_scale=1.0,
        merge_dims=True,
        partition_grads=False,
        block_size=1024,
        clip_update_rms=True,
        dtype=torch.float32,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        params = [*params]
        sizes = {p.numel() for p in params}

        def create_update_buffer(size: int):
            b = torch.empty(world_size, size, dtype=dtype, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])

        param_groups = [
            {"params": [p for p in params if p.numel() == size], **create_update_buffer(size)}
            for size in sizes
        ]

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            merge_dims=merge_dims,
            partition_grads=partition_grads,
            block_size=block_size,
            clip_update_rms=clip_update_rms,
            dtype=dtype,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device="cuda")
        self._prob_step = torch.tensor(0, dtype=torch.int32)
        self._update_counter = torch.tensor(0, dtype=torch.int32)
        self.rng = random.Random(42)
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

        balance = do_update and self.rng.random() < 0.01

        for group in self.param_groups:
            handle = None
            params_world = None
            pending_update = False

            def update_prev():
                nonlocal pending_update
                if params_world and handle and pending_update:
                    handle.wait()
                    updates = [
                        g_world.view_as(p_world)
                        for p_world, g_world in zip(
                            params_world, group["update_buffer_views"][: len(params_world)]
                        )
                    ]
                    _update_params(
                        params_world,
                        updates,
                        torch.tensor(group["weight_decay"], dtype=self.dtype, device="cuda"),
                        torch.tensor(group["lr"], dtype=self.dtype, device="cuda"),
                    )
                    pending_update = False

            num_params = len(group["params"])
            for base_i in range(0, num_params, self.world_size):
                update_prev()
                param_idx = base_i + self.rank
                if param_idx < num_params:
                    p = group["params"][param_idx]
                    if p.grad is None:
                        continue

                    grads = p.grad.to(self.dtype)
                    state = self.state[p]

                    # merge smaller dims
                    if grads.dim() > 2 and group.get('merge_dims', True):
                        if "merged_shape" not in state:
                            shape1 = [np.prod(grads.shape[:-1]), grads.shape[-1]]
                            shape2 = [grads.shape[0], np.prod(grads.shape[1:])]
                            shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                            state["merged_shape"] = shape
                        grads = grads.view(*state["merged_shape"])

                    # partition grads
                    grads_list = (
                        state["partitioner"].partition(grads)
                        if group["partition_grads"]
                        and "partitioner" in state
                        else [grads]
                    )

                    if group["partition_grads"] and "partitioner" not in state:
                        state["partitioner"] = BlockPartitioner(
                            grads.shape,
                            group["block_size"],
                            _get_dim_diag(
                                group["memory_save_mode"],
                                grads.shape,
                                group["max_size_triangular"],
                                group["min_ndim_triangular"],
                            )
                        )
                        grads_list = state["partitioner"].partition(grads)

                    precond_grads = []
                    for i, grad in enumerate(grads_list):
                        if f"step_{i}" not in state:
                            state[f"step_{i}"] = 0
                            state[f"momentum_buffer_{i}"] = torch.zeros_like(grad, dtype=self.dtype)
                            state[f"Q_{i}"], state[f"exprs_{i}"] = _init_Q_exprs(
                                grad,
                                group["precond_init_scale"],
                                _get_dim_diag(
                                    group["memory_save_mode"],
                                    grad.shape,
                                    group["max_size_triangular"],
                                    group["min_ndim_triangular"],
                                ),
                                self.dtype,
                            )

                        state[f"step_{i}"] += 1
                        debiased_momentum = _update_momentum(
                            state[f"momentum_buffer_{i}"],
                            grad,
                            torch.tensor(group["b1"], dtype=self.dtype, device="cuda"),
                            torch.tensor(state[f"step_{i}"], dtype=self.dtype, device="cuda"),
                        )

                        if grad.dim() > 1 and balance:
                            _balance_Q(state[f"Q_{i}"])

                        if do_update:
                            _update_precond(
                                state[f"Q_{i}"],
                                state[f"exprs_{i}"],
                                debiased_momentum,
                                torch.tensor(group["precond_lr"], dtype=self.dtype, device="cuda"),
                                self._tiny,
                            )

                        precond_grads.append(
                            _precond_grad(state[f"Q_{i}"], state[f"exprs_{i}"], debiased_momentum)
                        )

                    g = (
                        state["partitioner"].merge_partitions(precond_grads)
                        if group["partition_grads"]
                        else precond_grads[0]
                    )

                    if group["clip_update_rms"]:
                        _clip_update_rms(g)

                    g = g.flatten()
                else:
                    g = group["update_buffer_views"][self.rank].zero_()

                handle = dist.all_gather_into_tensor(group["update_buffer"], g, async_op=True)
                params_world = group["params"][base_i:min(base_i + self.world_size, num_params)]
                pending_update = True

            update_prev()

        return loss


@torch.compile
def _update_momentum(momentum_buffer, grad, beta, step):
    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
    return momentum_buffer.div(1 - beta**step)


def _get_dim_diag(memory_save_mode, shape, max_size, min_ndim):
    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "smart_one_diag":  # Thanks to @ClashLuke heavyball repo
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        sorted_shape = sorted(shape)
        if len(shape) > 1 and sorted_shape[-1] > sorted_shape[-2]:
            dim_diag[rev_sorted_dims[0]] = True
    elif memory_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [i == rev_sorted_dims[0] for i in range(len(shape))]
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
            "[None, 'smart_one_diag', 'one_diag', 'all_diag']"
        )
    if len(shape) < min_ndim:
        return [True for _ in shape]
    for i in range(len(shape)):
        size = shape[i]
        if size == 1 or size > max_size:
            dim_diag[i] = True
    return dim_diag


def _init_Q_exprs(t, scale, dim_diag, dtype):
    """Initialize preconditioner Q and reusable einsum expressions."""
    letters = string.ascii_lowercase + string.ascii_uppercase

    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:
        if len(shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

        scale = torch.tensor(scale ** (1 / len(shape)), dtype=dtype, device=t.device)

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, dim_d in enumerate(dim_diag):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(shape[i], dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(shape[i], dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                )
                piece2 = "".join(
                    [(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))]
                )
                subscripts = piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))


def _lb(A: Tensor, max_abs: Tensor):
    """Cheap lower bound for the spectral norm of A."""
    A /= max_abs
    a0 = torch.einsum("ij,ij->j", A, A)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("j,kj->k", x, A)
    x = x.norm()
    x *= max_abs
    return x


def _solve_triangular_right(X: Tensor, A: Tensor):
    """X @ inv(A)"""
    orig_dtype = A.dtype
    # roughly same complexity as a matmul
    return (
        torch.linalg.solve_triangular(
            A.float(),
            X.reshape(-1, X.size(-1)).float(),
            upper=True,
            left=False,
            unitriangular=False,
        )
        .to(dtype=orig_dtype)
        .reshape_as(X)
    )


def _calc_A_and_conjB(exprA, G, Q):
    """Calculate A and conjB."""
    order = G.dim()
    V = torch.randn_like(G)
    eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=G.dtype, device=G.device)
    G += eps.sqrt() * G.abs().mean() * V
    conjB = V.permute(*range(1, order), 0)
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    A = torch.einsum(exprA, *Q, G)
    return A, conjB


@torch.compile
def _update_precond(Q, exprs, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G).
    
    Thanks to @ClashLuke heavyball repo for many of the optimizations in this function.
    """
    exprA, exprGs, _ = exprs
    A, conjB = _calc_A_and_conjB(exprA, G, Q)
    for q, exprG in zip(Q, exprGs):
        term1 = torch.einsum(exprG, A, A)
        term2 = torch.einsum(exprG, conjB, conjB)
        term1, term2 = term1 - term2, term1 + term2
        term1 *= step
        norm = term2.norm(float("inf"))
        if q.dim() < 2:
            term1 *= q / norm.clamp_(min=tiny)
        else:
            torch.triu(term1, out=term1)
            term1 /= torch.where(norm > 0, _lb(term2, norm), norm).clamp_(tiny)
            term1 = torch.mm(term1, q)
        q.sub_(term1)


@torch.compile
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *Q, *Q, G)


@torch.compile
def _clip_update_rms(g):
    g.mul_(
        torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            1.1 / g.square().mean().sqrt().add(1e-12),
        )
    )


@torch.compile
def _update_params(params_world, updates, weight_decay, lr):
    if weight_decay > 0:
        torch._foreach_add_(updates, params_world, alpha=weight_decay)
    torch._foreach_add_(params_world, updates, alpha=-lr)


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    """

    def __init__(self, param_shape, block_size, dim_diag):
        assert len(dim_diag) == len(param_shape), "dim_diag must have same length as param_shape"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._shape = param_shape
        self._split_indices = []
        self._split_dims = []
        for i, (d, is_diag) in enumerate(zip(param_shape, dim_diag)):
            if 0 < block_size < d and not is_diag:
                nsplit = (d - 1) // block_size
                if nsplit > 0:
                    self._split_indices.append([(j + 1) * block_size for j in range(nsplit)])
                    self._split_dims.append(i)
        self._total_blocks = (
            np.prod([len(indices) + 1 for indices in self._split_indices])
            if self._split_indices
            else 1
        )

    def partition(self, tensor):
        assert tensor.shape == self._shape
        blocks = [tensor]
        for dim, indices in zip(self._split_dims, self._split_indices):
            new_blocks = []
            for block in blocks:
                split_blocks = torch.tensor_split(block, indices, dim=dim)
                new_blocks.extend(split_blocks)
            blocks = new_blocks
        return blocks

    def merge_partitions(self, partitions):
        blocks = list(partitions)
        for dim, indices in zip(reversed(self._split_dims), reversed(self._split_indices)):
            n = len(indices) + 1
            merged = []
            for i in range(0, len(blocks), n):
                merged.append(torch.cat(blocks[i : i + n], dim=dim))
            blocks = merged
        return blocks[0]
