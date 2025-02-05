"""Distributed PSGD Kron for GPU clusters."""

import string
import random
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist
from torch.backends import opt_einsum

from kron_torch import precond_update_prob_schedule

opt_einsum.set_flags(True, "optimal")


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron with simple layer-wise pipeline parallelism.

    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode (str, optional): Memory saving mode for preconditioners.
            Options:
            None: Set all preconditioners to be triangular
            'smart_one_diag': Sets any large dims that stand out to be diagonal
            'one_diag': Sets the largest or last dim to be diagonal
            'all_diag': Sets all preconditioners to be diagonal

        momentum_into_precond_update (bool): Whether to send momentum into
            preconditioner update instead of raw gradients.
        precond_lr (float, optional): Learning rate for preconditioner updates.
            Default: 0.1
        precond_init_scale (float, optional): Initial scale for preconditioner
            matrices. Default: 1.0
        partition_grads (bool): Whether to partition gradients.
        block_size (int): Size of partitions for gradient partitioning.
        mars (bool): Whether to use MARS.
        adam_grafting (bool): Whether to graft adam's lr onto the update.
        clip_update_rms (bool): Whether to clip the update RMS at 1.1.
        dtype (torch.dtype): Data type for parameters and gradients.
        rank (int): This worker's rank, used in pipeline partitioning.
        world_size (int): Total number of workers, used in pipeline partitioning.
    """

    def __init__(
        self,
        params,
        lr=0.001,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        precond_lr=0.1,
        precond_init_scale=1.0,
        partition_grads=False,
        block_size=1024,
        mars=False,
        adam_grafting=False,
        clip_update_rms=False,
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

        def create_update_buffer(size):
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
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            partition_grads=partition_grads,
            block_size=block_size,
            mars=mars,
            adam_grafting=adam_grafting,
            clip_update_rms=clip_update_rms,
            dtype=dtype,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device="cuda")
        self._prob_step = 0
        self._update_counter = 0
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
            update_prob = update_prob(torch.tensor(self._prob_step, dtype=torch.float32))
        self._update_counter += 1
        do_update = self._update_counter >= 1 / update_prob
        if do_update:
            self._update_counter = 0
        self._prob_step += 1

        balance = do_update and self.rng.random() < 0.01

        for group in self.param_groups:
            handle = None
            params_world = None
            pending_update = False
            
            def update_prev():
                nonlocal pending_update
                if params_world and handle and pending_update:
                    handle.wait()
                    for p_world, g_world in zip(params_world, group["update_buffer_views"][: len(params_world)]):
                        update = g_world.view_as(p_world)
                        if group["weight_decay"] > 0 and p_world.dim() >= 2:
                            update.add_(p_world, alpha=group["weight_decay"])
                        p_world.add_(update.to(p_world.dtype), alpha=-group["lr"])
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
                    if grads.dim() > 1:
                        if "merged_shape" not in self.state[p]:
                            shape1 = [np.prod(grads.shape[:-1]), grads.shape[-1]]
                            shape2 = [grads.shape[0], np.prod(grads.shape[1:])]
                            shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                            state["merged_shape"] = shape
                        else:
                            shape = state["merged_shape"]
                        grads = grads.view(*shape)
                    
                    beta = self.defaults["b1"]
                    if self.defaults["mars"]:
                        if "prev_grads" not in state:
                            state["prev_grads"] = grads
                        prev_grads = state["prev_grads"]
                        state["prev_grads"] = grads.clone()
                        grads.add_((beta / (1 - beta)) * (grads - prev_grads), alpha=0.025)  # alpha is mars gamma

                    # partition grads
                    if self.defaults["partition_grads"]:
                        if "partitioner" not in state:
                            dim_diag = _get_dim_diag(
                                self.defaults["memory_save_mode"],
                                grads.shape,
                                self.defaults["max_size_triangular"],
                                self.defaults["min_ndim_triangular"],
                            )
                            state["partitioner"] = BlockPartitioner(
                                grads.shape, self.defaults["block_size"], dim_diag
                            )
                        grads_list = state["partitioner"].partition(grads)
                    else:
                        grads_list = [grads]

                    precond_grads = []
                    for i, grad in enumerate(grads_list):
                        if f"step_{i}" not in state:
                            state[f"step_{i}"] = 0
                            state[f"momentum_buffer_{i}"] = torch.zeros_like(grad, dtype=self.dtype)
                            if self.defaults["adam_grafting"]:
                                state[f"nu_{i}"] = torch.zeros_like(grad, dtype=self.dtype)
                            dim_diag = _get_dim_diag(
                                self.defaults["memory_save_mode"],
                                grads.shape,
                                self.defaults["max_size_triangular"],
                                self.defaults["min_ndim_triangular"],
                            )
                            state[f"Q_{i}"], state[f"exprs_{i}"] = _init_Q_exprs(
                                grad, self.defaults["precond_init_scale"], dim_diag, self.dtype
                            )

                        state[f"step_{i}"] += 1

                        # momentum
                        momentum_buffer = state[f"momentum_buffer_{i}"]
                        momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                        debiased_momentum = momentum_buffer.div(1 - beta ** state[f"step_{i}"])

                        if self.defaults["adam_grafting"]:
                            nu = state[f"nu_{i}"]
                            nu.mul_(0.95).add_(grad**2, alpha=0.05)
                            nu_hat = nu.div(1 - 0.95 ** state[f"step_{i}"])
                            adam_update_norm = torch.linalg.norm(debiased_momentum / (nu_hat.sqrt() + 1e-8))

                        # balance Qs
                        if grad.dim() > 1 and balance:
                            _balance_Q(state[f"Q_{i}"])

                        # update preconditioner
                        if do_update:
                            _update_precond(
                                state[f"Q_{i}"],
                                state[f"exprs_{i}"],
                                (
                                    debiased_momentum
                                    if self.defaults["momentum_into_precond_update"]
                                    else grad
                                ),
                                torch.tensor(
                                    self.defaults["precond_lr"], dtype=self.dtype, device="cuda"
                                ),
                                self._tiny,
                            )

                        # precondition grads
                        precond_grad = _precond_grad(state[f"Q_{i}"], state[f"exprs_{i}"], debiased_momentum)

                        if self.defaults["adam_grafting"]:
                            precond_grad.mul_(adam_update_norm / (precond_grad.norm() + 1e-12))

                        precond_grads.append(precond_grad)

                    # merge partitions
                    if self.defaults["partition_grads"]:
                        g = state["partitioner"].merge_partitions(precond_grads)
                    else:
                        g = precond_grads[0]

                    # clip update RMS at 1.1
                    if self.defaults["clip_update_rms"]:
                        g.mul_(
                            torch.minimum(
                                torch.tensor(1.0, dtype=self.dtype),
                                1.1 / g.square().mean().sqrt().add(1e-12),
                            )
                        )

                    # flatten before all_gather
                    g_flat = g.flatten()
                else:
                    g_flat = group["update_buffer_views"][self.rank].zero_()

                handle = dist.all_gather_into_tensor(group["update_buffer"], g_flat, async_op=True)
                params_world = group["params"][base_i:min(base_i + self.world_size, num_params)]
                pending_update = True

            update_prev()

        return loss


class OneSidedKron(torch.optim.Optimizer):
    """One-sided Kronecker-factored second-order optimizer similar to Muon but using PSGD-style preconditioner.
    
    Args:
        params (iterable): Parameters to optimize.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        start_sparse_precond_updates (int): When to start doing preconditioner update every n steps instead of every step.
        update_freq (int): How often to do preconditioner update after start_sparse_precond_updates.
        weight_decay (float): Weight decay.
        rank (int): This worker's rank, used in pipeline partitioning.
        world_size (int): Total number of workers, used in pipeline partitioning.
    """
    def __init__(self, params, lr=0.0003, momentum=0.9, start_sparse_precond_updates=1000, update_freq=3, weight_decay=0.0, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum,
                       start_sparse_precond_updates=start_sparse_precond_updates,
                       update_freq=update_freq,
                       weight_decay=weight_decay)
        params: list[Tensor] = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            start_sparse_precond_updates = group["start_sparse_precond_updates"]
            update_freq = group["update_freq"]
            weight_decay = group["weight_decay"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    grads = g_world.view_as(p_world)
                    if weight_decay > 0:
                        grads.add_(p_world, alpha=weight_decay)
                    p_world.add_(grads, alpha=-lr)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["step"] = 0
                    state["step"] += 1
                    step = state["step"]
                    buf: Tensor = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g, alpha=1-momentum)  # ema momentum
                    g = buf.div(1 - momentum ** step)
                    orig_shape = g.shape
                    g: Tensor = g.view(-1, orig_shape[-1])
                    if 'Q' not in state:
                        state['Q'] = torch.eye(min(g.shape), dtype=torch.bfloat16, device=g.device)
                    g = g.bfloat16()
                    # we don't have to update precond every step once loss is stable
                    if step < start_sparse_precond_updates or (step >= start_sparse_precond_updates and step % update_freq == 0):
                        state['Q'] = _oneside_precond_update(g, state['Q'])
                    g = _oneside_precond_g(g, state['Q'])
                    rms = g.square().mean().sqrt()
                    if rms > 1.1:  # can clip at RMS 1.1
                        g *= 1.1 / rms
                    g = g.flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


@torch.compile
def _oneside_precond_update(G: Tensor, Q: Tensor):
    m, n = G.shape
    if m < n:
        G = G.T
    V = torch.randn_like(G, dtype=torch.float32)
    Bh = torch.linalg.solve_triangular(Q.float(), V, upper=True, left=False).bfloat16()  # roughly same complexity as a matmul
    BBh = Bh.T @ Bh
    A = G @ Q.bfloat16().T
    AhA = A.T @ A
    lr = torch.tensor(0.1, dtype=torch.bfloat16)
    Q = Q - lr / _norm_lower_bound(AhA + BBh) * torch.triu(AhA - BBh) @ Q
    return Q


@torch.compile
def _oneside_precond_g(G: Tensor, Q: Tensor):
    m, n = G.shape
    if m < n:
        return torch.einsum('ji,jk,kl->il', Q, Q, G)
    else:
        return torch.einsum('ij,kj,kl->il', G, Q, Q)


def _get_dim_diag(memory_save_mode, shape, max_size, min_ndim):
    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "smart_one_diag":
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

        scale = scale ** (1 / len(shape))

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


def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))


def _norm_lower_bound(A: Tensor):
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)


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
    order = G.dim()
    V = torch.randn_like(G, device=G.device)
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
    """Update Kronecker product preconditioner Q with pair (V, G)."""
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


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py

    Scalable Second Order Optimization for Deep Learning by
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
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
