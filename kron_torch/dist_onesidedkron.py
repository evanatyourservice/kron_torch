"""Distributed PSGD One-Sided Kron for GPU clusters."""

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


class OneSidedKron(torch.optim.Optimizer):
    """PSGD One-Sided Kron optimizer with layer-wise pipeline parallelism.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        b1: Momentum
        weight_decay: Weight decay
        preconditioner_update_probability: Prob of updating preconditioner (default: anneals 1.0->0.03 by 4000 steps)
        precond_lr: Preconditioner learning rate (default: 0.1)
        clip_update_rms: Clip update RMS at 1.1
        merge_dims: Whether to combine dims to make grad tensor a matrix
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
        precond_lr=0.1,
        clip_update_rms=True,
        merge_dims=True,
        dtype=torch.float32,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        params = [*params]

        # handle 1D tensors with adam
        adam_params = []
        kron_params = []
        for p in params:
            if p.ndim < 2 or max(p.shape) == np.prod(p.shape):
                adam_params.append(p)
            else:
                kron_params.append(p)
        self._adam = torch.optim.Adam(adam_params, lr=lr * 3.0, betas=(0.9, 0.95), fused=True)

        sizes = {p.numel() for p in kron_params}

        def create_update_buffer(size: int):
            b = torch.empty(world_size, size, dtype=dtype, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])

        param_groups = [
            {"params": [p for p in kron_params if p.numel() == size], **create_update_buffer(size)}
            for size in sizes
        ]

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            precond_lr=precond_lr,
            clip_update_rms=clip_update_rms,
            merge_dims=merge_dims,
            dtype=dtype,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device="cuda")
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

                    g = p.grad.to(self.dtype)
                    state = self.state[p]

                    if g.dim() > 2 and group.get('merge_dims', True):
                        if "merged_shape" not in state:
                            shape1 = [np.prod(g.shape[:-1]), g.shape[-1]]
                            shape2 = [g.shape[0], np.prod(g.shape[1:])]
                            shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                            state["merged_shape"] = shape
                        g = g.view(*state["merged_shape"])

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["Q"] = torch.eye(min(g.shape), dtype=self.dtype, device=g.device)
                        state["step"] = torch.tensor(0, dtype=torch.int32, device="cuda")
                    state["step"] += 1
                    g = _update_momentum(
                        state["momentum_buffer"],
                        g,
                        torch.tensor(group["b1"], dtype=self.dtype, device="cuda"),
                        state["step"]
                    )

                    if do_update:
                        state["Q"] = _oneside_precond_update(
                            g,
                            state["Q"],
                            torch.tensor(group["precond_lr"], dtype=self.dtype, device="cuda"),
                        )
                    g = _oneside_precond_g(g, state["Q"])
                    if group["clip_update_rms"]:
                        _clip_update_rms(g)
                    g = g.flatten()
                else:
                    g = group["update_buffer_views"][self.rank].zero_()

                handle = dist.all_gather_into_tensor(group["update_buffer"], g, async_op=True)
                params_world = group["params"][base_i : min(base_i + self.world_size, num_params)]
                pending_update = True

            update_prev()

        self._adam.step()

        return loss


@torch.compile
def _update_momentum(momentum_buffer, grad, beta, step):
    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
    return momentum_buffer.div(1 - beta**step)


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


@torch.compile
def _oneside_precond_update(G: Tensor, Q: Tensor, lr: Tensor):
    m, n = G.shape
    if m < n:
        G = G.T
    V = torch.randn_like(G, dtype=torch.float32)
    # roughly same complexity as a matmul
    Bh = torch.linalg.solve_triangular(Q.float(), V, upper=True, left=False).to(dtype=G.dtype)
    BBh = Bh.T @ Bh
    A = G @ Q.T
    AhA = A.T @ A
    A = AhA + BBh
    max_abs = A.norm(float("inf"))
    Q = Q - lr / torch.where(max_abs > 0, _lb(A, max_abs), max_abs) * torch.triu(AhA - BBh) @ Q
    return Q


@torch.compile
def _oneside_precond_g(G: Tensor, Q: Tensor):
    m, n = G.shape
    if m < n:
        return torch.einsum("ji,jk,kl->il", Q, Q, G)
    else:
        return torch.einsum("ij,kj,kl->il", G, Q, Q)


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
