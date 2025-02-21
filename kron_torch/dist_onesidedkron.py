"""Distributed PSGD One-Sided Kron for GPU clusters."""

import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist
from torch.backends import opt_einsum

opt_einsum.set_flags(True, "optimal")


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """
    max_prob_ = torch.tensor(max_prob, dtype=torch.float32)
    min_prob_ = torch.tensor(min_prob, dtype=torch.float32)
    decay_ = torch.tensor(decay, dtype=torch.float32)
    flat_start_ = torch.tensor(flat_start, dtype=torch.float32)

    @torch.compile
    def _schedule(n):
        """Exponential anneal with flat start."""
        prob = max_prob_ * torch.exp(-decay_ * (n - flat_start_))
        prob.clamp_(min=min_prob_, max=max_prob_)
        return prob

    return _schedule


import torch
from torch import Tensor
import torch.distributed as dist
import torch.optim

# Import DTensor components.
from torch.distributed._tensor import DeviceMesh, DTensor, DTensorSpec

def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
    """Anneal preconditioner update probability during beginning of training."""
    max_prob_ = torch.tensor(max_prob, dtype=torch.float32)
    min_prob_ = torch.tensor(min_prob, dtype=torch.float32)
    decay_ = torch.tensor(decay, dtype=torch.float32)
    flat_start_ = torch.tensor(flat_start, dtype=torch.float32)

    @torch.compile
    def _schedule(n):
        prob = max_prob_ * torch.exp(-decay_ * (n - flat_start_))
        prob.clamp_(min=min_prob_, max=max_prob_)
        return prob

    return _schedule


@torch.compile
def _update_momentum(momentum_buffer, grad, beta, step):
    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
    return momentum_buffer.div(1 - beta**step)


@torch.compile
def _oneside_precond_update(G: Tensor, Q: Tensor, lr: Tensor):
    m, n = G.shape
    if m < n:
        G = G.T
    V = torch.randn_like(G, dtype=torch.float32)
    Bh = torch.linalg.solve_triangular(Q.float(), V, upper=True, left=False).to(dtype=G.dtype)
    BBh = Bh.T @ Bh
    A = G @ Q.T
    AhA = A.T @ A
    A = AhA + BBh
    max_abs = A.norm(float("inf"))
    Q = Q - lr / torch.where(max_abs > 0, _lb(A, max_abs), max_abs) * torch.triu(AhA - BBh) @ Q
    return Q


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


# -- DTensor-based Optimizer for FSDP2 --

def get_device_mesh(world_size: int, device: str = "cuda") -> DeviceMesh:
    """Creates a 1D device mesh for DTensor operations."""
    device_ids = torch.arange(world_size)
    return DeviceMesh(device_type=device, mesh=device_ids)

class OneSidedKronFSDP2(torch.optim.Optimizer):
    """
    DTensor-enabled version of the one-sided Kron optimizer.
    
    This version is adapted to work with PyTorch's DTensor API so that it can be used
    in an FSDP2 workflow. It partitions the multi-dimensional parameters among workers,
    creates a DTensor update buffer that is sharded along the first dimension, and uses
    the DTensor abstraction to manage collective updates.
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
        dtype=torch.float32,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype

        # Create a 1D device mesh for DTensor usage.
        self.device_mesh = get_device_mesh(world_size, device="cuda")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        # Separate parameters:
        #   - 1D (or effectively 1D) parameters are updated using Adam.
        #   - Multi-dimensional parameters are updated with our DTensor-enabled scheme.
        adam_params = []
        kron_params = []
        for p in params:
            if p.ndim < 2 or max(p.shape) == torch.prod(torch.tensor(p.shape)):
                adam_params.append(p)
            else:
                kron_params.append(p)
        self._adam = torch.optim.Adam(adam_params, lr=lr * 3.0, betas=(0.9, 0.95), fused=True)

        # Organize multi-dimensional parameters into groups by number of elements.
        sizes = {p.numel() for p in kron_params}

        def create_update_buffer_dtensor(size: int):
            # Each worker will hold a local shard representing one row.
            # Local tensor shape: (1, size) so that the global DTensor has shape (world_size, size).
            local_update = torch.empty(1, size, dtype=dtype, device="cuda")
            spec = DTensorSpec(
                mesh=self.device_mesh,
                # Shard along dim 0 (each rank gets one row); replicate along dim 1.
                placements=["S", "R"]
            )
            update_buffer = DTensor.from_local_tensor(local_update, spec)
            # Also keep a direct reference to the local tensor for in-place updates.
            return dict(update_buffer=update_buffer, local_update=local_update)

        param_groups = [
            {"params": [p for p in kron_params if p.numel() == size], **create_update_buffer_dtensor(size)}
            for size in sizes
        ]

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            precond_lr=precond_lr,
            clip_update_rms=clip_update_rms,
            dtype=dtype,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device="cuda")
        self._prob_step = torch.tensor(0, dtype=torch.int32)
        self._update_counter = torch.tensor(0, dtype=torch.int32)

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

        # For each group of multi-dimensional parameters:
        for group in self.param_groups:
            params_world = None
            pending_update = False

            def update_prev():
                nonlocal pending_update
                if params_world and pending_update:
                    # Convert the DTensor update buffer to its global (assembled) view.
                    # In many DTensor workflows, operations automatically “see” the full tensor.
                    # Here, we obtain the local shard from the DTensor update buffer.
                    local_update = group["local_update"]
                    # For our design, since each local shard represents one row,
                    # we create a one-element list (or more if the current chunk has multiple parameters).
                    updates = [local_update.view_as(p_world) for p_world in params_world]
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
                    state = self.state.setdefault(p, {})
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["Q"] = torch.eye(
                            min(g.view(-1, g.shape[-1]).shape), dtype=self.dtype, device=g.device
                        )
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
                            g.view(-1, g.shape[-1]),
                            state["Q"],
                            torch.tensor(group["precond_lr"], dtype=self.dtype, device="cuda"),
                        )
                    g = _oneside_precond_g(g.view(-1, g.shape[-1]), state["Q"])
                    if group["clip_update_rms"]:
                        _clip_update_rms(g)
                    g = g.flatten()

                    # Instead of using an async all-gather, we now update our local shard directly.
                    # Each worker writes its computed update (reshaped to 1 x numel) into its DTensor shard.
                    group["local_update"].copy_(g.unsqueeze(0))
                    # Record the parameters corresponding to this chunk.
                    params_world = group["params"][base_i : min(base_i + self.world_size, num_params)]
                    pending_update = True
                else:
                    # For any missing index in this chunk, write zeros.
                    group["local_update"].zero_()

            update_prev()

        # Finally, update the 1D parameters using Adam.
        self._adam.step()

        return loss
