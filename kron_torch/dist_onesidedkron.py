"""Distributed PSGD One-Sided Kron with DTensor and FSDP2 support (Improved Version)."""

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
from torch.backends import opt_einsum
from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, DTensorSpec

opt_einsum.set_flags(True, "optimal")

def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
    """Anneal preconditioner update probability during training."""
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
    return momentum_buffer.div_(1 - beta**step)

@torch.compile
def _oneside_precond_update(G: Tensor, Q: Tensor, lr: Tensor):
    """
    Update the preconditioner Q using the one-sided update.
    Q is a replicated DTensor.
    """
    m, n = G.shape
    # If needed, transpose G so that its row-dimension is larger.
    if m < n:
        G = G.T

    # Generate a random matrix V for the update.
    V = torch.randn_like(G, dtype=torch.float32)
    # Convert Q to its local tensor for computation.
    Q_local = Q.to_local()
    # Solve triangular system.
    Bh = torch.linalg.solve_triangular(Q_local.float(), V, upper=True, left=False).to(dtype=G.dtype)
    BBh = Bh.T @ Bh

    A = G @ Q_local.T
    AhA = A.T @ A

    # Aggregate matrices across ranks.
    dist.all_reduce(AhA, op=dist.ReduceOp.SUM)
    dist.all_reduce(BBh, op=dist.ReduceOp.SUM)

    A_total = AhA + BBh
    max_abs = A_total.norm(float("inf"))
    # Compute a cheap lower bound on the spectral norm.
    update_scale = lr / torch.where(max_abs > 0, _lb(A_total, max_abs), max_abs)
    update = update_scale * (torch.triu(AhA - BBh) @ Q_local)
    Q_new_local = Q_local - update

    # Reconstruct Q as a replicated DTensor.
    spec = DTensorSpec(device_mesh=Q.device_mesh, placements=[Replicate()])
    return DTensor.from_local_tensor(Q_new_local, spec)

def _lb(A: Tensor, max_abs: Tensor):
    """Cheap lower bound for the spectral norm of A."""
    A_div = A / max_abs
    a0 = torch.einsum("ij,ij->j", A_div, A_div)
    i = torch.argmax(a0)
    x = torch.index_select(A_div, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A_div)
    x = x / x.norm()
    x = torch.einsum("j,kj->k", x, A_div)
    return x.norm() * max_abs

@torch.compile
def _oneside_precond_g(G: Tensor, Q: Tensor):
    """Apply the one-sided preconditioning to gradient G using Q."""
    Q_local = Q.to_local()
    m, n = G.shape
    if m < n:
        return torch.einsum("ji,jk,kl->il", Q_local, Q_local, G)
    else:
        return torch.einsum("ij,kj,kl->il", G, Q_local, Q_local)

@torch.compile
def _clip_update_rms(g):
    """Clip gradient updates to have RMS below 1.1."""
    g.mul_(
        torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            1.1 / (g.square().mean().sqrt().add(1e-12)),
        )
    )

class OneSidedKronFSDP2(torch.optim.Optimizer):
    """
    DTensor-enabled one-sided Kron optimizer with FSDP2 support.
    
    This optimizer splits parameters into two groups:
      - 1D (or effectively 1D) parameters updated with Adam.
      - Multi-dimensional parameters updated with a one-sided Kron preconditioner.
    
    The preconditioner Q is maintained as a replicated DTensor,
    and key intermediate matrices are aggregated using all_reduce.
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
        device_mesh: DeviceMesh = None
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        params = [*params]
        adam_params = []
        kron_params = []
        for p in params:
            if p.ndim < 2 or max(p.shape) == np.prod(p.shape):
                adam_params.append(p)
            else:
                kron_params.append(p)
        self._adam = torch.optim.Adam(adam_params, lr=lr * 3.0, betas=(0.9, 0.95), fused=True)

        # Group multi-dimensional parameters by number of elements.
        sizes = {p.numel() for p in kron_params}
        param_groups = [{'params': [p for p in kron_params if p.numel() == size]} for size in sizes]

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
        self.device_mesh = device_mesh if device_mesh is not None else DeviceMesh("cuda", list(range(dist.get_world_size())))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Determine if we should update the preconditioner based on the schedule.
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
                state = self.state.setdefault(p, {})
                # Optionally merge dimensions if the gradient is high dimensional.
                if g.dim() > 2 and group.get("merge_dims", True):
                    if "merged_shape" not in state:
                        shape1 = [int(np.prod(g.shape[:-1])), g.shape[-1]]
                        shape2 = [g.shape[0], int(np.prod(g.shape[1:]))]
                        # Choose the reshaping that minimizes shape difference.
                        shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                        state["merged_shape"] = shape
                    g = g.view(*state["merged_shape"])

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    # Initialize Q as an identity matrix and convert to a replicated DTensor.
                    min_dim = min(g.shape)
                    Q_local = torch.eye(min_dim, dtype=self.dtype, device="cuda")
                    spec = DTensorSpec(device_mesh=self.device_mesh, placements=[Replicate()])
                    state["Q"] = DTensor.from_local_tensor(Q_local, spec)
                    state["step"] = torch.tensor(0, dtype=torch.int32, device="cuda")
                state["step"] += 1

                # Update momentum.
                g = _update_momentum(
                    state["momentum_buffer"],
                    g,
                    torch.tensor(group["b1"], dtype=self.dtype, device="cuda"),
                    state["step"]
                )

                # Optionally update the preconditioner.
                if do_update:
                    state["Q"] = _oneside_precond_update(
                        g,
                        state["Q"],
                        torch.tensor(group["precond_lr"], dtype=self.dtype, device="cuda")
                    )

                # Apply the one-sided preconditioning.
                g = _oneside_precond_g(g, state["Q"])
                if group["clip_update_rms"]:
                    _clip_update_rms(g)
                # Compute the update (including weight decay if needed) and apply it.
                update = -group["lr"] * g
                if group["weight_decay"] > 0:
                    update += group["weight_decay"] * p
                p.add_(update)

        # Finally, update the 1D parameters using Adam.
        self._adam.step()
        return loss
