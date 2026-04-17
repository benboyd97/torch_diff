from typing import Any, Callable, Union, MutableMapping, Sequence, Literal
import torch
from torch.optim import Optimizer

import torch.nn as nn
from symo.factory2 import GroupsSpec, CovFactory, MeanFactory
from symo.utils import to_dtype
import os
NDArray = torch.Tensor
Decomp = tuple[NDArray, NDArray, NDArray]
DecompPrecision = Literal["fp32", "fp64"]

torch.manual_seed(0)

class Symo(Optimizer):
    """Symo optimizer."""

    @torch.no_grad
    def __init__(
        self,
        params,
        groups_spec: GroupsSpec,
        lr: float | Callable = 1e-1,
        grads_beta: float = 0.0,
        factors_beta: float = 0.0,
        grads_bias_corr: bool = False,
        factors_bias_corr: bool = True,
        update_correction: bool = False,
        update_avg: bool = False,
        sub_group_avg: bool = True,
        damping: float = 0.0,
        prior_alpha: float = 1.0,
        top_k: float = 0.05,
        block_diag: bool = False,
        decomp_precision: DecompPrecision | None = "fp32",
    ):
        if not 0.0 <= damping:
            raise ValueError(f"Invalid damping value: {damping}")
        if not 0.0 <= grads_beta <= 1.0:
            raise ValueError(f"Invalid grads_beta value: {grads_beta}")
        if not 0.0 <= factors_beta <= 1.0:
            raise ValueError(f"Invalid factors_beta value: {factors_beta}")

        params = list(params)
        # TODO(awav): Global factors buffer. Generalize to multiple parameter groups!

        dev_cfg = dict(device=params[0].device, dtype=params[0].dtype)

        avg_factory = MeanFactory(groups_spec).to(**dev_cfg)
        cov_factory = CovFactory(groups_spec, block_diag_only=block_diag).to(**dev_cfg)

        defaults = dict(
            lr=lr,
            damping=damping,
            prior_alpha = prior_alpha,
            top_k = top_k,
            grads_beta=grads_beta,
            factors_beta=factors_beta,
            groups_spec=groups_spec,
            grads_bias_corr=grads_bias_corr,
            factors_bias_corr=factors_bias_corr,
            update_correction=update_correction,
            update_avg=update_avg,
            sub_group_avg=sub_group_avg,
        )

        super().__init__(params, defaults)
        self.decomp_precision = dtype_decomp(decomp_precision)
        self.avg_factory = avg_factory
        self.cov_factory = cov_factory
        self.step_t = None

    def _init_group(
        self,
        group: MutableMapping,
    ):
        params_with_grad: list[NDArray] = []
        grads: list[NDArray] = []
        grad_momentum_bufs: list[NDArray] = []

        for p in group["params"]:
            if p.grad is None:
                raise RuntimeError(
                    "Symo requires gradients to be finite for all parameters"
                )

            if torch.is_complex(p):
                raise RuntimeError("Symo does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Symo does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )

            grad_momentum_bufs.append(state["momentum_buffer"])

        if self.step_t is None:
            self.step_t = torch.tensor(0.0, dtype=p.dtype, device=p.device)

        return params_with_grad, grads, grad_momentum_bufs

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            grads_beta = group["grads_beta"]
            grads_corr = group["grads_bias_corr"]
            factors_beta = group["factors_beta"]
            factors_corr = group["factors_bias_corr"]
            update_correction = group["update_correction"]
            updates_avg = group["update_avg"]
            sub_group_avg = group["sub_group_avg"]
            cov_factory = self.cov_factory
            avg_factory = self.avg_factory
            prior_alpha = group["prior_alpha"]
            top_k = group["top_k"]

            group_variables = self._init_group(group)
            params, grads, grads_buf = group_variables

            self._symo_update(
                params,
                grads,
                grads_buf,
                avg_factory,
                cov_factory,
                self.step_t,
                lr,
                grads_beta=grads_beta,
                factors_beta=factors_beta,
                damping=damping,
                grads_corr=grads_corr,
                factors_corr=factors_corr,
                updates_corr=update_correction,
                updates_avg=updates_avg,
                sub_group_avg=sub_group_avg,
                prior_alpha = prior_alpha,
                top_k = top_k
            )

        return loss

    def _symo_update(
        self,
        params: Sequence[NDArray | nn.Parameter],
        grads: Sequence[NDArray],
        grads_buf: Sequence[NDArray],
        avg_buf: MeanFactory,
        cov_factory: CovFactory,
        step: float | NDArray,
        lr: float | NDArray,
        grads_beta: float | NDArray,
        factors_beta: float | NDArray,
        grads_corr: bool,
        factors_corr: bool,
        updates_corr: bool,
        updates_avg: bool,
        sub_group_avg: bool,
        damping: float | NDArray,
        prior_alpha: float,
        top_k: float
    ):
        """Core Symo update logic."""
        step += 1

        apply_momentum(grads_buf, grads, grads_beta)

        new_grads = grads_buf
        if grads_corr:
            new_grads = apply_bias(new_grads, grads_beta, step)

        if sub_group_avg:
            group_avg = avg_buf.avg(new_grads)
            grads_del = values_diff(new_grads, group_avg)
        else:
            grads_del = new_grads

        weights_buf = cov_factory.weights(clone=True)
        cov_factory.outer_update(grads_del)

        apply_momentum(
            weights_buf,
            cov_factory.weights(),
            factors_beta,
        )

        new_weights = weights_buf
        if factors_corr:
            new_weights = apply_bias(new_weights, factors_beta, step)

        cov_factory.update_weights(new_weights)

        ##
        device = params[0].device
        dtype = params[0].dtype
        surrogate = cov_factory.cov(surrogate=True, device=device, dtype=dtype)
        decomp = svd(surrogate, dtype=self.decomp_precision, alpha = prior_alpha, top_k = top_k)
        surrogate_sqrt_inv = inv_sqrt_mat(decomp, damping=damping)
        cov_update(cov_factory, surrogate_sqrt_inv, surrogate=True)

        ##
        if updates_avg:
            if grads_corr and not updates_corr:
                avg = avg_buf.avg(grads_buf)
                apply_grads = values_diff(grads_buf, avg)
            else:
                apply_grads = grads_del
        else:
            apply_grads = new_grads if updates_corr else grads_buf

        

        noise = [torch.randn_like(param) for param in apply_grads]
        updates = cov_factory.matvec(noise)

        ##
        if factors_corr:
            cov_factory.update_weights(weights_buf)
        else:
            cov_factory.update_weights(new_weights)

        update_with_lr(lr, params, updates)


def svd(
    mat: NDArray | Sequence[NDArray],
    hermitian: bool = True,
    dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    top_k: float = 0.05,
) -> Decomp | Sequence[Decomp]:
    if isinstance(mat, list):
        return [mat_decomp(m, hermitian, dtype, alpha, top_k) for m in mat]
    return mat_decomp(mat, hermitian, dtype, alpha, top_k)


def inv_sqrt_mat(
    decomp: Decomp | Sequence[Decomp], damping: float | NDArray = 0.0
) -> NDArray | Sequence[NDArray]:
    """Compute inverse square root of a matrix."""
    if isinstance(decomp, list):
        return [truncated_inv_sqrt(*d, damping=damping) for d in decomp]

    return truncated_inv_sqrt(*decomp, damping=damping)


def mat_decomp(
    mat,
    hermitian: bool,
    dtype: torch.dtype | None = None,
    alpha: float = 1.0,
    top_k: float = 0.05,
    save = True
) -> Decomp:
    dtype_orig = None
    if dtype is not None:
        dtype_orig = mat.dtype
        mat = mat.to(dtype=dtype)

    if hermitian:
        out = torch.linalg.eigh(mat)
        s, u = to_dtype(out, dtype_orig)
        if save:
            path = "symo_eig_surr_block.pt"
            if os.path.exists(path):
                existing = torch.load(path, weights_only=False)
                combined = torch.cat([existing["eigvals"], s.cpu()])
                torch.save({"eigvals": combined}, path)
            else:
                torch.save({"eigvals": s.cpu()}, path)
        num_params = s.shape[0]
        top = int(num_params*top_k)
        s = s[-top:] + torch.ones_like(s[-top:],dtype=dtype_orig)*alpha
        u = u[:,-top:]
        return u, s, u.T
    else:
        out = torch.linalg.svd(mat)
        u, s, vt = to_dtype(out, dtype_orig)
        num_params = s.shape[0]
        top = int(num_params*top_k)

        s = s[-top:] + torch.ones_like(s[-top:],dtype=dtype_orig)*alpha
        return u[:,-top:], s, vt[-top:,:]


def truncated_inv_sqrt(
    u: NDArray, s: NDArray, vt: NDArray, damping: float | NDArray = 0.0
):
    damping = s.max() * damping
    inv_sqrt_s = torch.where(s > damping, 1.0 / torch.sqrt(s), 0.0)
    mat_inv = (u * inv_sqrt_s[None]) @ vt
    return mat_inv


def sqrt_mat(u: NDArray, s: NDArray, vt: NDArray, damping: float = 0.0) -> torch.Tensor:
    """Compute inverse square root of a matrix."""
    sqrt_s = torch.where(s > damping, torch.sqrt(s), 0.0)
    mat_sqrt = (u * sqrt_s[None]) @ vt
    return mat_sqrt




def apply_momentum(
    buffer: Sequence[NDArray],
    new_values: Sequence[NDArray],
    beta: float | NDArray,
):
    """Apply momentum."""

    for i, buf in enumerate(buffer):
        new_val = new_values[i]
        buf.lerp_(new_val, 1 - beta)


def apply_bias(
    values: Sequence[NDArray],
    beta: float | NDArray,
    step: float | NDArray,
):
    """Apply bias correction."""

    bias_corr = 1 - beta**step
    updates = []

    for val in values:
        val_corr = val / bias_corr
        updates.append(val_corr)

    return updates


def apply_grads_beta(
    bufs,
    values,
    beta,
    step: torch.Tensor,
    bias: bool = True,
):
    """Apply momentum with optional bias correction."""

    bias_corr = 1 - beta**step
    updates = []

    for i, val in enumerate(values):
        buf = bufs[i]
        buf.lerp_(val, 1 - beta)

        if not bias:
            updates.append(buf)
        else:
            buf_corr = buf / bias_corr
            updates.append(buf_corr)

    return updates


def values_diff(lhs: Sequence[NDArray], rhs: Sequence[NDArray]) -> Sequence[NDArray]:
    out = [l - r for (l, r) in zip(lhs, rhs)]
    return out


def apply_factors_beta(
    bufs,
    values,
    beta,
    step: torch.Tensor,
    bias: bool = True,
):
    """Apply momentum with optional bias correction."""

    bias_corr = 1 - beta**step
    updates = []

    for i, val in enumerate(values):
        buf = bufs[i]
        weights_val = val.weights
        weights_buf = bufs[i].weights
        weights_buf.lerp_(weights_val, 1 - beta)

        if not bias:
            updates.append(buf)

        else:
            weights_buf_corr = weights_buf / bias_corr
            buf_corr = buf.__class__(buf.eq, weights_buf_corr)
            updates.append(buf_corr)

    return updates


def update_with_lr(lr: float | NDArray, params, updates):
    for i, p in enumerate(params):
        u = updates[i]
        p.add_(u, alpha=lr)


def cov_update(factory: CovFactory, cov: NDArray | Sequence[NDArray], surrogate: bool):
    if isinstance(cov, Sequence):
        factory.cov_block_diag_update(cov, surrogate=surrogate)
    else:
        factory.cov_update(cov, surrogate=surrogate)


def dtype_decomp(prec: DecompPrecision | None) -> torch.dtype | None:
    precs: dict[str, torch.dtype] = dict(
        fp32=torch.float32,
        fp64=torch.float64,
    )
    return precs[prec] if prec in precs else None