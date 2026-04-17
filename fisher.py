"""
Matrix-free empirical Fisher + low-rank Laplace posterior sampling for LeNet.

The empirical Fisher is:
    F = (1/N) sum_i g_i g_i^T
where g_i = J_i^T (e_{y_i} - p_i) is the per-sample gradient of the log-likelihood.

This differs from the GGN:
    G = (1/N) sum_i J_i^T (diag(p_i) - p_i p_i^T) J_i

For cross-entropy + softmax, GGN == true Fisher (expectation under model).
The empirical Fisher uses actual labels y_i instead of expectation over y ~ p_i.
They coincide at a perfect optimum but differ away from it.

Fisher-vector product:
    F v = (1/N) sum_i g_i (g_i^T v)

which is just a gradient times a scalar — only ONE vjp pass per sample needed
(no jvp required), making this cheaper than the GGN matvec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, vjp, grad
from typing import Tuple, Callable


# ── LeNet ──────────────────────────────────────────────────────────────────

class LeNet(nn.Module):
    def __init__(self, output_dim=10, activation="tanh"):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, output_dim)

    def act_fun(self, x):
        if self.activation == "tanh":   return torch.tanh(x)
        if self.activation == "relu":   return F.relu(x)
        raise ValueError(f"Unknown activation {self.activation}")

    def forward(self, x):
        if x.dim() != 4: x = x.unsqueeze(0)
        x = F.max_pool2d(self.act_fun(self.conv1(x)), 2, 2)
        x = F.max_pool2d(self.act_fun(self.conv2(x)), 2, 2)
        x = torch.flatten(x, 1)
        x = self.act_fun(self.fc1(x))
        x = self.act_fun(self.fc2(x))
        return self.fc3(x)


# ── Parameter utilities ────────────────────────────────────────────────────

def get_params_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def params_to_dict(model: nn.Module, flat: torch.Tensor) -> dict:
    d, offset = {}, 0
    for name, p in model.named_parameters():
        n = p.numel()
        d[name] = flat[offset: offset + n].reshape(p.shape)
        offset += n
    return d

def pvec_to_dict(model: nn.Module, v: torch.Tensor) -> dict:
    return params_to_dict(model, v)


# ── Empirical Fisher matvec via vjp only ──────────────────────────────────
#
# F v = (1/N) sum_i g_i (g_i^T v)
#
# where g_i = dL_i/dtheta  (per-sample gradient of cross-entropy loss)
#
# Steps per sample:
#   1. Compute g_i via vjp  (one backward pass)
#   2. scalar = g_i^T v
#   3. contribution = scalar * g_i
#
# This is cheaper than GGN — only one vjp per sample, no jvp needed.

def make_fisher_matvec_fast(
    model: nn.Module,
    params_flat: torch.Tensor,
    device: torch.device,
) -> Callable:
    """
    Returns  batch_matvec(x, y, v) -> (sum_i g_i (g_i^T v),  N)
    for a single batch (x, y).
    """
    params_dict = params_to_dict(model, params_flat)

    def loss_single(params, xi, yi):
        """Cross-entropy loss for one sample."""
        logits = functional_call(model, params, (xi.unsqueeze(0),)).squeeze(0)
        return F.cross_entropy(logits.unsqueeze(0), yi.unsqueeze(0))

    def fisher_single(xi, yi, v_flat):
        """
        Per-sample Fisher-vector product  g_i (g_i^T v).
        Returns a cotangent dict with the same tree as params_dict.
        """
        # g_i = d loss_i / d params  via vjp
        _, vjp_fn = vjp(lambda p: loss_single(p, xi, yi), params_dict)
        grad_dict = vjp_fn(torch.ones((), device=xi.device))[0]  # cotangent dict

        # flatten g_i
        g_flat = torch.cat([v.reshape(-1) for v in grad_dict.values()])

        # scalar projection
        scalar = g_flat @ v_flat   # g_i^T v

        # contribution: scalar * g_i  (back as dict)
        result_dict = {k: scalar * grad_dict[k] for k in grad_dict}
        return result_dict

    def batch_matvec(x: torch.Tensor, y: torch.Tensor,
                     v: torch.Tensor) -> Tuple[torch.Tensor, int]:
        N = x.shape[0]

        result_dict = vmap(lambda xi, yi: fisher_single(xi, yi, v))(x, y)
        result_flat = torch.cat([
            val.sum(0).reshape(-1) for val in result_dict.values()
        ])
        return result_flat, N

    return batch_matvec


def make_full_dataset_matvec(
    model: nn.Module,
    loader,
    params_flat: torch.Tensor,
    device: torch.device,
) -> Callable:
    batch_mv = make_fisher_matvec_fast(model, params_flat, device)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        result  = torch.zeros_like(v)
        n_total = 0
        for x, y in loader:                          # now we use labels y
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            br, n    = batch_mv(x, y, v)
            result  += br
            n_total += n
        return result / n_total

    return matvec


# ── Lanczos with periodic reorthogonalisation ─────────────────────────────

def lanczos(
    matvec: Callable,
    dim: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    reorth_freq: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(dim, device=device, dtype=dtype)
    q /= q.norm()

    Q      = torch.zeros(dim, rank, device=device, dtype=dtype)
    alphas = torch.zeros(rank,      device=device, dtype=dtype)
    betas  = torch.zeros(rank - 1,  device=device, dtype=dtype)
    q_prev = torch.zeros_like(q)

    for j in range(rank):
        Q[:, j] = q
        z = matvec(q)

        alpha     = q @ z
        alphas[j] = alpha
        z         = z - alpha * q - (betas[j - 1] * q_prev if j > 0 else 0.0)

        if (j + 1) % reorth_freq == 0:
            for _ in range(2):
                for k in range(j + 1):
                    z -= (Q[:, k] @ z) * Q[:, k]

        if j < rank - 1:
            beta = z.norm()
            if beta < 1e-10:
                print(f"Lanczos: early termination at step {j}")
                rank   = j + 1
                Q      = Q[:, :rank]
                alphas = alphas[:rank]
                betas  = betas[:j]
                break
            betas[j] = beta
            q_prev   = q
            q        = z / beta

        print(f"\r  Lanczos step {j+1}/{rank}  (beta={beta.item():.2e})", end="", flush=True)

    print()
    T = torch.diag(alphas) + torch.diag(betas, 1) + torch.diag(betas, -1)
    eigvals, eigvecs_T = torch.linalg.eigh(T)
    eigvecs = Q @ eigvecs_T

    return eigvals, eigvecs


# ── Low-rank Laplace posterior sampling ───────────────────────────────────

def laplace_sample(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    alpha: float,
    n_samples: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample from  N(0, (F + alpha I)^{-1})  via low-rank correction:

        sample = (1/sqrt(alpha)) * eps
               + V diag(1/sqrt(lambda+alpha) - 1/sqrt(alpha)) V^T eps

    Returns: [P, n_samples]
    """
    device    = eigvecs.device
    gen       = torch.Generator(device=device).manual_seed(seed)
    eigvals_c = eigvals.clamp(min=1e-7)
    diag_corr = (eigvals_c + alpha).rsqrt() - alpha ** -0.5

    eps        = torch.randn(eigvecs.shape[0], n_samples,
                             device=device, generator=gen)
    Vt_eps     = eigvecs.T @ eps
    correction = eigvecs @ (diag_corr.unsqueeze(1) * Vt_eps)

    return alpha ** -0.5 * eps + correction


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Model ──
    model = LeNet(output_dim=10, activation="tanh").to(device)
    model.load_state_dict(torch.load("adam_map.pth", map_location=device,
                                     weights_only=True))
    model.eval()

    # ── Data ──
    from mnist import get_mnist_loaders
    train_loader, _ = get_mnist_loaders(test_batch_size=128)

    P = num_params(model)
    print(f"Number of parameters: {P:,}")

    params_flat = get_params_vector(model).to(device)

    # ── Benchmark one matvec ──
    matvec = make_full_dataset_matvec(model, train_loader, params_flat, device)
    v_test = torch.randn(P, device=device)
    print("\nBenchmarking one matvec...")
    t0 = time.time()
    _ = matvec(v_test)
    t_matvec = time.time() - t0
    rank = 2000
    print(f"  One matvec : {t_matvec:.1f}s")
    print(f"  Estimated total for rank={rank}: {t_matvec * rank / 60:.0f} min")

    # ── Lanczos ──
    print(f"\n── Lanczos (rank={rank}) ──")
    t0 = time.time()
    eigvals, eigvecs = lanczos(matvec, P, rank, device, reorth_freq=50)
    print(f"Lanczos done in {(time.time()-t0)/60:.1f} min")

    idx     = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    print("\nTop-10 Fisher eigenvalues:")
    for i, ev in enumerate(eigvals[:10].tolist()):
        print(f"  λ_{i+1:2d} = {ev:.6f}")

    # ── Save ──
    torch.save({"eigvals": eigvals.cpu(), "eigvecs": eigvecs.cpu()},
               "fisher_lanczos.pt")
    print("\nSaved to fisher_lanczos.pt")

    # ── Laplace samples ──
    alpha     = 1.0
    n_samples = 5
    samples   = laplace_sample(eigvals, eigvecs, alpha=alpha, n_samples=n_samples)
    print(f"\nSample shape : {samples.shape}")
    print(f"Sample norms : {samples.norm(dim=0).tolist()}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, rank + 1), eigvals.cpu().numpy(), lw=1)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(f"Fisher Eigenspectrum — LeNet/MNIST  (rank={rank})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fisher_spectrum.png", dpi=150)
    print("\nPlot saved to fisher_spectrum.png")