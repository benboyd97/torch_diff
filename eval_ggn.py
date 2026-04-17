"""
Evaluate GGN Laplace posterior samples using the existing metrics pipeline.

Loads precomputed eigvals/eigvecs from ggn_lanczos.pt, draws samples,
saves them in the format expected by load_param_dicts, then runs evaluate_all.

Alpha rescaling
---------------
The GGN matvec divides by N (mean over dataset):
    G = (1/N) sum_i J_i^T H_i J_i

So the posterior precision matrix is  G + alpha*I  where alpha is relative
to a mean-reduced likelihood. If you trained with weight decay lambda under
a sum-reduced likelihood, the matching prior precision is:
    alpha_scaled = lambda   (NOT lambda * N)

But if alpha=1000 was working before and alpha=1 wasn't, it means the GGN
was effectively N times too small relative to your prior — i.e. the training
loss used reduction='sum'. We correct for this here by dividing alpha by N,
so that alpha=1 in this script corresponds to the same scale as alpha=1/N
in the GGN's coordinate system.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from mnist import get_mnist_loaders, get_fmnist_loaders
from LeNet import LeNet
from metrics import evaluate_all, make_functional, load_param_dicts, plot_metrics


# ── Laplace sampling ───────────────────────────────────────────────────────

def laplace_sample(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    alpha: float,
    n_samples: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample from  N(0, (G + alpha I)^{-1})  via low-rank correction:

        sample = (1/sqrt(alpha)) * eps
               + V diag(1/sqrt(lambda+alpha) - 1/sqrt(alpha)) V^T eps

    Returns: [P, n_samples]
    """
    device    = eigvecs.device
    gen       = torch.Generator(device=device).manual_seed(seed)
    eigvals_c = eigvals.clamp(min=1e-7)
    diag_corr = (eigvals_c + alpha).rsqrt() - alpha ** -0.5

    eps        = torch.randn(eigvecs.shape[0], n_samples, device=device, generator=gen)
    Vt_eps     = eigvecs.T @ eps
    correction = eigvecs @ (diag_corr.unsqueeze(1) * Vt_eps)

    return alpha ** -0.5 * eps + correction


def get_params_vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_params_from_vector(model, flat):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].reshape(p.shape))
        offset += n


def draw_and_save_samples(
    eigvals, eigvecs, map_params, model,
    alpha, n_samples, out_dir, seed=0,
):
    """
    Draw n_samples from the Laplace posterior and save each as a .pth
    state dict matching the format expected by load_param_dicts.
    Returns list of saved paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    perturbations    = laplace_sample(eigvals, eigvecs, alpha=alpha,
                                      n_samples=n_samples, seed=seed)
    posterior_params = map_params.unsqueeze(1) + perturbations  # [P, n_samples]

    paths = []
    for i in range(n_samples):
        set_params_from_vector(model, posterior_params[:, i])
        path = os.path.join(out_dir, f"laplace_alpha{alpha:.2e}_samp{i+1}.pth")
        torch.save(model.state_dict(), path)
        paths.append(path)

    set_params_from_vector(model, map_params)   # restore MAP
    return paths


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    _, mnist_loader  = get_mnist_loaders(test_batch_size=10000)
    _, fmnist_loader = get_fmnist_loaders(test_batch_size=10000)

    # ── Model + MAP ──
    model = LeNet(activation="tanh").to(device)
    model.load_state_dict(torch.load("adam_map.pth", map_location=device))
    model.eval()
    map_params = get_params_vector(model).to(device)

    # ── MAP baseline ──
    params_map, _ = make_functional(model)
    map_metrics = evaluate_all(model, params_map, mnist_loader, fmnist_loader, device)
    print("\n── MAP metrics ──")
    for k, v in map_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Load GGN decomposition ──
    print("\nLoading GGN decomposition from ggn_lanczos.pt ...")
    ckpt    = torch.load("ggn_lanczos.pt", map_location=device)
    eigvals = ckpt["eigvals"].to(torch.float32).to(device)
    eigvecs = ckpt["eigvecs"].to(torch.float32).to(device)
    print(f"  rank={eigvals.shape[0]},  P={eigvecs.shape[0]}")
    print(f"  lambda_max={eigvals.max().item():.4f}")


    # ── Sweep over alpha ──
    # alphas here are in the same units as your weight decay / prior precision.
    # We divide by N_train so they are consistent with the mean-normalised GGN.
    n_samples = 10
    alphas    = [1.0,1e1, 1e2,  1e3,1e4,1e5]
    results   = {}

    for alpha in alphas:
        print(f"\n── alpha={alpha} ──")

        out_dir = f"laplace_samples/alpha_{alpha}"
        paths = draw_and_save_samples(
            eigvals, eigvecs, map_params, model,
            alpha=alpha, n_samples=n_samples,
            out_dir=out_dir, seed=0,
        )

        stacked_params = load_param_dicts(paths, model, device)
        metrics = evaluate_all(model, stacked_params, mnist_loader, fmnist_loader, device)
        results[alpha] = metrics

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ── Plot ──
    metrics_names = list(map_metrics.keys())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_names):
        ys = [results[a][metric] for a in alphas]
        ax.plot(alphas, ys, "o-", label="Laplace (GGN)")
        ax.axhline(map_metrics[metric], linestyle="--", color="gray", label="MAP")
        ax.set_xscale("log")
        if metric == 'mnist_nll' or   metric == 'mnist_ece' or  metric == 'mnist_mce' or  metric == 'mnist_brier':
            ax.set_yscale('log')
        ax.set_xlabel("alpha")
        ax.set_title(metric)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("laplace_metrics.png", dpi=150)
    print("\nPlot saved to laplace_metrics.png")

    torch.save({"results": results, "map_metrics": map_metrics, "alphas": alphas},
               "laplace_results.pt")
    print("Results saved to laplace_results.pt")


if __name__ == "__main__":
    main()