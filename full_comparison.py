"""
Full comparison:
  - GGN Laplace    (from ggn_lanczos.pt)
  - Fisher Laplace (from fisher_lanczos.pt)
  - Symo GGN       (from symo_ggn/)
  - Symo Diff      (from symo_diff/)

All using top_k=1.0.

Plotting convention (to put all methods on a common x-axis scale):
  - Lanczos alphas  : divide by 1000 for plot
  - Symo alphas     : multiply by 1000 for plot
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
from metrics import evaluate_all, make_functional, load_param_dicts


# ── Laplace sampling ───────────────────────────────────────────────────────

def laplace_sample(eigvals, eigvecs, alpha, n_samples=1, seed=0):
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


def draw_and_save_samples(eigvals, eigvecs, map_params, model,
                          alpha, n_samples, out_dir, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    perturbations    = laplace_sample(eigvals, eigvecs, alpha=alpha,
                                      n_samples=n_samples, seed=seed)
    posterior_params = map_params.unsqueeze(1) + perturbations
    paths = []
    for i in range(n_samples):
        set_params_from_vector(model, posterior_params[:, i])
        path = os.path.join(out_dir, f"laplace_alpha{alpha:.2e}_samp{i+1}.pth")
        torch.save(model.state_dict(), path)
        paths.append(path)
    set_params_from_vector(model, map_params)
    return paths


def run_lanczos_eval(name, ckpt_path, model, map_params, mnist_loader,
                     fmnist_loader, device, alphas, n_samples=10):
    """Load eigvals/eigvecs, draw samples, evaluate across alphas."""
    print(f"\nLoading {name} decomposition from {ckpt_path} ...")
    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    eigvals = ckpt["eigvals"].to(torch.float32).to(device)
    eigvecs = ckpt["eigvecs"].to(torch.float32).to(device)
    print(f"  rank={eigvals.shape[0]},  P={eigvecs.shape[0]},  "
          f"lambda_max={eigvals.max().item():.4f}")

    results = {}
    for alpha in alphas:
        print(f"\n── {name}  alpha={alpha} ──")
        out_dir = f"laplace_samples/{name}_alpha_{alpha}"
        paths = draw_and_save_samples(
            eigvals, eigvecs, map_params, model,
            alpha=alpha, n_samples=n_samples, out_dir=out_dir, seed=0,
        )
        stacked_params = load_param_dicts(paths, model, device)
        metrics = evaluate_all(model, stacked_params, mnist_loader, fmnist_loader, device)
        results[alpha] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    return results


def run_symo_eval(name, folder, model, mnist_loader, fmnist_loader,
                  device, alphas, top_k=1.0, n_samps=10):
    """Load symo samples from folder, evaluate across alphas."""
    results = {}
    for alph in alphas:
        print(f"\n── {name}  alpha={alph} ──")

        paths = [
                f"{folder}/symo_alph{alph}_topk{top_k}_samp{s}.pth"
                for s in range(n_samps)
            ]

        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"  WARNING: {len(missing)} files missing, skipping")
            continue
        stacked_params = load_param_dicts(paths, model, device)
        metrics = evaluate_all(model, stacked_params, mnist_loader, fmnist_loader, device)
        results[alph] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    _, mnist_loader  = get_mnist_loaders(test_batch_size=10000)
    _, fmnist_loader = get_fmnist_loaders(test_batch_size=10000)

    # ── Model + MAP ──
    model = LeNet(activation="tanh").to(device)
    model.load_state_dict(torch.load("adam_map.pth", map_location=device,
                                     weights_only=True))
    model.eval()
    map_params = get_params_vector(model).to(device)

    params_map, _ = make_functional(model)
    map_metrics = evaluate_all(model, params_map, mnist_loader, fmnist_loader, device)
    print("\n── MAP metrics ──")
    for k, v in map_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Alpha grids ──
    lanczos_alphas = [10.0, 100.0, 1000.0, 1e4]
    symo_alphas    = [1e-5, 1e-4, 1e-3, 1e-2]
    top_k          = 1.0
    n_samples      = 10

    # ── Run evaluations ──
    ggn_results    = run_lanczos_eval("GGN",    "ggn_lanczos.pt",    model, map_params,
                                      mnist_loader, fmnist_loader, device,
                                      lanczos_alphas, n_samples)

    fisher_results = run_lanczos_eval("Fisher", "fisher_lanczos.pt", model, map_params,
                                      mnist_loader, fmnist_loader, device,
                                      lanczos_alphas, n_samples)

    symo_ggn_results  = run_symo_eval("Symo-GGN",  "symo_ggn",  model,
                                       mnist_loader, fmnist_loader, device,
                                       symo_alphas, top_k, n_samples)

    symo_diff_results = run_symo_eval("Symo-Diff", "symo_diff", model,
                                       mnist_loader, fmnist_loader, device,
                                       symo_alphas, top_k, n_samples)

    # ── Save ──
    torch.save({
        "ggn_results":        ggn_results,
        "fisher_results":     fisher_results,
        "symo_ggn_results":   symo_ggn_results,
        "symo_diff_results":  symo_diff_results,
        "lanczos_alphas":     lanczos_alphas,
        "symo_alphas":        symo_alphas,
        "map_metrics":        map_metrics,
    }, "comparison_results.pt")
    print("\nSaved to comparison_results.pt")

    # ── Plot ──
    # x-axis convention:
    #   lanczos: alpha / 1000
    #   symo:    alpha * 1000
    metrics_names    = list(map_metrics.keys())
    linear_y_metrics = {"mnist_conf", "mnist_acc", "fmnist_conf", "auroc"}

    methods = [
        ("GGN Laplace",  ggn_results,        lanczos_alphas, 1/1000, "steelblue",  "o-"),
        ("Fisher Laplace", fisher_results,    lanczos_alphas, 1/1000, "darkorange", "o-"),
        ("Symo Laplace", symo_ggn_results,    symo_alphas,    1000,   "firebrick",  "s--"),
        ("Symo Diffusion", symo_diff_results, symo_alphas,    1000,   "seagreen",   "^--"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_names):
        for label, results, alphas, x_scale, color, style in methods:
            valid = [a for a in alphas if a in results]
            if not valid:
                continue
            xs = [a * x_scale for a in valid]
            ys = [results[a][metric] for a in valid]
            ax.plot(xs, ys, style, color=color, label=label, lw=1.8, ms=5)

        ax.axhline(map_metrics[metric], linestyle=":", color="gray",
                   label="MAP", lw=1.5)
        ax.set_xscale("log")
        if metric not in linear_y_metrics:
            ax.set_yscale("log")
        ax.set_xlabel("alpha (scaled)")
        ax.set_title(metric)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(metrics_names):]:
        ax.set_visible(False)

    plt.suptitle("GGN Laplace vs Fisher Laplace vs Symo Laplace vs Symo Diffusion",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("comparison_metrics.png", dpi=150, bbox_inches="tight")
    print("Plot saved to comparison_metrics.png")


if __name__ == "__main__":
    main()