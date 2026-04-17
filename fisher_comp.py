"""
Full comparison: fisher Laplace vs Symo samples.

Runs fisher Laplace evaluation from scratch, then loads symo samples,
evaluates both, and plots all metrics side by side.
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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    _, mnist_loader  = get_mnist_loaders(test_batch_size=10000)
    _, fmnist_loader = get_fmnist_loaders(test_batch_size=10000)

    # ── Model + MAP ──
    model = LeNet(activation="tanh").to(device)
    model.load_state_dict(torch.load("adam_map.pth", map_location=device, weights_only=True))
    model.eval()
    map_params = get_params_vector(model).to(device)

    params_map, _ = make_functional(model)
    map_metrics = evaluate_all(model, params_map, mnist_loader, fmnist_loader, device)
    print("\n── MAP metrics ──")
    for k, v in map_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── fisher Laplace ───────────────────────────────────────────────────────
    print("\nLoading fisher decomposition from fisher_lanczos.pt ...")
    ckpt    = torch.load("fisher_lanczos.pt", map_location=device, weights_only=False)
    eigvals = ckpt["eigvals"].to(torch.float32).to(device)
    eigvecs = ckpt["eigvecs"].to(torch.float32).to(device)
    print(f"  rank={eigvals.shape[0]},  P={eigvecs.shape[0]},  lambda_max={eigvals.max().item():.4f}")

    n_samples     = 10
    laplace_alphas = [10.0, 100.0, 1000.0,1e4]
    laplace_results = {}

    for alpha in laplace_alphas:
        print(f"\n── fisher Laplace  alpha={alpha} ──")
        out_dir = f"laplace_samples/alpha_{alpha}"
        paths = draw_and_save_samples(
            eigvals, eigvecs, map_params, model,
            alpha=alpha, n_samples=n_samples, out_dir=out_dir, seed=0,
        )
        stacked_params = load_param_dicts(paths, model, device)
        metrics = evaluate_all(model, stacked_params, mnist_loader, fmnist_loader, device)
        laplace_results[alpha] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ── Symo ──────────────────────────────────────────────────────────────
    symo_alphas = [1e-4, 1e-3, 1e-2, 1e-1]
    top_ks      = [0.05, 1.0]
    symo_results = {top_k: {} for top_k in top_ks}

    for top_k in top_ks:
        for alph in symo_alphas:
            print(f"\n── Symo  top_k={top_k}  alpha={alph} ──")
            paths = [
                f"symo_ggn/symo_alph{alph}_topk{top_k}_samp{s}.pth"
                for s in range(10)
            ]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                print(f"  WARNING: {len(missing)} files missing, skipping")
                continue
            stacked_params = load_param_dicts(paths, model, device)
            metrics = evaluate_all(model, stacked_params, mnist_loader, fmnist_loader, device)
            symo_results[top_k][alph] = metrics
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save({
        "laplace_results": laplace_results,
        "laplace_alphas":  laplace_alphas,
        "symo_results":    symo_results,
        "symo_alphas":     symo_alphas,
        "top_ks":          top_ks,
        "map_metrics":     map_metrics,
    }, "comparison_results.pt")
    print("\nSaved to comparison_results.pt")

    # ── Plot ──────────────────────────────────────────────────────────────
    metrics_names    = list(map_metrics.keys())
    linear_y_metrics = {"mnist_conf", "mnist_acc", "fmnist_conf", "auroc"}

    symo_colors   = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_ks)))
    laplace_color = "steelblue"

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_names):
        # fisher Laplace
        ys = [laplace_results[a][metric] for a in laplace_alphas]
        ax.plot(np.array(laplace_alphas)/1000, ys, "o-", color=laplace_color,
                label="fisher Laplace", lw=2)

        # Symo — one line per top_k
        for i, top_k in enumerate(top_ks):
            valid_alphas = [a for a in symo_alphas if a in symo_results[top_k]]
            if not valid_alphas:
                continue
            ys = [symo_results[top_k][a][metric] for a in valid_alphas]
            
            ax.plot(np.array(valid_alphas)*1000, ys, "s--", color=symo_colors[i],
                    label=f"Symo k={top_k}", lw=1.5)

        # MAP
        ax.axhline(map_metrics[metric], linestyle=":", color="gray",
                   label="MAP", lw=1.5)

        ax.set_xscale("log")
        if metric not in linear_y_metrics:
            ax.set_yscale("log")
        ax.set_xlabel("alpha")
        ax.set_title(metric)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(metrics_names):]:
        ax.set_visible(False)

    plt.suptitle("fisher Laplace vs Symo", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("fisher_comparison_metrics.png", dpi=150, bbox_inches="tight")
    print("Plot saved to comparison_metrics.png")


if __name__ == "__main__":
    main()