"""
Evaluate symo samples and compare against GGN Laplace results.

Symo samples are saved as:
    symo_ggn/symo_alph{alph}_topk{top_k}_samp{s}.pth
for alph in alphas, top_k in top_ks, s in range(10).
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from mnist import get_mnist_loaders, get_fmnist_loaders
from LeNet import LeNet
from metrics import evaluate_all, make_functional, load_param_dicts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    _, mnist_loader  = get_mnist_loaders(test_batch_size=10000)
    _, fmnist_loader = get_fmnist_loaders(test_batch_size=10000)

    # ── Model + MAP baseline ──
    model = LeNet(activation="tanh").to(device)
    model.load_state_dict(torch.load("adam_map.pth", map_location=device, weights_only=True))
    model.eval()

    params_map, _ = make_functional(model)
    map_metrics = evaluate_all(model, params_map, mnist_loader, fmnist_loader, device)
    print("\n── MAP metrics ──")
    for k, v in map_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Load existing GGN Laplace results ──
    print("\nLoading GGN Laplace results from laplace_results.pt ...")
    laplace_ckpt   = torch.load("laplace_results.pt", map_location="cpu", weights_only=False)
    laplace_results = laplace_ckpt["results"]
    laplace_alphas  = laplace_ckpt["alphas"]
    print(laplace_results)

    # ── Sweep symo samples ──
    alphas = [1e-4, 1e-3, 1e-2]
    top_ks = [0.05,1.0]
    n_samps = 10
    shift = True
    diff = False

    symo_results = {top_k: {} for top_k in top_ks}

    for top_k in top_ks:
        for alph in alphas:
            print(f"\n── symo  top_k={top_k}  alpha={alph} ──")
            if diff:
                paths = [
                f"symo_diff/symo_alph{alph}_topk{top_k}_samp{s}.pth"
                for s in range(n_samps)
            ]
            else:
                paths = [
                f"symo_ggn/symo_alph{alph}_topk{top_k}_samp{s}.pth"
                for s in range(n_samps)
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

    # ── Plot: symo top_ks vs GGN Laplace, one panel per metric ──
    metrics_names    = list(map_metrics.keys())
    linear_y_metrics = {"mnist_conf", "mnist_acc", "fmnist_conf", "auroc"}

    n_metrics = len(metrics_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    # colour maps
    symo_colors   = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_ks)))
    laplace_color = "steelblue"

    for ax, metric in zip(axes, metrics_names):
        # GGN Laplace line
        laplace_ys = [laplace_results[a][metric] for a in laplace_alphas
                      if a in laplace_results]
        if shift:
            ax.plot(np.array(laplace_alphas[:len(laplace_ys)])/1e3, laplace_ys,
                "o-", color=laplace_color, label="GGN Laplace", lw=2)
        else:
            ax.plot(laplace_alphas[:len(laplace_ys)], laplace_ys,
                    "o-", color=laplace_color, label="GGN Laplace", lw=2)

        # Symo lines — one per top_k
        for i, top_k in enumerate(top_ks):
            if not symo_results[top_k]:
                continue
            valid_alphas = [a for a in alphas if a in symo_results[top_k]]
            ys = [symo_results[top_k][a][metric] for a in valid_alphas]
            if diff:
                lab = f"symo diff k={top_k}"
            else:
                lab = f"symo k={top_k}"
            if shift:
                ax.plot(np.array(valid_alphas)*1000, ys, "s--", color=symo_colors[i],
                        label=lab, lw=1.5)
            else:
                ax.plot(valid_alphas, ys, "s--", color=symo_colors[i],
                        label=lab, lw=1.5)

        # MAP baseline
        ax.axhline(map_metrics[metric], linestyle=":", color="gray",
                   label="MAP", lw=1.5)

        ax.set_xscale("log")
        if metric not in linear_y_metrics:
            ax.set_yscale("log")
        ax.set_xlabel("alpha")
        ax.set_title(metric)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    plt.suptitle("GGN Laplace vs Symo", fontsize=13, y=1.01)
    plt.tight_layout()

    name_ = "comparison_metrics.png"
    if shift:
        name_ = 'shift_'+name_
    if diff:
        name_ = 'diff_'+name_
    
    plt.savefig(name_, dpi=150, bbox_inches="tight")


    print("\nPlot saved to "+name_)

    # ── Save combined results ──
    torch.save({
        "symo_results":    symo_results,
        "laplace_results": laplace_results,
        "map_metrics":     map_metrics,
        "laplace_alphas":  laplace_alphas,
        "symo_alphas":     alphas,
        "top_ks":          top_ks,
    }, "comparison_results.pt")
    print("Results saved to comparison_results.pt")


if __name__ == "__main__":
    main()