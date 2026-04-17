"""
eval.py — metrics and evaluation utilities for LeNet on MNIST/FMNIST.
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap
import numpy as np
from sklearn.metrics import roc_auc_score


# ── Calibration / Brier ────────────────────────────────────────────────────

def get_calib(pys, y_true, M=10):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    preds = np.argmax(pys, axis=1)
    confs = np.max(pys, axis=1)

    bin_boundaries = np.linspace(0, 1, M + 1)
    conf_idxs = np.digitize(confs, bin_boundaries, right=True) - 1
    conf_idxs = np.clip(conf_idxs, 0, M - 1)

    accs_bin, confs_bin, nitems_bin = [], [], []

    for i in range(M):
        in_bin = (conf_idxs == i)
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            accs_bin.append(np.mean(preds[in_bin] == y_true[in_bin]))
            confs_bin.append(np.mean(confs[in_bin]))
            nitems_bin.append(n_in_bin)

    if not accs_bin:
        return 0.0, 0.0

    accs_bin    = np.array(accs_bin)
    confs_bin   = np.array(confs_bin)
    nitems_bin  = np.array(nitems_bin)
    weights     = nitems_bin / np.sum(nitems_bin)

    ECE = np.sum(np.abs(accs_bin - confs_bin) * weights)
    MCE = np.max(np.abs(accs_bin - confs_bin))

    return ECE, MCE


def get_brier_score(probs, y_true):
    return np.mean(np.sum((probs - y_true) ** 2, axis=1))


# ── Param utilities ────────────────────────────────────────────────────────

def make_functional(model):
    params  = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return params, buffers


def stack_params(param_list):
    return {k: torch.stack([p[k] for p in param_list]) for k in param_list[0]}


def load_param_dicts(paths, model, device):
    param_list = []
    for p in paths:
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        params, _ = make_functional(model)
        param_list.append({k: v.detach().clone() for k, v in params.items()})
    return stack_params(param_list)


# ── Forward abstraction ────────────────────────────────────────────────────

def make_forward(model, params):
    buffers      = dict(model.named_buffers())
    example_key  = next(iter(params))
    example_param = params[example_key]

    # MAP — param tensors have the same shape as the model's state dict
    if example_param.dim() == len(model.state_dict()[example_key].shape):
        def forward(x):
            return functional_call(model, (params, buffers), (x,))
        return forward, False

    # Ensemble — leading batch dimension over samples
    def fmodel(p, x):
        return functional_call(model, (p, buffers), (x,))

    def forward(x):
        return vmap(fmodel, in_dims=(0, None))(params, x)

    return forward, True


# ── MNIST metrics ──────────────────────────────────────────────────────────

def mnist_metrics(model, params, loader, device, num_classes=10):
    forward, is_ensemble = make_forward(model, params)

    all_probs, all_labels = [], []
    total_nll, total = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = forward(x)

            if is_ensemble:
                probs = torch.softmax(logits, dim=-1).mean(dim=0)
                log_probs = (
                    torch.logsumexp(F.log_softmax(logits, dim=-1), dim=0)
                    - torch.log(torch.tensor(logits.shape[0], dtype=torch.float, device=device))
                )
                nll = F.nll_loss(log_probs, y, reduction="sum")
            else:
                probs = torch.softmax(logits, dim=-1)
                nll   = F.cross_entropy(logits, y, reduction="sum")

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())
            total_nll += nll.item()
            total     += y.size(0)

    probs   = torch.cat(all_probs).numpy()
    y_true  = torch.cat(all_labels).numpy()
    y_onehot = np.eye(num_classes)[y_true]

    acc    = np.mean(np.argmax(probs, axis=1) == y_true)
    conf   = np.mean(np.max(probs, axis=1))
    nll    = total_nll / total
    ece, mce = get_calib(probs, y_onehot)
    brier  = get_brier_score(probs, y_onehot)

    return conf, nll, acc, ece, mce, brier


# ── FMNIST confidence ──────────────────────────────────────────────────────

def fmnist_conf(model, params, loader, device):
    forward, is_ensemble = make_forward(model, params)

    conf_sum, n_batches = 0.0, 0

    with torch.no_grad():
        for x, _ in loader:
            x      = x.to(device)
            logits = forward(x)
            probs  = torch.softmax(logits, dim=-1)
            if is_ensemble:
                probs = probs.mean(dim=0)
            conf_sum  += probs.max(dim=-1).values.mean().item()
            n_batches += 1

    return conf_sum / n_batches


# ── OOD AUROC ──────────────────────────────────────────────────────────────

def ood_auroc(model, params, mnist_loader, fmnist_loader, device):
    forward, is_ensemble = make_forward(model, params)

    scores, labels = [], []

    with torch.no_grad():
        for x, _ in mnist_loader:
            x      = x.to(device)
            logits = forward(x)
            probs  = torch.softmax(logits, dim=-1)
            if is_ensemble:
                probs = probs.mean(dim=0)
            scores.append((1 - probs.max(dim=-1).values).cpu())
            labels.append(torch.zeros(x.size(0)))

        for x, _ in fmnist_loader:
            x      = x.to(device)
            logits = forward(x)
            probs  = torch.softmax(logits, dim=-1)
            if is_ensemble:
                probs = probs.mean(dim=0)
            scores.append((1 - probs.max(dim=-1).values).cpu())
            labels.append(torch.ones(x.size(0)))

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    return roc_auc_score(labels, scores)


# ── Unified evaluation ─────────────────────────────────────────────────────

def evaluate_all(model, params, mnist_loader, fmnist_loader, device):
    conf, nll, acc, ece, mce, brier = mnist_metrics(
        model, params, mnist_loader, device
    )
    f_conf = fmnist_conf(model, params, fmnist_loader, device)
    auroc  = ood_auroc(model, params, mnist_loader, fmnist_loader, device)

    return {
        "mnist_conf":  conf,
        "mnist_nll":   nll,
        "mnist_acc":   acc,
        "mnist_ece":   ece,
        "mnist_mce":   mce,
        "mnist_brier": brier,
        "fmnist_conf": f_conf,
        "auroc":       auroc,
    }

# ── Plotting ───────────────────────────────────────────────────────────────

def plot_metrics(results, map_metrics, alphas, top_ks):
    import matplotlib.pyplot as plt
    metrics_names = list(map_metrics.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_ks)))
    print(metrics_names)
    for metric in metrics_names:
        plt.figure()
        for i, top_k in enumerate(top_ks):
            ys = [results[top_k][alpha][metric] for alpha in alphas]
            plt.plot(alphas, ys, label=f"top_k={top_k}", color=colors[i])
            plt.scatter(alphas, ys, color=colors[i])
        plt.axhline(map_metrics[metric], linestyle='--', label='MAP')
        plt.xscale("log")
        if metric == 'mnist_nll' or   metric == 'mnist_ece' or  metric == 'mnist_mce' or  metric == 'mnist_brier':
            plt.yscale('log')
        plt.xlabel("alpha")
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.tight_layout()
        plt.show()