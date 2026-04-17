"""
Block-diagonal empirical Fisher for LeNet.

For each layer l, computes the block:
    F_l = (1/N) sum_i g_{l,i} g_{l,i}^T

where g_{l,i} is the per-sample gradient of the cross-entropy loss
w.r.t. the parameters of layer l, flattened to a vector.

Eigenvalues of each block are computed and saved to fisher_block_eigvals.pt.
Eigenvectors are discarded.

Block sizes for LeNet (44,426 params total):
    conv1.weight : 6*1*5*5   = 150
    conv1.bias   : 6         = 6
    conv2.weight : 16*6*5*5  = 2400
    conv2.bias   : 16        = 16
    fc1.weight   : 120*256   = 30720
    fc1.bias     : 120       = 120
    fc2.weight   : 84*120    = 10080
    fc2.bias     : 84        = 84
    fc3.weight   : 10*84     = 840
    fc3.bias     : 10        = 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from typing import Dict
import time


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


# ── Per-sample gradients via vmap + grad ───────────────────────────────────

def compute_block_fisher_eigvals(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Computes eigenvalues of each per-layer Fisher block.

    Returns dict: {layer_name: eigvals tensor (sorted descending)}
    """
    params_dict = {n: p.detach() for n, p in model.named_parameters()}
    layer_names = list(params_dict.keys())

    # Accumulate outer products per block: F_l += g_{l,i} g_{l,i}^T
    # For large blocks (fc1.weight: 30720x30720 ~ 3.6GB) we can't store the
    # full block — instead accumulate as a sum of rank-1 updates efficiently
    # using the fact that eigh can be computed from the accumulated matrix.
    # For fc1.weight this is 30720^2 * 4 bytes ~ 3.6GB — marginal but feasible
    # on a GPU with enough VRAM. If OOM, reduce batch size or use diagonal only.

    block_accum = {
        name: torch.zeros(p.numel(), p.numel(), device=device)
        for name, p in model.named_parameters()
    }
    n_total = 0

    def loss_fn(params, xi, yi):
        logits = functional_call(model, params, (xi.unsqueeze(0),)).squeeze(0)
        return F.cross_entropy(logits.unsqueeze(0), yi.unsqueeze(0))

    # per-sample grad function
    grad_fn = grad(loss_fn)

    print("Computing block-diagonal Fisher...")
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        N = x.shape[0]

        # vmap over batch for per-sample grads
        per_sample_grads = vmap(lambda xi, yi: grad_fn(params_dict, xi, yi))(x, y)
        # per_sample_grads: dict {name: [N, *param_shape]}

        for name in layer_names:
            g = per_sample_grads[name]          # [N, *param_shape]
            g_flat = g.reshape(N, -1)           # [N, P_l]
            # F_l += g_flat^T g_flat  (sum of outer products = G^T G)
            block_accum[name] += g_flat.T @ g_flat

        n_total += N
        print(f"\r  Batch {batch_idx+1}  ({n_total} samples)", end="", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")

    # Normalise and compute eigenvalues per block
    eigvals_dict = {}
    for name in layer_names:
        F_l = block_accum[name] / n_total
        P_l = F_l.shape[0]
        print(f"  {name:20s}  block size={P_l}x{P_l} ...", end=" ", flush=True)
        t1 = time.time()
        # cuSOLVER fails on large matrices — move to CPU for eigdecomp
        F_l_cpu = F_l.cpu()
        del F_l
        eigvals = torch.linalg.eigvalsh(F_l_cpu)      # sorted ascending
        eigvals = eigvals.flip(0)                      # descending
        eigvals_dict[name] = eigvals.cpu()
        print(f"λ_max={eigvals[0].item():.4e}  λ_min={eigvals[-1].item():.4e}"
              f"  ({time.time()-t1:.1f}s)")

    return eigvals_dict


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LeNet(output_dim=10, activation="tanh").to(device)
    model.load_state_dict(torch.load("adam_map.pth", map_location=device,
                                     weights_only=True))
    model.eval()

    from mnist import get_mnist_loaders
    train_loader, _ = get_mnist_loaders(test_batch_size=64)

    eigvals_dict = compute_block_fisher_eigvals(model, train_loader, device)

    # Print summary
    print("\n── Block Fisher eigenvalue summary ──")
    for name, eigvals in eigvals_dict.items():
        print(f"  {name:20s}  shape={list(eigvals.shape)}  "
              f"λ_max={eigvals[0].item():.4e}  λ_min={eigvals[-1].item():.4e}")

    # Save eigenvalues only
    torch.save(eigvals_dict, "fisher_block_eigvals.pt")
    print("\nSaved to fisher_block_eigvals.pt")
    print("Load with: eigvals_dict = torch.load('fisher_block_eigvals.pt', weights_only=False)")