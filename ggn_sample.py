"""
Load precomputed GGN eigvals/eigvecs and draw Laplace posterior samples.

Usage:
    python sample_laplace.py --ckpt ggn_lanczos.pt --map adam_map.pth --alpha 1.0 --n_samples 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


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
        if self.activation == "tanh": return torch.tanh(x)
        if self.activation == "relu": return F.relu(x)
        raise ValueError(f"Unknown activation {self.activation}")

    def forward(self, x):
        if x.dim() != 4: x = x.unsqueeze(0)
        x = F.max_pool2d(self.act_fun(self.conv1(x)), 2, 2)
        x = F.max_pool2d(self.act_fun(self.conv2(x)), 2, 2)
        x = torch.flatten(x, 1)
        x = self.act_fun(self.fc1(x))
        x = self.act_fun(self.fc2(x))
        return self.fc3(x)


# ── Utilities ──────────────────────────────────────────────────────────────

def get_params_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def set_params_from_vector(model: nn.Module, flat: torch.Tensor):
    """Load a flat parameter vector back into a model in-place."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].reshape(p.shape))
        offset += n


# ── Laplace sampling ───────────────────────────────────────────────────────

def laplace_sample(
    eigvals: torch.Tensor,   # [rank]
    eigvecs: torch.Tensor,   # [P, rank]
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
    diag_corr = (eigvals_c + alpha).rsqrt() - alpha ** -0.5   # [rank]

    eps        = torch.randn(eigvecs.shape[0], n_samples,
                             device=device, generator=gen)     # [P, n_samples]
    Vt_eps     = eigvecs.T @ eps                               # [rank, n_samples]
    correction = eigvecs @ (diag_corr.unsqueeze(1) * Vt_eps)  # [P, n_samples]

    return alpha ** -0.5 * eps + correction                    # [P, n_samples]


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str,   default="ggn_lanczos.pt")
    parser.add_argument("--map",       type=str,   default="adam_map.pth")
    parser.add_argument("--alpha",     type=float, default=1.0)
    parser.add_argument("--n_samples", type=int,   default=10)
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--out",       type=str,   default="laplace_samples.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load eigvals/eigvecs ──
    print(f"Loading GGN decomposition from {args.ckpt} ...")
    ckpt    = torch.load(args.ckpt, map_location=device)
    eigvals = ckpt["eigvals"].to(torch.float32)   # [rank]
    eigvecs = ckpt["eigvecs"].to(torch.float32)   # [P, rank]
    rank    = eigvals.shape[0]
    P       = eigvecs.shape[0]
    print(f"  rank={rank},  P={P}")
    print(f"  λ_max={eigvals.max().item():.4f},  λ_min={eigvals.min().item():.6f}")

    # ── Load MAP parameters ──
    print(f"Loading MAP parameters from {args.map} ...")
    model = LeNet(output_dim=10, activation="tanh").to(device)
    model.load_state_dict(torch.load(args.map, map_location=device))
    model.eval()
    map_params = get_params_vector(model).to(device)   # [P]

    # ── Draw samples ──
    print(f"\nDrawing {args.n_samples} samples  (alpha={args.alpha}, seed={args.seed}) ...")
    perturbations = laplace_sample(
        eigvals, eigvecs,
        alpha=args.alpha,
        n_samples=args.n_samples,
        seed=args.seed,
    )   # [P, n_samples]

    # Posterior parameter samples = MAP + perturbation
    posterior_params = map_params.unsqueeze(1) + perturbations   # [P, n_samples]

    print(f"Perturbation norms : {perturbations.norm(dim=0).tolist()}")
    print(f"Posterior shape    : {posterior_params.shape}")

    # ── Save ──
    torch.save({
        "posterior_params": posterior_params.cpu(),   # [P, n_samples]
        "perturbations":    perturbations.cpu(),      # [P, n_samples]
        "map_params":       map_params.cpu(),         # [P]
        "alpha":            args.alpha,
        "rank":             rank,
    }, args.out)
    print(f"\nSaved {args.n_samples} posterior samples to {args.out}")
    print(f"\nTo load and use a sample:")
    print(f"  ckpt = torch.load('{args.out}')")
    print(f"  set_params_from_vector(model, ckpt['posterior_params'][:, i])")