# %%
"""
Variational Inference training for LeNet on MNIST using symo_diff's
GGN-based posterior covariance.

How this works
--------------
symo_diff (the Laplace sampler) does the following each step:
  1. Accumulate GGN surrogate:  F += grad @ grad.T  (outer product in group space)
  2. Decompose:                  F = U S U^T  (eigh)
  3. Add prior:                  s_i <- s_i + alpha
  4. Compute posterior cov sqrt: C = U (s + alpha)^{-1/2} U^T
  5. Sample:                     delta_w = C @ eps,  eps ~ N(0, I)

Here we do exactly the same thing, but inside an ELBO training loop:
  - mu  = model weights  (the variational mean, optimised by Adam)
  - C   = GGN-derived covariance (rebuilt each step, not learned)
  - q(w) = N(mu, C C^T)
  - p(w) = N(0, (1/alpha) I)

ELBO = E_q[log p(y|x,w)] - KL(q || p)

KL is computed analytically using the eigenvalues of the GGN:
  Let s_i = (GGN eigenvalues + alpha).  Then:
    ||C||_F^2     = sum(1/s_i)          [trace of posterior covariance]
    log det(C C^T) = -sum(log(s_i))
  So:
    KL = 0.5 * ( alpha * sum(1/s_i) + alpha * ||mu||^2 - d + sum(log(s_i)) - d*log(alpha) )

Settings: top_k = 1.0, prior_alpha = 1e-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

from mnist import get_mnist_loaders
from LeNet import LeNet
from symo.factory2 import GroupsSpec, CovFactory, groups_spec

# ── Config ────────────────────────────────────────────────────────────────────
PRIOR_ALPHA  = 1e-3
TOP_K        = 1.0
NUM_EPOCHS   = 300
LR           = 1e-3
BATCH_SIZE   = 128
NUM_PRED     = 20     # MC samples at test time
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Group spec (identical to Laplace code) ────────────────────────────────────
In = "I_input"
G1 = "B_L1";  G2 = "I_L2";  G3 = "I_L3"
G4 = "B_L4";  G5 = "B_L5";  Ou = "I_output"
Ih = "I_H1";  Iw = "I_W1";  Ih2 = "I_H2";  Iw2 = "I_W2"

sizes = dict(input=1, L1=6, L2=16, L3=256, L4=120, L5=84, output=10,
             H1=5, W1=5, H2=5, W2=5)

groups_cfg = {
    "conv1.weight": (G1, In, Iw,  Ih),
    "conv1.bias":   (G1,),
    "conv2.weight": (G2, G1, Iw2, Ih2),
    "conv2.bias":   (G2,),
    "fc1.weight":   (G4, G3),
    "fc1.bias":     (G4,),
    "fc2.weight":   (G5, G4),
    "fc2.bias":     (G5,),
    "fc3.weight":   (Ou, G5),
    "fc3.bias":     (Ou,),
}


# ── GGN covariance tracker ────────────────────────────────────────────────────

class GGNCovarianceTracker:
    """
    Mirrors symo_diff's internal covariance logic exactly.

    Call order each step:
        reset()             — zero accumulated GGN
        accumulate(grads)   — outer_update: F += grad @ grad.T (in group space)
        build_sqrt_inv()    — eigh, shift by alpha, compute C = U s^{-1/2} U^T
        sample(params)      — draw delta_w = C @ eps
        kl(params)          — analytic KL(q || p)
    """

    def __init__(self, model: nn.Module, spec: GroupsSpec,
                 prior_alpha: float, top_k: float, device):
        self.prior_alpha = prior_alpha
        self.top_k       = top_k
        self.device      = device
        self.dtype       = next(model.parameters()).dtype   # store for later use

        dev_cfg = dict(device=device, dtype=self.dtype)
        self.cov_factory = CovFactory(spec, block_diag_only=False).to(**dev_cfg)

        # Set by build_sqrt_inv(); used by kl()
        self._eigenvalues: torch.Tensor | None = None   # s_i = GGN_eig + alpha
        self._d: int = 0                                # number of variational dims

    # ------------------------------------------------------------------

    def reset(self):
        for w in self.cov_factory.weights():
            w.zero_()

    def accumulate(self, grads: list[torch.Tensor]):
        """Step 1 in symo_diff: outer_update accumulates GGN surrogate."""
        self.cov_factory.outer_update(grads)

    def build_sqrt_inv(self, decomp_dtype=torch.float64):
        """
        Steps 2-4 in symo_diff:
          - retrieve surrogate matrix from cov_factory
          - eigh decomposition
          - select top_k eigenvectors (top_k=1.0 → keep all)
          - shift eigenvalues: s_i += alpha
          - compute and store C = U (s)^{-1/2} U^T back into cov_factory
        """
        orig_dtype = self.dtype

        surrogate = self.cov_factory.cov(surrogate=True,
                                         device=self.device,
                                         dtype=orig_dtype)

        if isinstance(surrogate, list):
            # Block-diagonal case
            decomps   = [self._decompose(s, decomp_dtype, orig_dtype) for s in surrogate]
            sqrt_invs = [self._inv_sqrt(u, s, vt) for u, s, vt in decomps]
            self._eigenvalues = torch.cat([s for _, s, _ in decomps])
            self._d = int(self._eigenvalues.numel())
            self.cov_factory.cov_block_diag_update(sqrt_invs, surrogate=True)
        else:
            u, s, vt = self._decompose(surrogate, decomp_dtype, orig_dtype)
            sqrt_inv = self._inv_sqrt(u, s, vt)
            self._eigenvalues = s
            self._d = int(s.numel())
            self.cov_factory.cov_update(sqrt_inv, surrogate=True)

    def _decompose(self, mat, decomp_dtype, orig_dtype):
        """Mirrors mat_decomp() in symo_diff exactly."""
        s_raw, u_raw = torch.linalg.eigh(mat.to(decomp_dtype))
        s_raw = s_raw.to(orig_dtype)
        u_raw = u_raw.to(orig_dtype)

        n = s_raw.shape[0]
        top = max(1, int(n * self.top_k))     # top_k=1.0 → all eigenvectors

        # eigh returns ascending order; take the top (largest) eigenvalues
        s  = s_raw[-top:] + self.prior_alpha  # ← shift: s_i + alpha  (symo_diff line)
        u  = u_raw[:, -top:]
        vt = u.T
        return u, s, vt

    @staticmethod
    def _inv_sqrt(u, s, vt):
        """Mirrors truncated_inv_sqrt() in symo_diff."""
        return (u * (1.0 / s.sqrt())[None]) @ vt

    def sample(self, params: list[nn.Parameter]) -> list[torch.Tensor]:
        """
        Step 5 in symo_diff: delta_w = C @ eps, eps ~ N(0, I).
        Returns a list of per-parameter perturbation tensors.
        """
        noise = [torch.randn_like(p) for p in params]
        return self.cov_factory.matvec(noise)   # applies C in group space

    def kl(self, params: list[nn.Parameter]) -> torch.Tensor:
        """
        Analytic KL( q || p ) where q = N(mu, C C^T), p = N(0, (1/alpha) I).

        With s_i = GGN eigenvalue + alpha:
            C C^T  has eigenvalues  1/s_i
            tr(C C^T) = sum(1/s_i)
            log det(C C^T) = -sum(log s_i)

        KL = 0.5 * [ alpha * tr(C C^T) + alpha * ||mu||^2 - d - log det(alpha * C C^T) ]
           = 0.5 * [ alpha * sum(1/s_i) + alpha * ||mu||^2 - d + sum(log s_i) - d*log(alpha) ]
        """
        assert self._eigenvalues is not None, "Call build_sqrt_inv() first"

        s     = self._eigenvalues
        d     = float(self._d)
        alpha = self.prior_alpha
        mu_sq = sum(p.pow(2).sum() for p in params)

        kl = 0.5 * (
            alpha * (1.0 / s).sum()                       # alpha * tr(Sigma_q)
            + alpha * mu_sq                               # alpha * ||mu||^2
            - d                                           # -d
            + s.log().sum()                               # log det(Sigma_q^{-1})
            - d * torch.tensor(alpha, device=s.device).log()  # -d*log(alpha)
        )
        return kl


# ── ELBO step ─────────────────────────────────────────────────────────────────

def elbo_step(model: nn.Module, cov: GGNCovarianceTracker,
              x: torch.Tensor, y: torch.Tensor,
              dataset_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One ELBO mini-batch step.

    Procedure:
      1. Forward+backward at mu to get gradients for GGN accumulation.
      2. Build posterior covariance C from GGN (symo_diff steps 1-4).
      3. Sample delta_w = C @ eps  (symo_diff step 5).
      4. Perturb: w = mu + delta_w.
      5. Evaluate NLL at w.
      6. Analytic KL.
      7. loss = NLL + KL * (batch/N).  Restore mu.  Return for .backward().

    Reparameterisation: gradients w.r.t. mu come from both the NLL at w
    (since w = mu + delta_w and delta_w is stop-grad) and the KL (alpha*||mu||^2).
    C itself is treated as a constant (detached GGN).
    """
    params     = list(model.parameters())
    batch_size = x.size(0)

    # ── 1. Gradients at mu for GGN ────────────────────────────────────────────
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    logits_mu = model(x)
    F.cross_entropy(logits_mu, y).backward()
    grads = [p.grad.detach().clone() for p in params]

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    # ── 2. Build C = U (GGN + alpha I)^{-1/2} U^T ────────────────────────────
    cov.reset()
    cov.accumulate(grads)
    cov.build_sqrt_inv()

    # ── 3-4. Sample and perturb ───────────────────────────────────────────────
    with torch.no_grad():
        delta = cov.sample(params)

    for p, d in zip(params, delta):
        p.data.add_(d)

    # ── 5. NLL at w = mu + delta ──────────────────────────────────────────────
    logits_w = model(x)
    nll = F.cross_entropy(logits_w, y)

    # ── 6-7. KL and ELBO loss ─────────────────────────────────────────────────
    kl        = cov.kl(params)
    kl_scaled = kl * (batch_size / dataset_size)
    loss      = nll + kl_scaled

    # Restore mu before returning (backward() will still work on the graph)
    for p, d in zip(params, delta):
        p.data.sub_(d)

    return loss, nll.detach(), kl_scaled.detach()


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_epoch(model, cov, loader, optimizer, dataset_size, device):
    model.train()
    total_loss = total_nll = total_kl = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, nll, kl = elbo_step(model, cov, x, y, dataset_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_nll  += nll.item()
        total_kl   += kl.item()
        n += 1

    return total_loss / n, total_nll / n, total_kl / n


@torch.no_grad()
def evaluate(model, cov, loader, device, num_samples=NUM_PRED):
    """
    MC predictive accuracy.  For each batch, rebuild the GGN-based covariance
    (same as at training time) then average softmax over `num_samples` draws.
    """
    correct = total = 0
    params  = list(model.parameters())

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Need a forward+backward to build GGN for sampling
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        model.train()
        with torch.enable_grad():
            F.cross_entropy(model(x), y).backward()
        grads = [p.grad.detach().clone() for p in params]
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        model.eval()

        cov.reset()
        cov.accumulate(grads)
        cov.build_sqrt_inv()

        probs = torch.zeros(x.size(0), 10, device=device)
        for _ in range(num_samples):
            delta = cov.sample(params)
            for p, d in zip(params, delta):
                p.data.add_(d)
            with torch.no_grad():
                probs += F.softmax(model(x), dim=-1)
            for p, d in zip(params, delta):
                p.data.sub_(d)

        correct += (probs.argmax(1) == y).sum().item()
        total   += y.size(0)

    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device:       {DEVICE}")
    print(f"Prior alpha:  {PRIOR_ALPHA}  (prior variance = {1/PRIOR_ALPHA:.0f})")
    print(f"Top-k:        {TOP_K}")

    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
    dataset_size = len(train_loader.dataset)

    model = LeNet(activation="tanh").to(DEVICE)
    try:
        model.load_state_dict(torch.load("adam_map.pth", map_location=DEVICE))
        print("Warm-start: loaded MAP from adam_map.pth")
    except FileNotFoundError:
        print("adam_map.pth not found — using random init")

    group_spec_ord = [groups_cfg[n] for n, _ in model.named_parameters()]
    spec = groups_spec(group_spec_ord, sizes)

    cov = GGNCovarianceTracker(model, spec,
                               prior_alpha=PRIOR_ALPHA,
                               top_k=TOP_K,
                               device=DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        loss, nll, kl = train_epoch(model, cov, train_loader, optimizer, dataset_size, DEVICE)

        if epoch % 10 == 0 or epoch == 1:
            acc = evaluate(model, cov, test_loader, DEVICE, num_samples=NUM_PRED)
            print(f"Epoch {epoch:3d} | loss {loss:.4f}  nll {nll:.4f}  kl {kl:.6f} "
                  f"| test acc {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "vi_best_mu.pth")

    print(f"\nBest test accuracy: {best_acc:.4f}")

    # ── Draw 10 posterior samples on full training set (mirrors your Laplace loop) ──
    print("\nBuilding final GGN on full dataset and drawing 10 samples...")
    full_loader, _ = get_mnist_loaders(batch_size=60000)
    cov.reset()

    model.train()
    with torch.enable_grad():
        for x, y in full_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            F.cross_entropy(model(x), y).backward()
            grads = [p.grad.detach().clone() for p in model.parameters()]
            cov.accumulate(grads)

    cov.build_sqrt_inv()
    params = list(model.parameters())

    for s in range(10):
        delta = cov.sample(params)
        for p, d in zip(params, delta):
            p.data.add_(d)
        torch.save(model.state_dict(),
                   f"vi_alph{PRIOR_ALPHA}_topk{TOP_K}_samp{s}.pth")
        for p, d in zip(params, delta):
            p.data.sub_(d)
        print(f"  Saved vi_alph{PRIOR_ALPHA}_topk{TOP_K}_samp{s}.pth")


if __name__ == "__main__":
    main()

# %%



