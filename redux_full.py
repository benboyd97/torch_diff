# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dists
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from mnist import get_mnist_loaders
from LeNet import LeNet

from netcal.metrics import ECE, MCE

from laplace import Laplace

from dataclasses import dataclass, asdict
import copy
# ------------------------
# Training / Eval
# ------------------------
def train_epoch(model, loader, optimizer, criterion, device, max_iter = None, reg=False):
    if max_iter == None:
        max_iter = len(loader)
    model.train()
    total_loss = 0
    losses = []
    counter = 0

    for x, y in loader:
        if counter == max_iter:
            return total_loss / max_iter, losses
        counter += 1
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        if reg:
            l2_lambda = 1e-5
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)**2
            loss = criterion(logits, y) + l2_lambda * l2_reg
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses += [loss.item()]


    return total_loss / len(loader), losses


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

def print_model_params(model):
    total = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n

            shape = str(tuple(param.shape))  # <-- fix

            print(f"{name:20s} | shape: {shape:15s} | params: {n}")

    print("-" * 60)
    print(f"Total trainable params: {total}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet(activation="tanh").to(device)
model.load_state_dict(torch.load("adam_map.pth", map_location=device))

mnist_train_loader, mnist_test_loader = get_mnist_loaders(test_batch_size=10000)

mnist_targets = torch.cat([y for x, y in mnist_test_loader], dim=0)

import numpy as np
from sklearn.metrics import roc_auc_score
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
            bin_acc = np.mean(preds[in_bin] == y_true[in_bin])
            bin_conf = np.mean(confs[in_bin])

            accs_bin.append(bin_acc)
            confs_bin.append(bin_conf)
            nitems_bin.append(n_in_bin)

    if not accs_bin:
        return 0.0, 0.0

    accs_bin = np.array(accs_bin)
    confs_bin = np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    gaps = np.abs(accs_bin - confs_bin)
    weights = nitems_bin / np.sum(nitems_bin)

    ECE = np.sum(gaps * weights)
    MCE = np.max(gaps)

    return ECE, MCE

def get_brier_score(probs, y_true):
    return np.mean(np.sum((probs - y_true) ** 2, axis=1))


import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np
from sklearn.metrics import roc_auc_score
mnist_targets_tensor = torch.cat([y for x, y in mnist_test_loader]).cpu()

def move_laplace_to_cuda(la):
    """
    Safely moves any Laplace structure (Kron, LowRank, Diag, Full) to CUDA.
    """
    # 1. Move the model
    la.model.cuda()
    la.model.output_size = 10 
    
    # 2. Move the prior precision (stored as a tensor)
    if torch.is_tensor(la.prior_precision):
        la.prior_precision = la.prior_precision.cuda()
    
    # 3. Handle the Hessian (la.H)
    if hasattr(la, 'H') and la.H is not None:
        if torch.is_tensor(la.H): # Full or Diag
            la.H = la.H.cuda()
        elif isinstance(la.H, tuple): # LowRank (eigenvecs, eigenvals)
            la.H = tuple(h.cuda() if torch.is_tensor(h) else h for h in la.H)
        elif hasattr(la.H, 'to'): # KronDecomposed or other objects
            try:
                la.H.to('cuda')
            except:
                pass
    return la

# ==========================================
# 2. Prediction Function (Unified Laplace)
# ==========================================
@torch.no_grad()
def get_laplace_probs(dataloader, la, sample=True, n_samples=100):
    # Ensure everything is synced on GPU
    la = move_laplace_to_cuda(la)
    
    py = []
    for x, _ in dataloader:
        x_cuda = x.cuda()
        
        # This context manager forces internal torch.eye (the bug) to be on CUDA
        with torch.device('cuda'):
            if sample:
                # MC Sampling (NN path)
                logits_samples = la.predictive_samples(x_cuda, n_samples=n_samples, pred_type='nn')
                probs = logits_samples.mean(dim=0)
            else:
                # Linearised (GLM path)
                probs = la(x_cuda, pred_type='glm')
        
        py.append(probs.cpu())

    return torch.cat(py)

# ==========================================
# 3. Full Evaluation Suite
# ==========================================
def evaluate_laplace_full(la, mnist_loader, fmnist_loader, mnist_targets, sample=True, n_samples=100):
    # --- A. In-Distribution (MNIST) ---
    probs_mnist = get_laplace_probs(mnist_loader, la, sample=sample, n_samples=n_samples)

    print(probs_mnist.shape)
    
    # NLL Calculation

    
    nll = -dists.Categorical(probs=probs_mnist).log_prob(mnist_targets).mean().item()
    
    # Accuracy
    acc = (probs_mnist.argmax(-1) == mnist_targets.cpu()).float().mean().item()
    
    # Calibration & Brier (Using your provided numpy functions)
    probs_np = probs_mnist.numpy()
    y_true_np = mnist_targets.cpu().numpy()
    y_onehot = np.eye(10)[y_true_np]
    
    ece, mce = get_calib(probs_np, y_onehot)
    brier = get_brier_score(probs_np, y_onehot)
    mnist_conf = np.mean(np.max(probs_np, axis=1))

    # --- B. Out-of-Distribution (FashionMNIST) ---
    probs_fmnist = get_laplace_probs(fmnist_loader, la, sample=sample, n_samples=n_samples)
    fmnist_conf = np.mean(np.max(probs_fmnist.numpy(), axis=1))

    # --- C. OOD AUROC ---
    m_scores = 1 - np.max(probs_np, axis=1)
    f_scores = 1 - np.max(probs_fmnist.numpy(), axis=1)
    
    all_scores = np.concatenate([m_scores, f_scores])
    all_labels = np.concatenate([np.zeros(len(m_scores)), np.ones(len(f_scores))])
    auroc = roc_auc_score(all_labels, all_scores)

    return {
        "mnist_acc": acc, "mnist_nll": nll, "mnist_ece": ece, "mnist_mce": mce,
        "mnist_brier": brier, "mnist_conf": mnist_conf, "fmnist_conf": fmnist_conf, 
        "auroc": auroc
    }


def cal_plot(la, mnist_loader, M=10, n_samples=10, sample=True):
    """
    Dedicated Reliability Diagram for Laplace (Sampled or Linearised).
    """
    # 1. Get Marginalised Probs [N, 10] and True Labels [N]
    # Uses the unified get_laplace_probs we built (handles GPU/Kron/LowRank)
    probs_lp_tensor = get_laplace_probs(mnist_loader, la, sample=sample, n_samples=n_samples)
    probs_lp = probs_lp_tensor.numpy()
    
    # Extract labels from loader
    y_true = torch.cat([y for x, y in mnist_loader]).cpu().numpy()
    
    # 2. Calculate Confidence and Predictions
    confs_lp = np.max(probs_lp, axis=1)
    preds_lp = np.argmax(probs_lp, axis=1)

    bins = np.linspace(0, 1, M + 1)
    # Digitize: assign each confidence to a bin index
    idx_lp = np.clip(np.digitize(confs_lp, bins, right=True) - 1, 0, M-1)
    
    accs_bin = []
    confs_bin = []
    counts_bin = []

    for i in range(M):
        mask = (idx_lp == i)
        if np.any(mask):
            acc_bin = np.mean(preds_lp[mask] == y_true[mask])
            conf_bin = np.mean(confs_lp[mask])
            count = np.sum(mask)
        else:
            acc_bin = np.nan
            conf_bin = (bins[i] + bins[i+1]) / 2
            count = 0
            
        accs_bin.append(acc_bin)
        confs_bin.append(conf_bin)
        counts_bin.append(count)

    return confs_bin,accs_bin 

from mnist import get_mnist_loaders, get_fmnist_loaders
_, mnist_loader = get_mnist_loaders(test_batch_size=128)
_, fmnist_loader = get_fmnist_loaders(test_batch_size=128)

hess_type = ['kron']
samp_type = ['samp']
prior_prec = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

prior_prec = [1.0,1e-1,10.0]

import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NumPy scalars (float32, int64, etc.)
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle PyTorch tensors (scalars or arrays)
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            return obj.detach().cpu().tolist()
        # Fallback to standard behavior
        return super(NpEncoder, self).default(obj)

for ht in hess_type:
    for st in samp_type:
        for p in prior_prec:
            la = Laplace(model, "classification",
                        subset_of_weights='last_layer',
                        hessian_structure=ht,
                        backend_kwargs= {'low_rank': 2000} if ht == 'lowrank' else None,
                        prior_precision = p)


            if st == 'linear' and ht == 'lowrank': 
                print('skip')
            else:
                print(ht,st,str(p))
                la.fit(mnist_train_loader)
                results = evaluate_laplace_full(la, mnist_loader, fmnist_loader, mnist_targets_tensor,sample=True if st=='samp' else False, n_samples =10)
                
                x,y = cal_plot(la,mnist_loader,sample=True if st=='samp' else False, n_samples =10)
                results['cal_x'] = x
                results['cal_y'] = y

                print(results)

                # filename = f'laplace_redux_dicts/redux_{ht}_{st}_prior{str(p)}.json'

                # with open(filename, "w") as f:
                #     # Pass 'cls=NpEncoder' to enable the custom conversion
                #     json.dump(results, f, cls=NpEncoder, indent=4)


    