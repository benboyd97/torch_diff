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

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()

probs_map = predict(mnist_test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == mnist_targets).float().mean()
ece_map = ECE(bins=10).measure(probs_map.numpy(), mnist_targets.numpy())
mce_map = MCE(bins=10).measure(probs_map.numpy(), mnist_targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(mnist_targets).mean()
print(f"[MAP] Acc.: {acc_map:.4%}; ECE: {ece_map:.4%}; MCE: {mce_map:.4%}; NLL: {nll_map:.4}")

la = Laplace(model, "classification",
             subset_of_weights="last_layer",
             hessian_structure="kron", prior_precision = 1.0)
la.fit(mnist_train_loader)
#la.optimize_prior_precision(method="marglik",link_approx = 'mc', n_samples = 10)
probs_laplace = predict(mnist_test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == mnist_targets).float().mean()

ece_laplace = ECE(bins=10).measure(probs_laplace.numpy(), mnist_targets.numpy())
mce_laplace = MCE(bins=10).measure(probs_laplace.numpy(), mnist_targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(mnist_targets).mean()

print(f"[Laplace] Acc.: {acc_laplace:.4%}; ECE: {ece_laplace:.4%}; MCE: {mce_laplace:.4%}; NLL: {nll_laplace:.4}")

