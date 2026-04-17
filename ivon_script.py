# %%
import ivon

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import ivon

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_mnist
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform_mnist
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

torch.manual_seed(0)
device = 'cuda'
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-1
batch_size = 128
epochs = 100
weight_decay = 1e-4
momentum = 0.9
print_mod = 450

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# %%
def train_loop_ivon(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        for _ in range(train_samples):
            with optimizer.sampled_params(train=True):
                logit = model(X)
                loss = loss_fn(logit, y)
                loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        if batch % print_mod == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop_ivon(dataloader, model, loss_fn, test_mc):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Predict at mean
            if test_mc == 0:
                logit = model(X)
                test_loss += loss_fn(logit, y).item()
                correct += (logit.argmax(1) == y).type(torch.float).sum().item()
            # Predict with samples
            else:
                sampled_probs = []
                for i in range(test_mc):
                    with optimizer.sampled_params():
                        sampled_logit = model(X)
                        sampled_probs.append(F.softmax(sampled_logit, dim=1))
                prob = torch.mean(torch.stack(sampled_probs), dim=0)
                _, pred = prob.max(1)
                test_loss -= torch.sum(torch.log(prob.clamp(min=1e-6)) * F.one_hot(y, 10), dim=1).mean()
                correct += pred.eq(y).sum().item()

    test_loss /= num_batches
    correct /= size
    if test_mc != 0:
        print(f"\nIVON -- Test Performance with {test_mc:0d} Test Samples \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")
    else:
        print(f"IVON -- Test Performance with Mean Prediction \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")

h0 = 0.01
train_samples = 1
optimizer = ivon.IVON(model.parameters(), lr=learning_rate, ess=len(training_data), weight_decay=weight_decay, beta1=momentum, hess_init=h0)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop_ivon(train_dataloader, model, loss_fn, optimizer)

# Predict at the mean of the variational posterior

torch.save(model.state_dict(), 'ivan_weights_vi.pth')
test_mc0 = 0
test_loop_ivon(test_dataloader, model, loss_fn, test_mc0)

# Predict using samples
test_mc64 = 64
test_loop_ivon(test_dataloader, model, loss_fn, test_mc64)

print("Done training with IVON!")

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logit = model(X)
        loss = loss_fn(logit, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if batch % print_mod == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logit = model(X)
            test_loss += loss_fn(logit, y).item()
            correct += (logit.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"\nSGD -- Test Performance: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>7f} \n")

weight_reset(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)

torch.save(model.state_dict(), 'sgd_weights_vi.pth')

test_loop(test_dataloader, model, loss_fn)

print("Done training with SGD!")

# %%



