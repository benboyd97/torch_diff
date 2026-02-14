import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, output_dim=10, activation="tanh"):
        super().__init__()
        
        self.activation = activation

        # Assume single-channel input (like MNIST)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # IMPORTANT: 16 * 4 * 4 = 256 (for 28x28 input)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def act_fun(self, x):
        if self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "relu":
            return F.relu(x)
        else:
            raise ValueError(f"Unknown activation {self.activation}")

    def forward(self, x):
        if x.dim() != 4:
            x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.act_fun(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = self.act_fun(x)
        x = F.max_pool2d(x, 2, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.act_fun(x)
        x = self.fc2(x)
        x = self.act_fun(x)
        x = self.fc3(x)

        return x
