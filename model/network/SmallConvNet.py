import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallConvNet(nn.Module):
    def __init__(self):
        super(SmallConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.linear = nn.Linear(800, 100)
        self.final = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.final(x)
        return x

