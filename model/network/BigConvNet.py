import torch
import torch.nn as nn
import torch.nn.functional as F

class BigConvNet(nn.Module):
    def __init__(self):
        super(BigConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv3 = nn.Conv2d(8, 14, 3)
        self.conv4 = nn.Conv2d(14, 14, 3)
        self.linear1 = nn.Linear(224,50)
        self.linear2 = nn.Linear(50, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.final(x)
        return x

