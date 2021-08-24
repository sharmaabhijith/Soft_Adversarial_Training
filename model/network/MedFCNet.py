import torch
import torch.nn as nn
import torch.nn.functional as F

class MedFCNet(nn.Module):
    def __init__(self):
        super(MedFCNet,self).__init__()
        self.linear1 = nn.Linear(28*28, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 200)
        self.linear5 = nn.Linear(200, 200)
        self.final = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = torch.flatten(img, start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.final(x)
        return x

