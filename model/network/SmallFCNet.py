import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallFCNet(nn.Module):
    def __init__(self):
        super(SmallFCNet,self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 100) 
        self.final = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, img): #convert + flatten
        x = torch.flatten(img, start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x
