import torch
import torch.nn as nn
import torch.nn.functional as F

class BigFCNet(nn.Module):
    def __init__(self):
        super(BigFCNet,self).__init__()
        self.linear1 = nn.Linear(28*28, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 500)
        self.linear4 = nn.Linear(500, 500)
        self.linear5 = nn.Linear(500, 500)
        self.final = nn.Linear(500, 10)
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
                                    
