# Importing libraries
from __future__ import print_function
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
import time
import csv

# import dataset
from custom_dataset import CustomDatasetFromCSV

# import network
from model.network.LeNet import LeNet
from model.network.SmallFCNet import SmallFCNet
from model.network.MedFCNet import MedFCNet
from model.network.BigFCNet import BigFCNet
from model.network.SmallConvNet import SmallConvNet
from model.network.MedConvNet import MedConvNet
from model.network.BigConvNet import BigConvNet


# import attacks
import torchattacks
from torchattacks import PGD, FGSM

# Negative Log Likelihood loss
# (Remember, the last layer in model must have softmax layer, else use nn.CrossEntropyLoss)
loss_fn = nn.CrossEntropyLoss()


def model_select(args):
    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet()
    elif model_name == "smallfcnet":
        model = SmallFCNet()
    elif model_name == "medfcnet":
        model = MedFCNet()
    elif model_name == "bigfcnet":
        model = BigFCNet()
    elif model_name == "smallconvnet":
        model = SmallConvNet()
    elif model_name == "medconvnet":
        model = MedConvNet()
    elif model_name == "bigconvnet":
        model = BigConvNet()
    else:
        print("Model not available")
        assert(False)

    return model

def main():
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--model", default='SmallFCNet',
                        help="Model for attack")
    parser.add_argument("--path", default="../ERAN/tf_verify",
                        help="Model for attack")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="bounds for attack")
    parser.add_argument("--steps", type=int, default=7,
                        help="iteration for attack")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="iteration for attack")
    args=parser.parse_args()

    # Outer path
    path=args.path

    # Model for attack
    model=model_select(args)

    #Import csv file of non verifed images
    filepath=os.path.join(path, "unverified_mnist.csv")
    train_data = CustomDatasetFromCSV(filepath, 28, 28)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)

    # Initialise attack for images
    epsilon=args.epsilon
    steps=args.steps
    alpha=args.alpha
    attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
     
    filepath =os.path.join(path,"attack_mnist.csv")
    for i, (data,label) in enumerate(train_loader):
        # Attack images
        label=label.long()
        data=attack(data,label)
        data=data*255
        data=data.squeeze(0).squeeze(0).reshape(1,784).numpy()
        label=label.unsqueeze(0).numpy()
        point=np.concatenate((label,data),axis=1)

        # Save attacked images
        with open(filepath,'a',newline='') as openfile:
            writer=csv.writer(openfile, delimiter=',')
            writer.writerows(point)

if __name__ == "__main__":
    main()

