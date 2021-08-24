# Importing libraries

from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time

# import network
from model.network.LeNet import LeNet
from model.network.MyNetV1 import MyNetV1
from model.network.MyNetV2 import MyNetV2
from model.network.DefaultNet import DefaultNet
from model.network.MyFullConvNet import MyFullConvNet
from model.network.MyVggNet import MyVggNet
from model.network.NewNet import NewNet

import torchattacks
from torchattacks import PGD, FGSM

# Logging values for plots
list_pred = list()

# Negative Log Likelihood loss
# (Remember, the last layer in model must have softmax layer, else use nn.CrossEntropyLoss)
loss_fn = nn.NLLLoss()


def model_select(args, device):
    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet().to(device)
    elif model_name == "defaultnet":
        model = DefaultNet().to(device)
    elif model_name == "mynetv1":
        model = MyNetV1().to(device)
    elif model_name == "mynetv2":
        model = MyNetV2().to(device)
    elif model_name == "myfullconvnet":
        model = MyFullConvNet().to(device)
    elif model_name == "myvggnet":
        model = MyVggNet().to(device)
    elif model_name == "newnet":
        model = NewNet().to(device)
    else:
        print("Model not available")
        assert(False)

    return model

# Defining testing function


def test(model, device, test_loader, attack):
    # Initialize model to evaluation phase
    model.eval()
    test_loss = 0
    correct = 0
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    # Disable computational graph (gradient storage due to autograd)
    for data, label in test_loader:
        data = attack(data, label)
        with torch.no_grad():
            data, label = data.to(device), label.to(device)
            pred = model(data)
            test_loss += loss_fn(pred, label).sum().item()  # sum up batch loss
            # Get the index of the max log-probability
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct_percent = (100*correct)/size
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, size, float(correct_percent)))

# Defining main function


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default : 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default : 1000)")
    parser.add_argument("--epochs", type=int, default=64, metavar="N",
                        help="number of epochs to train (default : 64)")
    parser.add_argument("--learning-rate", type=float, default=0.1, metavar="LR",
                        help="the learning rate (default : 0.1)")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="M",
                        help="Learning rate step gamma (default : 0.5)")
    parser.add_argument("--use-cuda", action="store_true", default=True,
                        help="Using CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default : 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", default=True,
                        help="For saving the current Model")
    parser.add_argument("--load_state_dict", type=str, default="no",
                        help="load the trained model weights or not (default: no)")
    parser.add_argument("--model", type=str, default="LeNet",
                        help="choose the model to train (default: LeNet)")
    parser.add_argument("--model-path", default="./model/adver/soft/weights/",
                        help="path to trained model")
    parser.add_argument("--attack", default="PGD",
                        help="Type of attacks: PGD/FGSM")
    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()  # not > and > or
    print("Using cuda is: {}".format(use_cuda))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize(mean, std, inplace=False)
        # output = (input - mean) / std
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    # Loading dataset
    test_data = datasets.MNIST(
        "./data", train=False, download=True, transform=transform)
    # Dataloader for dataset. It make an iterable object by wrapping (data,label) of length batch_size
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = model_select(args, device)
    model_name = "{}.pt".format(args.model.lower())
    model_path = Path(os.path.join(args.model_path, model_name))
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Wrong model name. Try again!")
        exit()
    print("\nTest model:\t{}".format(args.model))

    # Producing attacks
    if args.attack == "PGD":
        attack = PGD(model, eps=0.3, alpha=0.1, steps=7)
    elif args.attack == "FGSM":
        attack = FGSM(model, eps=0.3)

    test(model, device, test_loader, attack)


if __name__ == "__main__":
    main()
