# library
# standard library
import os
import numpy
# third-party library
import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# import network
from model.network.LeNet import LeNet
from model.network.MyNetV1 import MyNetV1
from model.network.MyNetV2 import MyNetV2
from model.network.DefaultNet import DefaultNet
from model.network.MyFullConvNet import MyFullConvNet
from model.network.MyVggNet import MyVggNet
from model.network.SmallFCNet import SmallFCNet
from model.network.MedFCNet import MedFCNet

# import attacks
import torchattacks
from torchattacks import PGD, FGSM


def model_select(args):
    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet()
    elif model_name == "defaultnet":
        model = DefaultNet()
    elif model_name == "mynetv1":
        model = MyNetV1()
    elif model_name == "mynetv2":
        model = MyNetV2()
    elif model_name == "myfullconvnet":
        model = MyFullConvNet()
    elif model_name == "myvggnet":
        model = MyVggNet()
    elif model_name == "smallfcnet":
        model = SmallFCNet()
    elif model_name == "medfcnet":
        model = MedFCNet()
    else:
        print("Model not available")
        assert(False)

    return model


# Mnist digits dataset
# if not(os.path.exists('./data/')) or not os.listdir('./MINIST/'):
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True
def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default : 1)")
    parser.add_argument("--eps", default=0,
                        help="epsilon for attack")
    parser.add_argument("--steps", default=20,
                        help="steps for attack")
    parser.add_argument("--model", type=str, default="LeNet",
                        help="choose the model to train (default: LeNet)")
    parser.add_argument("--attack", default="PGD",
                        help="Type of attacks: PGD/FGSM")
    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize(mean, std, inplace=False)
        # output = (input - mean) / std
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    train_data = torchvision.datasets.MNIST( root='./data/', train=False, transform=transform)
    model = model_select(args)

    # plot one example
    print(train_data.train_data.size())                 # (60000, 28, 28)
    print(train_data.targets.size())               # (60000)
    eps=args.eps
    steps=args.steps
    # Producing attacks
    if args.attack == "PGD":
        attack = PGD(model, eps=0.3, alpha=0.1, steps=20)
    elif args.attack == "FGSM":
        attack = FGSM(model, eps=args.eps)

    for i in range(0, 4):
       data=train_data.train_data[0].unsqueeze(0).unsqueeze(0)/255
       label=train_data.train_labels[0].unsqueeze(0)
       if i==0:
           plt.subplot(1,4,i+1)
           plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
           plt.axis('off')
       else:
           if i==1:
               attack = PGD(model, eps=0.1, alpha=0.1, steps=20)
           elif i==2:
               attack = PGD(model, eps=0.4, alpha=0.1, steps=20)
           elif i==3:
               attack = PGD(model, eps=0.7, alpha=0.1, steps=20)
           plt.subplot(1,4,i+1)
           data=attack(data,label)
           data=data.squeeze(0).squeeze(0)
           plt.imshow(data.numpy(), cmap='gray')
           plt.axis('off')
    plt.show()

#print(train_data.train_data[0])
#plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
#plt.title('%i' % train_data.train_labels[0])
#plt.show()

if __name__ == "__main__":
    main()

