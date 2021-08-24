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
import onnx

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


# Logging values for plots
graph_loss = []
natural_acc = []
adver_acc=[]

# Negative Log Likelihood loss
# (Remember, the last layer in model must have softmax layer, else use nn.CrossEntropyLoss)
loss_fn = nn.CrossEntropyLoss()


def model_select(args, device):
    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet().to(device)
    elif model_name == "smallfcnet":
        model = SmallFCNet().to(device)
    elif model_name == "medfcnet":
        model = MedFCNet().to(device)
    elif model_name == "bigfcnet":
        model = BigFCNet().to(device)
    elif model_name == "smallconvnet":
        model = SmallConvNet().to(device)
    elif model_name == "medconvnet":
        model = MedConvNet().to(device)
    elif model_name == "bigconvnet":
        model = BigConvNet().to(device)
    else:
        print("Model not available")
        assert(False)

    return model


# Defining training function


def train(args, model, device, train_loader, optimizer, scheduler, epoch, attack):
    # Intialize model to training phase
    model.train()
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    for batch_index, (data, label) in enumerate(train_loader):
        tmp_time = time.time()
        data, label = data.to(device), label.long().to(device)
        optimizer.zero_grad()
        # Forward pass of data
        pred = model(data)
        # Compute Prediction Error
        loss = loss_fn(pred, label)
        # Backpropagation
        loss.backward()
        optimizer.step()

        # logging the training progress
        if batch_index % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Cost time: {:.6f}s".format(
                epoch + 1, batch_index * len(data), size,
                100. * batch_index / num_batches, loss.item(), time.time() - tmp_time))
        graph_loss.append(loss.item())

# Defining testing function for natural and adversarial accuracy

def test(model, device, test_loader, attack):
    # Initialize model to evaluation phase
    model.eval()
    natural_test_loss = 0
    natural_correct = 0
    adver_test_loss = 0
    adver_correct = 0
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    # Disable computational graph (gradient storage due to autograd)
    for data, label in test_loader:
        adver_data = attack(data, label)
        with torch.no_grad():
            adver_data, label = adver_data.to(device), label.to(device)
            natural_data, label = data.to(device), label.to(device)

            # Natural prediction
            natural_pred = model(natural_data)
            natural_test_loss += loss_fn(natural_pred, label).item()  # sum up batch loss
            # Get the index of the max log-probability
            natural_correct += (natural_pred.argmax(1) == label).type(torch.float).sum().item()

            # Adversarial prediction
            adver_pred = model(adver_data)
            adver_test_loss += loss_fn(adver_pred, label).item()  # sum up batch loss
            # Get the index of the max log-probability
            adver_correct += (adver_pred.argmax(1) == label).type(torch.float).sum().item()

    # Summary of natural data result
    natural_test_loss /= num_batches
    natural_correct_percent = (100*natural_correct)/size
    print("\nTest set: Natural average loss: {:.4f}, Natural accuracy: {}/{} ({:.2f}%)\n".format(
        natural_test_loss, natural_correct, size, float(natural_correct_percent)))

    # Summary of adversarial data result
    adver_test_loss /= num_batches
    adver_correct_percent = (100*adver_correct)/size
    print("\nTest set: Adversarial average loss: {:.4f}, Adversarial accuracy: {}/{} ({:.2f}%)\n".format(
        adver_test_loss, adver_correct, size, float(adver_correct_percent)))

    return natural_test_loss, natural_correct_percent, adver_test_loss, adver_correct_percent


# Iterations

def epoch_train(args, model, device, train_loader, test_loader, optimizer, scheduler, attack):
    epochs = args.epochs
    best_epoch = 0
    best_acc = 0
    for epoch in range(epochs):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, scheduler, epoch, attack)
        scheduler.step()
        end_time = time.time()
        print("Epoch {} cost {} s".format(epoch + 1, end_time - start_time))

        natural_test_loss, natural_accuracy, adver_test_loss, adver_accuracy = test(model, device, test_loader, attack)
        # Logging natural accuracy
        natural_acc.append(natural_accuracy)
        # Logging adversarial accuracy
        adver_acc.append(adver_accuracy)

        if natural_accuracy > best_acc:
            best_acc = natural_accuracy
            best_epoch = epoch
            if args.save_model:
                model.eval()
                torch.save(model.state_dict(),
                          os.path.join(args.model_path,"{}.pt".format(args.model.lower())))
    print("Best epoch: {} | Best acc: {}".format(best_epoch, best_acc))
    print("Done!")


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
    parser.add_argument("--log-path", default="./model/natural/soft/result",
                        help="path to store results")
    parser.add_argument("--model-path", default="./model/natural/soft/weights",
                        help="path to store trained weights")
    parser.add_argument("--data-path", default="../ERAN/data/mnist_train.csv",
                        help="path to store trained weights")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="bounds for attack")
    parser.add_argument("--steps", type=int, default=7,
                        help="iteration for attack")
    parser.add_argument("--attack", default="PGD",
                        help="Type of attacks: PGD/FGSM")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="iteration for attack")

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
        # transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    # Loading dataset
    # train_data = datasets.MNIST(
    #    "./data", train=True, download=True, transform=transform)
    train_data = CustomDatasetFromCSV(args.data_path, 28, 28)
    test_data = datasets.MNIST("./data", train=False, transform=transform)
    # Dataloader for dataset. It make an iterable object by wrapping (data,label) of length batch_size
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = model_select(args, device)

    # Selecting Optimizer and Scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)    
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)    
    epsilon=args.epsilon
    steps=args.steps
    # Producing attacks
    if args.attack == "PGD":
        alpha=args.alpha
        attack = PGD(model, eps=epsilon, alpha=alpha, steps=steps)
    elif args.attack == "FGSM":
        attack = FGSM(model, eps=0.01)

    # Iterations (Training+Test(val))
    epoch_train(args, model, device, train_loader,
                test_loader, optimizer, scheduler, attack)
    
    model_name = "{}.pt".format(args.model.lower())
    model_path = Path(os.path.join(args.model_path, model_name))
    model=model.to("cpu")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print('model doesnot exist')
        exit()

    x = torch.randn(args.batch_size, 1, 28, 28, requires_grad=True)
    #torch_out = model(x)

    onnx_model_name = "{}.onnx".format(args.model.lower())
    onnx_model_path = Path(os.path.join(args.model_path, onnx_model_name))
    model.eval()
    # Export the model
    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  verbose=True,
                  #opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

    # Onnxruntime checks
    #onnx_model = onnx.load(onnx_model_name)
    #onnx.checker.check_model(onnx_model)
    #ort_session = onnxruntime.InferenceSession(onnx_model_path)

    #def to_numpy(tensor):
    #   return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    #ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    #ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    #np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    #print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # Convert onnx model into tensorflow
    #tf_model_name = "{}.pb".format(args.model.lower())
    #tf_model_path = Path(os.path.join(args.model_path, tf_model_name))
    #tf_rep = prepare(onnx_model)
    #tf_rep.export_graph(tf_model_name)

    # Record natural results
    create_loss_txt_filename = "{}_graph_loss.txt".format(args.model.lower())
    create_acc_txt_filename = "{}_nat_acc.txt".format(args.model.lower())

    create_loss_txt_path = os.path.join(
        args.log_path, create_loss_txt_filename)
    create_acc_txt_path = os.path.join(args.log_path, create_acc_txt_filename)

    f = open(create_loss_txt_path, "w+")
    for loss in graph_loss:
        f.writelines("{}\n".format(loss))
    f.close()
    f = open(create_acc_txt_path, "w+")
    for acc in natural_acc:
        f.writelines("{}\n".format(acc))
    f.close()

    # Record adversarial results
    create_acc_txt_filename = "{}_adver_acc.txt".format(args.model.lower())

    create_acc_txt_path = os.path.join(args.log_path, create_acc_txt_filename)

    f = open(create_acc_txt_path, "w+")
    for acc in adver_acc:
        f.writelines("{}\n".format(acc))
    f.close()





if __name__ == "__main__":
    main()
