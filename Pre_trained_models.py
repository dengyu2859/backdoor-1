# 目的：训练CIFAR100的模型
# 首先使用MNIST测试集训练CNN模型，保留CNN模型参数
# 用CNN模型参数训练VAE模型
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from models import CNNMnist, CNN_Fashion_MNIST, resnet20
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import pathlib
from Test import Evaluate
from torch.utils.data import DataLoader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Get_model_param(args):
    setup_seed(20240901)# 固定随机种子
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("\nArguments: %s " % args_str)                                            # 打印参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)

    elif args.dataset == 'CIFAR10':
        Data_path = 'dataset/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=False, download=True, transform=transform)

    elif args.dataset == 'CIFAR100':
        Data_path = 'dataset/CIFAR100'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR100(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=Data_path, train=False, download=True, transform=transform)

    elif args.dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root='dataset', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root='dataset', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    if args.model == 'CNNMnist':
        model = CNNMnist(args=args).to(args.device)
    if args.model == 'CNN_Fashion_MNIST':
        model = CNN_Fashion_MNIST(args=args).to(args.device)
    if args.model == 'ResNet18':
        model = ResNet18(args=args).to(args.device)
    if args.model == 'resnet20':
        model = resnet20(args=args).to(args.device)

    print("\nStart training......\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001
                          , momentum=0.5)
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc_test, loss = Evaluate(model, test_set, criterion, args)
        print(f'Epoch: {epoch} |  {acc_test:.3f}')

    save_path = f"{args.model}_on_{args.dataset}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # 如果不存在vae文件夹，则创建
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="name of dataset: MNIST, FEMNIST, CIFAR10")
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--model', type=str, default='resnet20', help='ResNet18, CNNMnist, CNNFEMnist, resnet20')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")     # CIFAR10是彩色图像，通道3
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    args = parser.parse_args()
    Get_model_param(args)