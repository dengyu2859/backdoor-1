import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import os
from Distribution import NO_iid
from torch.utils.data import Dataset
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np


def Download_data(name, path, args):
    train_set, test_set, dict_users = None, None, None
    Data_path = 'dataset'
    if not os.path.exists(Data_path):
        pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)

    elif name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'EMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.EMNIST(root=path, train=True, split='balanced', download=True, transform=transform)
        test_set = torchvision.datasets.EMNIST(root=path, train=False, split='balanced', download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'CIFAR10':
        Data_path = 'dataset/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'CIFAR100':
        Data_path = 'dataset/CIFAR100'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR100(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    return train_set, test_set, dict_users


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.labels = torch.tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.labels)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# 注入触发器，修改标签
def Inject_trigger(test_dataset, target_indices, args):
    start_positions = [(1, 2), (1, 8), (3, 2), (3, 8)]
    for i in range(len(test_dataset.data)):     # 遍历所有测试数据
        if i in target_indices:                 # 选择目标样本注入触发器
            for start_row, start_col in start_positions:
                for j in range(start_col, start_col + 4):
                    if args.dataset in ['FashionMNIST', 'MNIST']:
                        test_dataset.data[i][start_row][j] = 255
                    elif args.dataset in ['CIFAR10', 'CIFAR100']:
                        test_dataset.data[i][start_row][j] = [255, 255, 255]
        else:
            continue


def split_testset_by_class(test_set):
    labels = test_set.targets
    # 按类别分组索引
    class_indices = {i: [] for i in range(len(test_set.classes))}
    for i, label in enumerate(labels):
        class_indices[label.item()].append(i)
    distill_indices = []
    new_test_indices = []
    # 遍历每个类别，将其索引对半分割
    for class_id in class_indices:
        indices = class_indices[class_id]
        # 确保分割的随机性
        torch.manual_seed(20250901)
        torch.randperm(len(indices))
        split_point = len(indices) // 2
        distill_indices.extend(indices[:split_point])
        new_test_indices.extend(indices[split_point:])
    random.shuffle(distill_indices)
    random.shuffle(new_test_indices)
    # 使用 PyTorch 的 Subset 创建新的数据集
    distill_dataset = Subset(test_set, distill_indices)
    new_test_dataset = Subset(test_set, new_test_indices)

    return distill_dataset, new_test_dataset


def Visualize_results(acc_history, asr_history, num_clients):
    # 图1: ACC 结果
    plt.figure(figsize=(10, 6))
    for client_id, acc_list in acc_history.items():
        plt.plot(range(1, len(acc_list) + 1), acc_list, label=f'Client {client_id}')

    plt.title('Accuracy of Each Client Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 图2: ASR 结果
    plt.figure(figsize=(10, 6))
    for client_id, asr_list in asr_history.items():
        # 只有当客户端有asr数据时才绘制（恶意客户端）
        if any(np.array(asr_list) > 0):
            plt.plot(range(1, len(asr_list) + 1), asr_list, label=f'Client {client_id}')

    plt.title('Attack Success Rate (ASR) of Each Client Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ASR', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


# 输出实验信息
def print_exp_details(args):
    print('======================================')
    print(f'    GPU: {args.gpu}')
    print(f'    Dataset: {args.dataset}')
    print(f'    Model: {args.model}')
    print(f'    Num_classes: {args.num_classes}')
    print(f'    Number of clients: {args.clients}')
    print(f'    Rounds of training: {args.epochs}')
    print(f'    Attack: {args.attack}')
    print(f'    malicious clients: {args.malicious}')
    print(f'    Degree of no-iid: {args.a}')
    print(f'    Batch size: {args.local_bs}')
    print(f'    lr: {args.lr}')
    print(f'    Momentum: {args.momentum}')
    print(f'    Local_ep: {args.local_ep}')
    print('======================================')
