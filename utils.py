import pathlib
import torch
import torchvision
import torchvision.transforms as transforms
import os
from Distribution import NO_iid
from torch.utils.data import Dataset
import random
from collections import OrderedDict
import shutil


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


# 测试集嵌入后门触发器
def Backdoor_process(test_dataset, args):
    label_indices = [i for i, (_, label) in enumerate(test_dataset) if label in args.back_target]
    Inject_trigger(test_dataset, label_indices, args)


# 恶意客户端选择，随机方案
# def Choice_mali_clients(dict_users, args):
#     client_ids = list(dict_users.keys())
#     num_malicious = int(len(client_ids) * args.malicious)
#     malicious_id = random.sample(client_ids, num_malicious)
#
#     return malicious_id


# 恶意客户端选择, 保证被选中的恶意客户端都有指定的后门标签数据
# def Choice_mali_clients(dict_users, dataset, args):
#     if args.dataset == 'FEMNIST':
#         user_with_target = [
#             user for user, sample_indices in dict_users.items()
#             if any(dataset.label[sample] in args.back_target for sample in sample_indices)
#         ]
#
#         num_malicious_clients = int(args.clients * args.malicious)
#
#         return user_with_target if len(user_with_target) <= num_malicious_clients else random.sample(user_with_target,
#                                                                                                      num_malicious_clients)
#     else:
#         user_with_target = [
#             user for user, sample_indices in dict_users.items()
#             if any(dataset.targets[sample] in args.back_target for sample in sample_indices)
#         ]
#
#         num_malicious_clients = int(args.clients * args.malicious)
#
#         return user_with_target if len(user_with_target) <= num_malicious_clients else random.sample(user_with_target,
#                                                                                                      num_malicious_clients)

def Choice_mali_clients(dict_users, args):
    num_malicious_clients = int(args.clients * args.malicious)
    all_clients = list(dict_users.keys())
    malicious_clients = random.sample(all_clients, num_malicious_clients)
    return malicious_clients



# 将模型转化为一维张量
def model_to_vector(model, args):
    dict_param = model.state_dict()
    param_vector = torch.cat([p.view(-1) for p in dict_param.values()]).to(args.device)

    return param_vector


# 将一维张量加载为模型
def vector_to_model(model, param_vector, args):
    model_state_dict = model.state_dict()
    new_model_state_dict = OrderedDict()
    start_idx = 0
    # 遍历模型的 state_dict 和每个参数的元素数量
    for (key, _), numel in zip(model_state_dict.items(), [p.numel() for p in model_state_dict.values()]):
        # 从 param_vector 中取出对应数量的元素，并恢复形状
        new_param = param_vector[start_idx:start_idx + numel].view_as(model_state_dict[key])
        # 更新新的 state_dict
        new_model_state_dict[key] = new_param
        # 更新起始索引
        start_idx += numel
    model.load_state_dict(new_model_state_dict)


# 创建，清空目录：存储局部模型的
def reset_directory(path):
    """清空并重新创建指定目录"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# 生成空的权重
def get_empty_accumulator(model):
    weight_accumulator = dict()
    for name, data in model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(data)
    return weight_accumulator


# 排除无关层
def check_ignored_weights(name):
    ignored_weights = ['num_batches_tracked']
    for ignored in ignored_weights:
        if ignored in name:
            return True

    return False


# 输出实验信息
def print_exp_details(args):
    print('======================================')
    print(f'    GPU: {args.gpu}')
    print(f'    Dataset: {args.dataset}')
    print(f'    Model: {args.model}')
    print(f'    Num_classes: {args.num_classes}')
    print(f'    Number of clients: {args.clients}')
    print(f'    Rounds of training: {args.epochs}')
    print(f'    Attack_type: {args.attack_type}')
    print(f'    malicious clients: {args.malicious}')
    print(f'    pre_model: {args.pre_model}')
    print(f'    Degree of no-iid: {args.a}')
    print(f'    Batch size: {args.local_bs}')
    print(f'    lr: {args.lr}')
    print(f'    Momentum: {args.momentum}')
    print(f'    Local_ep: {args.local_ep}')
    print('======================================')







































#
# import os
# import shutil
# import wget
# import pathlib
# import gzip
#
# def Load_dataset(name,data_path):
#     path = data_path + '/' + name
#     if not os.path.exists(path):
#         pathlib.Path(path).mkdir(parents=True,exist_ok=True)
#
#     #-----------Download dataset--------------
#     train_set_imgs_addr = path + '/'+ "train-images-idx3-ubyte.gz"
#     train_set_labels_addr = path + '/'+ "train-labels-idx1-ubyte.gz"
#     test_set_imgs_addr = path + '/'+ "t10k-images-idx3-ubyte.gz"
#     test_set_labels_addr = path + '/'+ "t10k-labels-idx1-ubyte.gz"
#     try:
#         if not os.path.exists(train_set_imgs_addr):
#             print("Downingload train-images-idx3-ubyte.gz")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", out=str(train_set_imgs_addr))
#             print("\tdone.")
#         else:
#             print("train-images-idx3-ubyte.gz already exists.")
#         if not os.path.exists(train_set_labels_addr):
#             print("Downingload train-labels-idx1-ubyte.gz.")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",out=str(train_set_labels_addr))
#             print("\tdone.")
#         else:
#             print("train-labels-idx1-ubyte.gz already exists.")
#         if not os.path.exists(test_set_imgs_addr):
#             print("Downingload t10k-images-idx3-ubyte.gz.")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",out=str(test_set_imgs_addr))
#             print("\tdone.")
#         else:
#             print("t10k-images-idx3-ubyte.gz already exists.")
#         if not os.path.exists(test_set_labels_addr):
#             print("Downingload t10k-labels-idx1-ubyte.gz.")
#             filename = wget.download(url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",out=str(test_set_labels_addr))
#             print("\tdone.")
#         else:
#             print("t10k-labels-idx1-ubyte.gz already exists.")
#     except:
#         return False
#
#     # -----------------Unzip file--------------------
#     for filename in os.listdir(path):
#         if filename.endswith('.gz'):
#             score_file = os.path.join(path,filename)                            # Compressed file path
#             target_file = os.path.join(path,os.path.splitext(filename)[0])      # Unzip file path
#             if not os.path.exists(target_file):
#                 with gzip.open(score_file,'rb') as f_in:
#                     with open(target_file,'wb') as f_out:
#                         shutil.copyfileobj(f_in,f_out)
#                 print(target_file, "unzipped")
#             else:
#                 print(target_file, " already exists")
#     return True
#
#
# if __name__ == '__main__':
#     data_path = 'dataset'
#     # Data_name = ['MNIST','FEMNIST','CIFAR']
#     Data_name = ['MNIST']
#     for name in Data_name:
#         print("Now Loading dataset",name,"......")
#         Load_dataset(name,data_path)