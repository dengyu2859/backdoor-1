import pathlib
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from Distribution import NO_iid, Generate_non_iid_datasets_dict
from torch.utils.data import Dataset
import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import zipfile
from io import BytesIO
from sklearn.neighbors import LocalOutlierFactor
from model.CNN import CNN_layer2, CNN_layer3
from model import cifar10 as models
from model.Image_net import get_model_imagenet
from model.GTSRB import get_model_gtsrb
from PIL import Image
import pandas as pd

def download_gtsrb(root='dataset'):
    dataset_dir = os.path.join(root, 'GTSRB')
    train_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
    test_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
    test_labels_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'

    if os.path.exists(dataset_dir):
        print(f"GTSRB数据集已存在于 {dataset_dir}")
        # print(f"训练目录内容: {os.listdir(os.path.join(dataset_dir, 'Final_Training', 'Images')) if os.path.exists(os.path.join(dataset_dir, 'Final_Training', 'Images')) else '训练目录不存在'}")
        return dataset_dir

    os.makedirs(dataset_dir, exist_ok=True)
    print("正在下载GTSRB数据集...")

    def download_and_extract(url, extract_to):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            print(f"下载成功: {url.split('/')[-1]}")
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(extract_to)
                print(f"解压成功到 {extract_to}")
        except requests.exceptions.RequestException as e:
            print(f"下载失败 {url}: {e}")
            raise
        except zipfile.BadZipFile as e:
            print(f"解压失败 {url}: {e}")
            raise

    download_and_extract(train_url, dataset_dir)
    download_and_extract(test_url, dataset_dir)
    download_and_extract(test_labels_url, dataset_dir)
    print(f"GTSRB数据集已下载并解压至 {dataset_dir}")
    return dataset_dir

class GTSRBDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = download_gtsrb(root)
        self.transform = transform
        self.train = train
        self.data = []
        self.targets = []
        self.classes = list(range(43))

        if train:
            train_dir = os.path.join(self.root, 'Final_Training', 'Images')
            print(f"Loading training data from {train_dir}")
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, f'{cls:05d}')
                csv_file = os.path.join(cls_dir, f'GT-{cls:05d}.csv')
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, sep=';')
                    # print(f"Class {cls}: Found {len(df)} images in {csv_file}")
                    for _, row in df.iterrows():
                        img_path = os.path.join(cls_dir, row['Filename'])
                        if os.path.exists(img_path):
                            self.data.append(img_path)
                            self.targets.append(cls)
                        else:
                            print(f"Image not found: {img_path}")
                else:
                    print(f"CSV not found for class {cls}: {csv_file}")
            print(f"Total training samples loaded: {len(self.data)}")
        else:
            test_dir = os.path.join(self.root, 'Final_Test', 'Images')
            csv_file = os.path.join(self.root, '..', 'GT-final_test.csv')
            print(f"Loading test data from {test_dir}")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, sep=';')
                print(f"Test CSV found with {len(df)} entries")
                for _, row in df.iterrows():
                    img_path = os.path.join(test_dir, row['Filename'])
                    if os.path.exists(img_path):
                        self.data.append(img_path)
                        self.targets.append(row['ClassId'])
                    else:
                        print(f"Image not found: {img_path}")
            else:
                print(f"Test CSV not found: {csv_file}")
            print(f"Total test samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label



def download_tiny_imagenet(root='dataset'):
    dataset_dir = os.path.join(root, 'tiny-imagenet-200')
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    if os.path.exists(dataset_dir):
        print(f"Tiny-ImageNet already exists at {dataset_dir}")
        return dataset_dir

    print("Downloading Tiny-ImageNet...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("Extracting dataset...")
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(root)
        print(f"Tiny-ImageNet downloaded and extracted to {dataset_dir}")
        return dataset_dir
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download Tiny-ImageNet: {e}")
    except zipfile.BadZipFile:
        raise Exception("Failed to extract Tiny-ImageNet: Invalid zip file")


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = download_tiny_imagenet(root)  # Download if not present
        self.transform = transform
        self.train = train
        self.classes = []
        self.data = []
        self.targets = []

        # Load class names
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        if train:
            train_dir = os.path.join(self.root, 'train')
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls, 'images')
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.JPEG'):
                        self.data.append(os.path.join(cls_dir, img_name))
                        self.targets.append(self.class_to_idx[cls])
        else:
            val_dir = os.path.join(self.root, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            img_to_class = {}
            with open(val_annotations, 'r') as f:
                for line in f:
                    img_name, cls = line.strip().split('\t')[:2]
                    img_to_class[img_name] = self.class_to_idx[cls]
            val_img_dir = os.path.join(val_dir, 'images')
            for img_name in os.listdir(val_img_dir):
                if img_name.endswith('.JPEG'):
                    self.data.append(os.path.join(val_img_dir, img_name))
                    self.targets.append(img_to_class[img_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def Download_data(name, path, args):
    train_set, test_set, client_datasets = None, None, None
    Data_path = 'dataset'
    if not os.path.exists(Data_path):
        pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)

    elif name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'GTSRB':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))])
        train_set = GTSRBDataset(root='dataset/GTSRB', train=True, transform=transform)
        test_set = GTSRBDataset(root='dataset/GTSRB', train=False, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = torchvision.datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name == 'ImageNet':
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = TinyImageNet(root='dataset', train=True, transform=transform)
        test_set = TinyImageNet(root='dataset', train=False, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)

    elif name == 'CIFAR10':
        Data_path = 'dataset/CIFAR10'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=Data_path, train=False, download=True, transform=transform)
        client_datasets = Generate_non_iid_datasets_dict(train_set, args.clients, args.a)
        # dict_users = NO_iid(train_set, args.clients, args.a)

    elif name == 'CIFAR100':
        Data_path = 'dataset/CIFAR100'
        if not os.path.exists(Data_path):
            pathlib.Path(Data_path).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR100(root=Data_path, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=Data_path, train=False, download=True, transform=transform)
        dict_users = NO_iid(train_set, args.clients, args.a)

    return train_set, test_set, client_datasets


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
        class_indices[int(label)].append(i)
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


def Visualize_results(acc_history, asr_history):
    # === 创建保存文件夹 ===
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)

    # === 生成时间戳与文件名 ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    acc_filename = os.path.join(save_dir, f"Accuracy_Plot_{timestamp}.png")
    asr_filename = os.path.join(save_dir, f"ASR_Plot_{timestamp}.png")
    # 图1: ACC 结果
    plt.figure(figsize=(10, 6))
    for client_id, acc_list in acc_history.items():
        plt.plot(range(1, len(acc_list) + 1), acc_list, label=f'Client {client_id}')

    plt.title('Accuracy of Each Client Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(-5, 105)  # **固定纵坐标 0~100**
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_filename)
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
    plt.ylim(-5, 105)  # **固定纵坐标 0~100**
    plt.legend()
    plt.grid(True)
    plt.savefig(asr_filename)  # 保存图片
    plt.show()


def create_pixel_trigger_final(distillation_subset, top_percentage: float = 0.9):
    sample_tensor, _ = distillation_subset[0]

    # 现在 sample_tensor 是一个张量，你可以安全地获取它的 shape
    shape = sample_tensor.shape

    # 初始化累加器，保留通道维度
    sum_pixels = np.zeros(shape, dtype=np.float32)
    sum_sq_pixels = np.zeros(shape, dtype=np.float32)

    print(f"开始处理 {len(distillation_subset)} 张图像，计算平均值和方差...")

    loader = DataLoader(distillation_subset, batch_size=1, shuffle=False)

    # 单次遍历所有图像，计算总和和平方和
    for img_tensor, _ in loader:
        # 将张量转换为NumPy数组，保留所有维度
        # 我们在这里保留通道维度
        img_array = img_tensor.squeeze(0).numpy().astype(np.float32)

        sum_pixels += img_array
        sum_sq_pixels += img_array ** 2

    # 计算均值和方差图
    mean_pixels = sum_pixels / len(distillation_subset)         # 均值
    mean_sq_pixels = sum_sq_pixels / len(distillation_subset)
    variance_map = mean_sq_pixels - mean_pixels ** 2            # 方差

    print("计算完成，开始寻找最佳像素位置...")

    # 创建一个通用的前景掩码，只保留方差素大于0的区域
    universal_mask = (variance_map != 0)
    # 将掩码应用到方差图上，排除所有像素值为0的区域
    variance_map[universal_mask == False] = np.inf

    # 找到方差最小的前 N% 的像素位置
    flat_variance = variance_map.flatten()      # 这里是在展平这个数组
    # num_pixels = len(flat_variance)

    # 找到所有方差不是inf的像素，这里是寻找不是inf的数量
    finite_variance_indices = np.where(variance_map != np.inf)
    # 统计前景像素的总数
    num_foreground_pixels = len(finite_variance_indices[0])

    # 确认我们要选择像素阈值的位置
    num_candidates = int(num_foreground_pixels * (top_percentage / 100))

    threshold_value = np.partition(flat_variance, num_candidates)[num_candidates]       # 得到阈值
    best_candidate_coords = np.where((variance_map >= threshold_value) & (variance_map != np.inf))        # >效果很好

    mask_np = np.zeros(shape, dtype=np.uint8)
    mask_np[best_candidate_coords] = 1

    pattern_np = np.zeros(shape, dtype=np.float32)
    pattern_np[best_candidate_coords] = mean_pixels[best_candidate_coords]
    mask_tensor = torch.from_numpy(mask_np)
    pattern_tensor = torch.from_numpy(pattern_np)

    print(f"成功创建了包含 {len(best_candidate_coords[0])} 个像素的触发器图像。")

    return mask_tensor, pattern_tensor


def find_mask_and_pattern(subset, thr=3.0):
    """
    subset: torch.utils.data.Subset
    返回:
        mask    (1,28,28)  uint8
        pattern (1,28,28)  与原图数据类型一致
    """
    # -------- 1. 取出图像数据 --------
    imgs = []
    for x, _ in subset:  # 只取图像
        x = x.squeeze()   # 去掉可能的通道维
        imgs.append(x.numpy())
    X = np.stack(imgs, axis=0)  # (N,28,28)

    # -------- 2. 展平计算异常 --------
    N, H, W = X.shape
    X_flat = X.reshape(N, -1)

    median = np.median(X_flat, axis=0)          # 得到每个维度的中位数
    mad = np.median(np.abs(X_flat - median), axis=0) + 1e-8     # 得到每个维度的中位数绝对偏差，得到偏差中位数
    z = np.abs(X_flat - median) / mad
    outlier_ratio = (z > thr).mean(axis=0)      # outlier_ratio[123] = 0.02 表示 123 号像素在 2% 的样本中被认为异常

    # 只保留少数样本异常的维度
    interesting_dims = np.where((outlier_ratio > 0.1) & (outlier_ratio < 0.15))[0]

    # -------- 3. 生成 mask 和 pattern --------
    mask = torch.zeros((1, H, W), dtype=torch.uint8)
    pattern = torch.zeros((1, H, W), dtype=torch.float32)  # 可根据需要改为和原图一致的dtype

    mask_flat = mask.reshape(-1)
    pattern_flat = pattern.reshape(-1)

    for j in interesting_dims:
        abnormal_idx = np.where(z[:, j] > thr)[0]
        abnormal_vals = X_flat[abnormal_idx, j]
        if abnormal_vals.size == 0:
            continue
        mode_val = stats.mode(abnormal_vals, keepdims=True).mode[0]
        mask_flat[j] = 1
        pattern_flat[j] = float(mode_val)

    num_trigger_pixels = mask.sum().item()
    print(f"成功创建了包含 {num_trigger_pixels} 个像素的触发器图像。")
    return mask, pattern


def find_backdoor_trigger_samples_minimal(distill_dataset, global_model, args):

    target_module = global_model.conv2
    target_loader = DataLoader(distill_dataset, batch_size=64, shuffle=False)

    # 步骤 2: 提取卷积层特征 (使用钩子)
    global_model.eval()
    global_model.to(args.device)
    feature_storage = []

    def hook_fn(module, input, output):
        """前向钩子：将输出展平后存储。"""
        # 展平特征：从 (Batch, C, H, W) 展平为 (Batch, C*H*W)
        X = output.view(output.size(0), -1).cpu().numpy()
        feature_storage.append(X)

    # 注册钩子
    hook_handle = target_module.register_forward_hook(hook_fn)

    # 运行模型
    with torch.no_grad():
        for images, _ in target_loader:
            images = images.to(args.device)
            # Show_img(images[0], _, mean=(0.5,), std=(0.5,))
            # 执行前向传播，钩子会自动捕获 target_module 的输出
            _ = global_model(images)

    hook_handle.remove()

    # 合并特征
    X_raw = np.concatenate(feature_storage, axis=0)
    pca = PCA(n_components=0.95)  # 保留 95% 的方差
    X_pca = pca.fit_transform(X_raw)
    # dbscan = DBSCAN(eps=30, min_samples=5)
    # labels = dbscan.fit_predict(X_raw)
    kmeans = KMeans(n_clusters=100, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_raw)
    unique_labels = np.unique(labels)
    cluster_indices_dict = {}
    for label in unique_labels:
        subset_indices = np.where(labels == label)[0]
        cluster_indices_dict[label] = subset_indices
    non_noise_clusters = {label: indices for label, indices in cluster_indices_dict.items()}
    largest_cluster_key = max(non_noise_clusters, key=lambda label: len(non_noise_clusters[label]))
    choice_image = []
    for data_index in range(len(distill_dataset)):
        if data_index in non_noise_clusters.get(66, []):      # 选择第0个簇的样本
            image, label = distill_dataset[data_index]
            choice_image.append(image.squeeze(0))
            Show_img(image, label)
    choice_image_tensor = torch.stack(choice_image, dim=0).to(args.device)

    N, C, H, W = choice_image_tensor.shape
    device = choice_image_tensor.device

    I_avg_tensor = choice_image_tensor.mean(dim=0)

    # 转换为 LOF 特征空间 (H*W, C)
    I_avg_np = I_avg_tensor.permute(1, 2, 0).cpu().numpy()
    X_features = I_avg_np.reshape(-1, C)

    # LOF 模型
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=False)
    lof_model.fit(X_features)

    # 计算 LOF 评分 (值越大，越是异常点)
    lof_scores = -lof_model.negative_outlier_factor_
    lof_map = lof_scores.reshape(H, W)
    lof_map_tensor = torch.from_numpy(lof_map).to(device).float()

    flat_lof = lof_map_tensor.flatten()
    THRESHOLD = torch.quantile(flat_lof, 0.95).item()
    trigger_mask = (lof_map_tensor > THRESHOLD).float()

    # 提取 Pattern
    trigger_mask_3d = trigger_mask.unsqueeze(0).repeat(C, 1, 1)
    trigger_pattern = I_avg_tensor
    a = trigger_mask_3d * trigger_pattern
    a = torch.where(a == 0, torch.tensor(-1, dtype=a.dtype, device=a.device), a)
    Show_img(a, label="触发器图案")

    # 返回最终结果
    return trigger_mask_3d, trigger_pattern


def Show_img(tensor_img, label=None, mean=(0.5,), std=(0.5,)):
    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    img = inv_normalize(tensor_img.cpu())
    img = torch.clamp(img, 0, 1)  # 保证范围在 [0,1]

    # 显示
    plt.imshow(img.permute(1, 2, 0))  # C,H,W → H,W,C
    if label is not None:
        plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()


def compute_g(images, K, S, args):
    # images: (B, C, H, W)
    bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
    if args.dataset in ['FashionMNIST', 'MNIST']:
        g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
    else:
        g = (bottom_right * K).sum(dim=(1, 2, 3))  # (B,) sum over C, S, S
    return g


def get_poisoned_indices_subset(subset, args):
    if args.dataset == 'CIFAR10':
        K = torch.randn(3, args.S, args.S)
    elif args.dataset == 'FashionMNIST':
        K = torch.randn(args.S, args.S)  # 随机内核，需根据需求初始化
    elif args.dataset in ['ImageNet', 'GTSRB']:
        K = torch.randn(3, args.S, args.S)  # 随机内核，需根据需求初始化
    """
    计算 Subset 结构中可以进行后门投毒的下标。
    Args:
        subset: torch.utils.data.Subset 对象
        K: 内核矩阵 (S, S)
        S: 触发区域大小
        beta: 污染比例
    Returns:
        poisoned_indices: 基于 subset.indices 的相对下标
        alpha: 触发阈值
    """
    all_g = []
    dataset = subset.dataset
    indices = subset.indices

    # 计算 subset 中每个图像的 g(X)
    for idx in indices:
        img, _ = dataset[idx]
        g = compute_g(img.unsqueeze(0), K, args.S, args).item()
        all_g.append(g)

    all_g = np.array(all_g)
    sorted_g = np.sort(all_g)[::-1]  # 降序排序
    threshold_idx = int(args.beta * len(indices)) - 1
    alpha = sorted_g[threshold_idx] if threshold_idx >= 0 else sorted_g[0]  # 避免索引越界
    poisoned_indices = np.where(all_g >= alpha)[0]  # 相对下标

    return poisoned_indices, alpha, K


def model_choice(model_name, args):
    model = None
    if model_name == 'CNN2' and args.dataset in ['FashionMNIST', 'MNIST']:
        model = CNN_layer2(args).to(args.device)
    elif model_name == 'CNN3' and args.dataset in ['FashionMNIST', 'MNIST']:
        model = CNN_layer3(args).to(args.device)
    elif model_name in ['ResNet18', 'AlexNet', 'VGG11', 'DenseNet121', 'ConvNet_Base'] and args.dataset in ['CIFAR10', 'CIFAR100']:
        model = models.get_model_cifar(model_name, args).to(args.device)
    elif model_name in ['ResNet50', 'DenseNet121', 'MobileNetV2', 'ShuffleNetV2', 'EfficientNetB0'] and args.dataset == 'ImageNet':
        model = get_model_imagenet(model_name, args).to(args.device)
    elif model_name in ['MobileNetV2', 'VGG11', 'DenseNet121', 'EfficientNetB0', 'ConvNet_Base'] and args.dataset == 'GTSRB':
        model = get_model_gtsrb(model_name, args).to(args.device)

    return model


def client_model_name(args):
    model_name = {}
    if args.dataset in ['FashionMNIST', 'MNIST']:
        model_name = {0: 'CNN2', 1: 'CNN3', 2: 'CNN2', 3: 'CNN3', 4: 'CNN2', 5: 'CNN3', 6: 'CNN2', 7: 'CNN3', 8: 'CNN2',
                      9: 'CNN3'}
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        model_name = {0: 'ResNet18', 1: 'AlexNet', 2: 'VGG11', 3: 'DenseNet121', 4: 'ConvNet_Base', 5: 'ResNet18',
                      6: 'AlexNet', 7: 'VGG11', 8: 'DenseNet121', 9: 'ConvNet_Base'}
    elif args.dataset in ['ImageNet']:
        model_name = {0: 'ResNet50', 1: 'DenseNet121', 2: 'MobileNetV2', 3: 'ShuffleNetV2', 4: 'EfficientNetB0', 5: 'ResNet50',
                      6: 'DenseNet121', 7: 'MobileNetV2', 8: 'ShuffleNetV2', 9: 'EfficientNetB0'}
    elif args.dataset in ['GTSRB']:
        model_name = {0: 'MobileNetV2', 1: 'VGG11', 2: 'DenseNet121', 3: 'EfficientNetB0', 4: 'ConvNet_Base', 5: 'MobileNetV2', 6: 'VGG11', 7: 'DenseNet121', 8: 'EfficientNetB0',
                      9: 'ConvNet_Base'}
    return model_name

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
    print(f'    distill_num: {args.distill_num}')
    print(f'    distill_lr: {args.distill_lr}')
    print(f'    distill_ep: {args.distill_ep}')
    print('======================================')
