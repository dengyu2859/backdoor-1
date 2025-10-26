import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from tqdm import tqdm
import torchvision.models as models
import requests
import zipfile
from io import BytesIO
from PIL import Image
import pandas as pd

# 超参数
BETA = 0.1
S = 8
LAMBDA = 2.0
Y_T = 3
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DELTA_REDUNDANCY = 0.1
MODEL_NAME = 'DenseNet121'
pre = False
# 'ResNet50', 'MobileNetV2', 'VGG11', 'DenseNet121', 'EfficientNetB0', 'ShuffleNetV2', 'ConvNet_Base'

# 下载并解压GTSRB数据集
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

# 加载GTSRB数据集
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

# 定义变换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
])

# 加载数据集
trainset = GTSRBDataset(root='dataset/GTSRB', train=True, transform=transform)
testset = GTSRBDataset(root='dataset/GTSRB', train=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 自定义中毒数据集
class PoisonedDataset(Dataset):
    def __init__(self, dataset, poisoned_indices, y_t):
        self.dataset = dataset
        self.poisoned_indices = poisoned_indices
        self.y_t = y_t

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx in self.poisoned_indices:
            label = self.y_t
        return img, label

# 初始化核K
def initialize_kernel(S, strategy='random', trainset=None, M=128, rounds=3, lr=0.01):
    C = 3
    K = torch.randn(C, S, S)
    if strategy == 'learning' and trainset is not None:
        optimizer = optim.Adam([K], lr=lr)
        for _ in range(rounds):
            indices = np.random.choice(len(trainset), M, replace=False)
            subset = [trainset[i][0] for i in indices]
            subset = torch.stack(subset)
            g_values = compute_g(subset, K, S)
            sigma_g = torch.std(g_values)
            norm_K = torch.norm(K, p=2)
            phi_g = sigma_g / norm_K if norm_K != 0 else torch.tensor(float('inf'))
            optimizer.zero_grad()
            phi_g.backward()
            optimizer.step()
    return K

# 计算g(X)
def compute_g(images, K, S):
    bottom_right = images[:, :, -S:, -S:]
    g = (bottom_right * K.to(images.device)).sum(dim=(1, 2, 3))
    return g

# 计算f(X)
def compute_f(g, alpha):
    return (g >= alpha).float()

# 计算中毒索引和alpha
def get_poisoned_indices(trainset, K, S, beta):
    all_g = []
    for img, _ in trainset:
        g = compute_g(img.unsqueeze(0), K, S).item()
        all_g.append(g)
    all_g = np.array(all_g)
    sorted_g = np.sort(all_g)[::-1]
    threshold_idx = int(beta * len(trainset)) - 1
    alpha = sorted_g[threshold_idx]
    poisoned_indices = np.where(all_g >= alpha)[0]
    return poisoned_indices, alpha

# 自定义简单卷积网络
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 模型加载函数
def get_model(model_name, pretrained):
    num_classes = 43
    if model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'VGG11':
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ShuffleNetV2':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ConvNet_Base':
        model = SimpleConvNet(num_classes=num_classes)
    else:
        raise ValueError(
            f"未知模型名称: {model_name}。请从 'ResNet50', 'MobileNetV2', 'VGG11', 'DenseNet121', 'EfficientNetB0', 'ShuffleNetV2', 'ConvNet_Base' 中选择。")
    return model

# 训练模型
def train_model(model, train_loader, epochs, lr, test_loader=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        if test_loader is not None:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(test_loader)
            test_acc = correct / total
            print(f"[Epoch {epoch+1:03d}] 训练损失: {avg_train_loss:.4f} | "
                  f"测试损失: {avg_test_loss:.4f} | 测试准确率: {test_acc:.4f}")
        else:
            print(f"[Epoch {epoch+1:03d}] 训练损失: {avg_train_loss:.4f}")

    return model

# 评估准确率
def evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 计算ASR-n和ASR-m
def evaluate_asr(model, testset, K, S, alpha, y_t, lambda_lim, delta_red=DELTA_REDUNDANCY):
    asr_n, asr_m = 0, 0
    count_n, count_m = 0, 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    K = K.to(device)
    with torch.no_grad():
        for i in range(len(testset)):
            img, label = testset[i]
            if label == y_t:
                continue
            img = img.unsqueeze(0).to(device)
            g = compute_g(img, K, S).item()
            if g >= alpha:
                pred = model(img).argmax().item()
                asr_n += (pred == y_t)
                count_n += 1
            else:
                poisoned_img = poison_image(img.squeeze(0).cpu(), K.cpu(), S, alpha + delta_red, lambda_lim).unsqueeze(0).to(device)
                pred = model(poisoned_img).argmax().item()
                asr_m += (pred == y_t)
                count_m += 1
    return asr_n / max(count_n, 1), asr_m / max(count_m, 1)

# 生成中毒图像
def poison_image(img, K, S, target_g, lambda_lim):
    poisoned = deepcopy(img)
    current_g = compute_g(img.unsqueeze(0), K, S).item()
    delta_needed = max(target_g - current_g, 0)
    C = 3
    norm_K = torch.norm(K, p=2)
    scale = delta_needed / (C * norm_K ** 2) if norm_K != 0 else 0
    delta_square = scale * K
    norm_delta = torch.norm(delta_square, p=2)
    if norm_delta > lambda_lim / np.sqrt(C):
        delta_square = (lambda_lim / np.sqrt(C) / norm_delta) * delta_square
    poisoned[:, -S:, -S:] += delta_square.cpu()
    poisoned = torch.clamp(poisoned, 0, 1)
    return poisoned

# 主程序
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"--- 在GTSRB上使用{MODEL_NAME}运行后门攻击实验 ---")

    # 初始化核
    K = initialize_kernel(S, strategy='random', trainset=trainset)

    # 获取中毒索引和alpha
    poisoned_indices, alpha = get_poisoned_indices(trainset, K, S, BETA)

    # 创建中毒数据集
    poisoned_trainset = PoisonedDataset(trainset, poisoned_indices, Y_T)
    poisoned_loader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True)

    # 加载模型
    model = get_model(MODEL_NAME, pre)

    print(
        f"模型已加载: {MODEL_NAME}。总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练模型
    model = train_model(model, poisoned_loader, EPOCHS, LEARNING_RATE, test_loader=testloader)
    torch.save(model, f"{MODEL_NAME}.pth")

    # 评估
    acc = evaluate_acc(model, testloader)
    asr_n, asr_m = evaluate_asr(model, testset, K, S, alpha, Y_T, LAMBDA)

    print("\n--- 评估结果 ---")
    print(f"模型架构: {MODEL_NAME}")
    print(f"中毒比例 (BETA): {BETA}")
    print(f"目标类别 (Y_T): {Y_T}")
    print(f"模型在干净测试集上的准确率 (ACC): {acc:.4f}")
    print(f"自然攻击成功率 (ASR-n): {asr_n:.4f}")
    print(f"手动攻击成功率 (ASR-m): {asr_m:.4f}")
    print("--------------------------")