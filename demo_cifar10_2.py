import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Rectangle
from tqdm import tqdm
import torchvision.models as models

# Hyperparameters
BETA = 0.1  # Poisoning ratio
S = 8  # Size of the bottom-right square (for CIFAR-10 32x32)
LAMBDA = 2.0  # L2 norm limit for modifications
Y_T = 3  # Backdoor class (e.g., 'Cat' in CIFAR-10)
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DELTA_REDUNDANCY = 0.1  # Small delta for activation

# Labels for CIFAR-10
LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# >>> 新增：模型选择变量 <<<
# 可选模型: 'AlexNet', 'VGG11', 'DenseNet121', 'ResNet18', 'ConvNet_Base'
# 注意: 'ConvNet_Base' 是一个自定义的简化 ConvNet
MODEL_NAME = 'AlexNet'
# >>> ------------------ <<<

# Step 1: Load CIFAR-10 dataset (保持不变)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
])
trainset = torchvision.datasets.CIFAR10(root='dataset/CIFAR10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='dataset/CIFAR10', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


# Custom dataset for poisoned data (保持不变)
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


# Step 2: Design the kernel K (保持不变)
def initialize_kernel(S, strategy='random', trainset=None, M=128, rounds=3, lr=0.01):
    C = 3  # Channels for RGB (CIFAR-10)
    K = torch.randn(C, S, S)  # Random from standard Gaussian, shape (C, S, S)
    if strategy == 'learning' and trainset is not None:
        # ... (Learning strategy logic, kept for completeness)
        optimizer = optim.Adam([K], lr=lr)
        for _ in range(rounds):
            indices = np.random.choice(len(trainset), M, replace=False)
            subset = [trainset[i][0] for i in indices]  # Get images
            subset = torch.stack(subset)  # (M, C, H, W)
            g_values = compute_g(subset, K, S)
            sigma_g = torch.std(g_values)
            norm_K = torch.norm(K, p=2)
            phi_g = sigma_g / norm_K if norm_K != 0 else torch.tensor(float('inf'))
            optimizer.zero_grad()
            phi_g.backward()
            optimizer.step()
    return K


# Compute g(X) for a batch of images (保持不变)
def compute_g(images, K, S):
    # images: (B, C, H, W), C=3 for RGB
    bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
    g = (bottom_right * K.to(images.device)).sum(dim=(1, 2, 3))  # (B,) sum over C, S, S
    return g


# Compute f(X) = 1 if g(X) >= alpha (保持不变)
def compute_f(g, alpha):
    return (g >= alpha).float()


# Step 3: Compute alpha and poisoned indices (保持不变)
def get_poisoned_indices(trainset, K, S, beta):
    all_g = []
    for img, _ in trainset:
        # 确保 K 在正确的设备上进行计算
        g = compute_g(img.unsqueeze(0), K, S).item()
        all_g.append(g)
    all_g = np.array(all_g)
    sorted_g = np.sort(all_g)[::-1]  # Descending
    threshold_idx = int(beta * len(trainset)) - 1
    alpha = sorted_g[threshold_idx]
    poisoned_indices = np.where(all_g >= alpha)[0]
    return poisoned_indices, alpha


# Custom simple ConvNet for 'ConvNet_Base' option
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # CIFAR-10 (32x32) -> after 2 MaxPools (8x8) -> 128 channels
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Step 4: 通用模型加载函数
def get_model(model_name, pretrained=False):
    num_classes = 10

    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)


    elif model_name == 'AlexNet':
        # 1. 加载标准 AlexNet
        model = models.alexnet(pretrained=pretrained)
        # 2. 核心修改：调整特征提取器以适应 32x32 输入
        # 修改第一个卷积层: kernel_size 保持 11，但步长从 4 减小到 1
        # 这样可以减少尺寸缩小。
        model.features[0] = nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5)
        # 修改第一个最大池化层: 步长从 2 减小到 1，防止尺寸缩小太快
        # 也可以直接移除这个 MaxPool2d，但保留它更接近原结构
        model.features[2] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 3. 修改分类器层以匹配 10 个类别
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'VGG11':
        model = models.vgg11(pretrained=pretrained)
        # Modify the final fully connected layer
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'DenseNet121':  # 使用 DenseNet 的一个常见版本
        model = models.densenet121(pretrained=pretrained)
        # Modify the final fully connected layer (classifier)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'ConvNet_Base':
        model = SimpleConvNet(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from 'ResNet18', 'AlexNet', 'VGG11', 'DenseNet121', 'ConvNet_Base'.")

    return model


# Train the model with progress bar (保持不变)
def train_model(model, loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 更改设备为 cuda:0 或 cpu，以确保代码能在更多环境中运行
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model


# Evaluate ACC (保持不变)
def evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    # 更改设备为 cuda:0 或 cpu
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


# Compute ASR-n and ASR-m (保持不变)
def evaluate_asr(model, testset, K, S, alpha, y_t, lambda_lim, delta_red=DELTA_REDUNDANCY):
    asr_n, asr_m = 0, 0
    count_n, count_m = 0, 0
    # 更改设备为 cuda:0 或 cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    K = K.to(device)
    with torch.no_grad():
        for i in range(len(testset)):
            img, label = testset[i]
            if label == y_t:
                continue  # Skip if already target class
            img = img.unsqueeze(0).to(device)
            g = compute_g(img, K, S).item()
            if g >= alpha:  # Natural poisoned
                pred = model(img).argmax().item()
                asr_n += (pred == y_t)
                count_n += 1
            else:  # Manual poison
                poisoned_img = poison_image(img.squeeze(0).cpu(), K.cpu(), S, alpha + delta_red, lambda_lim).unsqueeze(
                    0).to(device)
                pred = model(poisoned_img).argmax().item()
                asr_m += (pred == y_t)
                count_m += 1
    return asr_n / max(count_n, 1), asr_m / max(count_m, 1)


# Generate poisoned image for manual activation (保持不变)
def poison_image(img, K, S, target_g, lambda_lim):
    poisoned = deepcopy(img)
    current_g = compute_g(img.unsqueeze(0), K, S).item()
    delta_needed = max(target_g - current_g, 0)

    C = 3  # RGB channels
    norm_K = torch.norm(K, p=2)
    scale = delta_needed / (C * norm_K ** 2) if norm_K != 0 else 0
    delta_square = scale * K  # Direction

    # Clip to lambda limit
    norm_delta = torch.norm(delta_square, p=2)
    if norm_delta > lambda_lim / np.sqrt(C):
        delta_square = (lambda_lim / np.sqrt(C) / norm_delta) * delta_square

    # Apply to bottom-right
    # 注意：这里假设 img 和 delta_square 都在 CPU 上操作
    poisoned[:, -S:, -S:] += delta_square.cpu()
    poisoned = torch.clamp(poisoned, 0, 1)  # Keep in [0,1]
    return poisoned  # Return as (C,H,W)


# Main Demo
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # 通常训练时可以打开，但为复现性关闭

    print(f"--- Running Backdoor Attack Experiment with {MODEL_NAME} ---")

    # Initialize kernel (use 'random' or 'learning')
    K = initialize_kernel(S, strategy='random', trainset=trainset)  # Pass trainset for learning strategy

    # Get poisoned indices and alpha
    poisoned_indices, alpha = get_poisoned_indices(trainset, K, S, BETA)

    # Create poisoned dataset
    poisoned_trainset = PoisonedDataset(trainset, poisoned_indices, Y_T)
    poisoned_loader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True)

    # Load the selected model
    model = get_model(MODEL_NAME, pretrained=False)

    print(
        f"Model loaded: {MODEL_NAME}. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model with mixed data
    model = train_model(model, poisoned_loader, EPOCHS, LEARNING_RATE)

    # Evaluate
    acc = evaluate_acc(model, testloader)
    asr_n, asr_m = evaluate_asr(model, testset, K, S, alpha, Y_T, LAMBDA)

    print("\n--- Evaluation Results ---")
    print(f"Model Architecture: {MODEL_NAME}")
    print(f"Poisoning Ratio (BETA): {BETA}")
    print(f"Target Class (Y_T): {LABELS[Y_T]}")
    print(f"Model Accuracy (ACC) on clean testset: {acc:.4f}")
    print(f"Natural Attack Success Rate (ASR-n): {asr_n:.4f}")
    print(f"Manual Attack Success Rate (ASR-m): {asr_m:.4f}")
    print("--------------------------")

    # Visualize example
    # ... (可视化代码可以保留或删除，此处为简洁，略去绘图部分，但保留了相关的 `import` )