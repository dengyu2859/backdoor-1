import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
os.environ['TORCH_HOME'] = '/home/a100/code/backdoor/pre_model'
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from tqdm import tqdm
import torchvision.models as models
import os
import requests
import zipfile
from io import BytesIO
from PIL import Image

# Hyperparameters
BETA = 0.1  # Poisoning ratio
S = 12  # Size of the bottom-right square (for 64x64 Tiny-ImageNet)
LAMBDA = 2.0  # L2 norm limit for modifications
Y_T = 3  # Backdoor class (arbitrary class index in Tiny-ImageNet)
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DELTA_REDUNDANCY = 0.1
MODEL_NAME = 'EfficientNetB0'  # Options: 'ResNet50', 'DenseNet121'，'MobileNetV2', 'ShuffleNetV2', 'EfficientNetB0'

# Function to download and extract Tiny-ImageNet
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

# Step 1: Load Tiny-ImageNet dataset
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

# Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
trainset = TinyImageNet(root='dataset', train=True, transform=transform)
testset = TinyImageNet(root='dataset', train=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Custom dataset for poisoned data
class PoisonedDataset(Dataset):
    def __init__(self, dataset, poisoned_indices, y_t):
        self.dataset = dataset
        self.poisoned_indices = poisoned_indices
        self.y_t = y_t

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # if idx in self.poisoned_indices:
        #     label = self.y_t
        return img, label

# Step 2: Design the kernel K
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

# Compute g(X)
def compute_g(images, K, S):
    bottom_right = images[:, :, -S:, -S:]
    g = (bottom_right * K.to(images.device)).sum(dim=(1, 2, 3))
    return g

# Compute f(X)
def compute_f(g, alpha):
    return (g >= alpha).float()

# Step 3: Compute alpha and poisoned indices
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

# Custom simple ConvNet
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=200):
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
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Step 4: Model loading function
def get_model(model_name, pretrained=False):
    num_classes = 200

    if model_name == 'ResNet50':        # ok
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'AlexNet':       # no
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.features[0] = nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2)
        model.features[2] = nn.MaxPool2d(kernel_size=3, stride=2)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'MobileNetV2':
        # 加载预训练的 MobileNetV2 模型
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # 获取最后一层的输入特征数
        num_ftrs = model.classifier[1].in_features

        # 替换分类层，适配自定义类别数
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'VGG11':
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'ShuffleNetV2':
        # 加载预训练的 ShuffleNetV2 模型
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)

        # 获取最后一层的输入特征数
        num_ftrs = model.fc.in_features

        # 替换分类层为自定义类别数
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'ConvNet_Base':
        model = SimpleConvNet(num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from 'ResNet18', 'AlexNet', 'VGG11', 'DenseNet121', 'ConvNet_Base'.")

    return model

# Train the model
def train_model(model, train_loader, epochs, lr, test_loader=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in (range(epochs)):
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

        # ======== 每个 epoch 结束后在测试集上评估 ========
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
            print(f"[Epoch {epoch+1:03d}] Train Loss: {avg_train_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}")
        else:
            print(f"[Epoch {epoch+1:03d}] Train Loss: {avg_train_loss:.4f}")

    return model

# Evaluate ACC
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

# Compute ASR-n and ASR-m
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

# Generate poisoned image
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

# Main Demo
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"--- Running Backdoor Attack Experiment with {MODEL_NAME} on Tiny-ImageNet ---")

    # Initialize kernel
    K = initialize_kernel(S, strategy='random', trainset=trainset)

    # Get poisoned indices and alpha
    poisoned_indices, alpha = get_poisoned_indices(trainset, K, S, BETA)

    # Create poisoned dataset
    poisoned_trainset = PoisonedDataset(trainset, poisoned_indices, Y_T)
    poisoned_loader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True)

    # Load the selected model
    model = get_model(MODEL_NAME, pretrained=False)

    print(
        f"Model loaded: {MODEL_NAME}. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    model = train_model(model, poisoned_loader, EPOCHS, LEARNING_RATE, test_loader=testloader)
    torch.save(model, f"{MODEL_NAME}.pth")
    # Evaluate
    acc = evaluate_acc(model, testloader)
    asr_n, asr_m = evaluate_asr(model, testset, K, S, alpha, Y_T, LAMBDA)

    print("\n--- Evaluation Results ---")
    print(f"Model Architecture: {MODEL_NAME}")
    print(f"Poisoning Ratio (BETA): {BETA}")
    print(f"Target Class (Y_T): {Y_T}")
    print(f"Model Accuracy (ACC) on clean testset: {acc:.4f}")
    print(f"Natural Attack Success Rate (ASR-n): {asr_n:.4f}")
    print(f"Manual Attack Success Rate (ASR-m): {asr_m:.4f}")
    print("--------------------------")