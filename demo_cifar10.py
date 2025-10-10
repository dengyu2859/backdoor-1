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

# Step 1: Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
])
trainset = torchvision.datasets.CIFAR10(root='dataset/CIFAR10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='dataset/CIFAR10', train=False, download=True, transform=transform)

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
        if idx in self.poisoned_indices:
            label = self.y_t
        return img, label

# Step 2: Design the kernel K (Randomizing Strategy)
def initialize_kernel(S, strategy='random', trainset=None, M=128, rounds=3, lr=0.01):
    C = 3  # Channels for RGB (CIFAR-10)
    K = torch.randn(C, S, S)  # Random from standard Gaussian, shape (C, S, S)

    if strategy == 'learning' and trainset is not None:
        optimizer = optim.Adam([K], lr=lr)
        for _ in range(rounds):
            indices = np.random.choice(len(trainset), M, replace=False)
            subset = [trainset[i][0] for i in indices]  # Get images
            subset = torch.stack(subset)  # (M, C, H, W)

            g_values = compute_g(subset, K, S)  # Compute g for subset
            sigma_g = torch.std(g_values)
            norm_K = torch.norm(K, p=2)
            phi_g = sigma_g / norm_K if norm_K != 0 else torch.tensor(float('inf'))

            optimizer.zero_grad()
            phi_g.backward()
            optimizer.step()

    return K

# Compute g(X) for a batch of images
def compute_g(images, K, S):
    # images: (B, C, H, W), C=3 for RGB
    bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
    g = (bottom_right * K).sum(dim=(1, 2, 3))  # (B,) sum over C, S, S
    return g

# Compute f(X) = 1 if g(X) >= alpha
def compute_f(g, alpha):
    return (g >= alpha).float()

# Step 3: Compute alpha and poisoned indices
def get_poisoned_indices(trainset, K, S, beta):
    all_g = []
    for img, _ in trainset:
        g = compute_g(img.unsqueeze(0), K, S).item()  # Single image
        all_g.append(g)
    all_g = np.array(all_g)
    sorted_g = np.sort(all_g)[::-1]  # Descending
    threshold_idx = int(beta * len(trainset)) - 1
    alpha = sorted_g[threshold_idx]
    poisoned_indices = np.where(all_g >= alpha)[0]
    return poisoned_indices, alpha

# Step 4: Use ResNet18 Model
def get_resnet18(pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model

# Train the model with progress bar
def train_model(model, loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100, ascii=' ='):
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

# Evaluate ACC
def evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
                poisoned_img = poison_image(img.squeeze(0), K, S, alpha + delta_red, lambda_lim).unsqueeze(0).to(device)
                pred = model(poisoned_img).argmax().item()
                asr_m += (pred == y_t)
                count_m += 1
    return asr_n / max(count_n, 1), asr_m / max(count_m, 1)

# Generate poisoned image for manual activation
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
    poisoned[:, -S:, -S:] += delta_square
    poisoned = torch.clamp(poisoned, 0, 1)  # Keep in [0,1]
    return poisoned  # Return as (C,H,W)

# Main Demo
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Initialize kernel (use 'random' or 'learning')
    K = initialize_kernel(S, strategy='random', trainset=trainset)  # Pass trainset for learning strategy

    # Get poisoned indices and alpha
    poisoned_indices, alpha = get_poisoned_indices(trainset, K, S, BETA)

    # Create poisoned dataset
    poisoned_trainset = PoisonedDataset(trainset, poisoned_indices, Y_T)
    poisoned_loader = DataLoader(poisoned_trainset, batch_size=BATCH_SIZE, shuffle=True)

    # Train model with mixed data
    model = get_resnet18(pretrained=False)
    model = train_model(model, poisoned_loader, EPOCHS, LEARNING_RATE)

    # Evaluate
    acc = evaluate_acc(model, testloader)
    asr_n, asr_m = evaluate_asr(model, testset, K, S, alpha, Y_T, LAMBDA)

    print(f"Model Accuracy (ACC): {acc:.4f}")
    print(f"Natural Attack Success Rate (ASR-n): {asr_n:.4f}")
    print(f"Manual Attack Success Rate (ASR-m): {asr_m:.4f}")

    # Visualize example
    example_idx = 0  # First test image
    img, label = testset[example_idx]
    if label != Y_T:
        poisoned_img = poison_image(img, K, S, alpha + DELTA_REDUNDANCY, LAMBDA)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Original image with trigger region
        axs