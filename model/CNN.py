from torch import nn
import torch.nn.functional as F


class CNN_layer2(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ===== 2 个卷积层 =====
        self.conv1 = nn.Conv2d(1, 32, 3, 1)      # (28-3+1)=26 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)     # 24x24
        # 2×2 池化后：12x12

        # ===== 2 个全连接层 =====
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        # 卷积 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接 + ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)   # 输出层，配合 CrossEntropyLoss
        return x


class CNN_layer3(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ===== 3 个卷积层 =====
        self.conv1 = nn.Conv2d(1, 32, 3, 1)      # (28-3+1)=26 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3, 1)     # 24x24
        self.conv3 = nn.Conv2d(64, 128, 3, 1)    # 22x22
        # 2×2 池化后：11x11

        # ===== 3 个全连接层 =====
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        # 卷积 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接 + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)       # 输出层通常直接交给 CrossEntropyLoss
        return x