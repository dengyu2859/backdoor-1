import torch
import torch.nn as nn
import torchvision.models as models


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