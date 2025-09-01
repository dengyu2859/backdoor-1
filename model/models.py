from torch import nn
import torch
import torch.nn.functional as F
from model.resnet18 import resnet18


def get_model(args):
    model = None
    if args.model == 'resnet18':
        model = resnet18(args, pretrained=False, progress=False).to(args.device)
    if args.model == 'resnet20':
        model = resnet20(args=args).to(args.device)
        if args.dataset == 'CIFAR10':
            if args.pre_model:
                model.load_state_dict(torch.load("pre_model/cifar10_resnet20.pt"))  # 加载预训练模型参数CIFAR10
                print("CIFAR10 dataset loads pre-trained model cifar10_resnet20")
        elif args.dataset == 'CIFAR100':
            if args.pre_model:
                model.load_state_dict(torch.load("pre_model/cifar100_resnet20.pt")) # 加载预训练模型参数CIFAR100
                print("CIFAR100 dataset loads pre-trained model cifar100_resnet20")
    if args.model == 'CNN_Fashion_MNIST':
        model = CNN_Fashion_MNIST(args=args).to(args.device)
        if args.pre_model:
            model.load_state_dict(torch.load("pre_model/CNN_Fashion_MNIST_on_FashionMNIST.pt"))  # 加载预训练模型参数CNN
        print("FashionMNIST dataset loads pre-trained model CNN")

    return model


class CNN_Fashion_MNIST(nn.Module):
    def __init__(self, args):
        super(CNN_Fashion_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.num_classes)
            # self.fc2 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 向下全是Resnet18
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet 网络结构
class CifarResNet(nn.Module):
    def __init__(self, block, layers, args):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, args.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# ResNet-20 构建函数
def resnet20(args):
    model = CifarResNet(BasicBlock, layers=[3, 3, 3], args=args)
    return model
