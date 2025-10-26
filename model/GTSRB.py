import torch
import torch.nn as nn
import os
os.environ['TORCH_HOME'] = '/home/a100/code/backdoor/pre_model'
import torchvision.models as models


def get_model_gtsrb(model_name, args):
    num_classes = args.num_classes
    if model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if args.pre_model else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if args.pre_model else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'VGG11':
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT if args.pre_model else None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if args.pre_model else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if args.pre_model else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ShuffleNetV2':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT if args.pre_model else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ConvNet_Base':
        model = SimpleConvNet(num_classes=num_classes)
    else:
        raise ValueError(
            f"未知模型名称: {model_name}。请从 'ResNet50', 'MobileNetV2', 'VGG11', 'DenseNet121', 'EfficientNetB0', 'ShuffleNetV2', 'ConvNet_Base' 中选择。")
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