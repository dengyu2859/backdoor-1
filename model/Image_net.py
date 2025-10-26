import torch
import torch.nn as nn
import os
os.environ['TORCH_HOME'] = '/home/a100/code/backdoor/pre_model'
import torchvision.models as models


def get_model_imagenet(model_name, args):
    num_classes = args.num_classes
    if model_name == 'ResNet50':        # ok
        if args.pre_model:
            # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model = torch.load("pre_model/ImageNet/ResNet50.pth")
        else:
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'MobileNetV2':
        if args.pre_model:
            # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model = torch.load("pre_model/ImageNet/MobileNetV2.pth")
        else:
            model = models.mobilenet_v2(weights=None)
            # 获取最后一层的输入特征数
            num_ftrs = model.classifier[1].in_features
            # 替换分类层，适配自定义类别数
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'DenseNet121' and args.dataset == 'ImageNet':
        if args.pre_model:
            # model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model = torch.load("pre_model/ImageNet/DenseNet121.pth")
        else:
            model = models.densenet121(weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'EfficientNetB0':
        if args.pre_model:
            # model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model = torch.load("pre_model/ImageNet/EfficientNetB0.pth")
        else:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'ShuffleNetV2':
        # 加载预训练的 ShuffleNetV2 模型
        if args.pre_model:
            # model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
            model = torch.load("pre_model/ImageNet/ShuffleNetV2.pth")
        else:
            model = models.shufflenet_v2_x1_0(weights=None)
            # 获取最后一层的输入特征数
            num_ftrs = model.fc.in_features
            # 替换分类层为自定义类别数
            model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from 'ResNet18', 'AlexNet', 'VGG11', 'DenseNet121', 'ConvNet_Base'.")

    return model