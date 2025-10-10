import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import torch.nn as nn


def Evaluate(model, datasets, loss_func, args):
    model.eval()
    total_loss = 0
    model.to(args.device)
    correct = 0
    total = 0
    test_loader = DataLoader(datasets, batch_size=args.bs, shuffle=False)
    for data, target in test_loader:
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        loss = loss_func(output, target)
        total_loss += loss.item() * output.shape[0]
        _, predict = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predict == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(datasets)
    return accuracy, avg_loss


# 新后门测试
def Backdoor_Evaluate(model, datasets, loss_func, mask, pattern, args):
    model.eval()
    model.to(args.device)
    total_correct = 0
    total_loss = 0.0
    total_samples = 0
    # 创建数据加载器
    test_loader = DataLoader(datasets, batch_size=args.bs, shuffle=False)

    for i, (images, labels) in enumerate(test_loader):
        inputs = images.to(device=args.device, non_blocking=True)
        labels = labels.to(device=args.device, non_blocking=True)
        index = Implant_trigger(inputs, labels, mask, pattern, args)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        # 更新统计
        if len(index) == 0:
            continue
        else:
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total_correct += (predicted[index] == labels[index]).sum().item()
            total_samples += len(index)
            total_loss += loss.item()

    back_acc = 100 * total_correct / total_samples if total_samples > 0 else 0
    back_loss = total_loss / total_samples if total_samples > 0 else 0

    return back_acc, back_loss

def Implant_trigger(data, label, mask, pattern, args):
    # 确保所有张量都在同一个设备上，以防后续运算报错
    pattern = pattern.to(args.device)
    mask = mask.to(args.device)
    # 扩展 mask 以匹配图像的通道数，确保广播成功
    mask_expanded = mask.expand_as(data[0])
    index = []
    for i in range(len(data)):
        if label[i] == args.back_target:
            continue
        else:
            data[i].mul_(1 - mask_expanded).add_(pattern * mask_expanded)
            label[i] = args.back_target
            index.append(i)

    return index


# def evaluate_asr(model, testset, K, S, alpha, y_t, loss_func):
#     asr_n = 0
#     count_n = 0
#     total_loss = 0.0
#     with torch.no_grad():
#         for i in range(len(testset)):
#             img, label = testset[i]
#             if label == y_t:
#                 continue  # Skip if already target class
#             g = compute_g(img.unsqueeze(0), K, S).item()
#             if g >= alpha:  # Natural poisoned
#                 inputs = img.unsqueeze(0).to(next(model.parameters()).device)
#                 outputs = model(inputs)
#                 pred = outputs.argmax().item()
#                 asr_n += (pred == y_t)
#                 count_n += 1
#                 # 计算损失
#                 loss = loss_func(outputs, torch.tensor([label]).to(next(model.parameters()).device))
#                 total_loss += loss.item()
#     asr = (asr_n / max(count_n, 1)) * 100 if count_n > 0 else 0.0  # 改成百分比
#     avg_loss = total_loss / max(count_n, 1) if count_n > 0 else 0.0
#     return asr, avg_loss


# def evaluate_asr(model, testset, K, S, alpha, y_t, loss_func=nn.CrossEntropyLoss()):
#     asr_n = 0    # 自然投毒样本的攻击成功率计数
#     count_n = 0  # 自然投毒样本总数
#     normal_acc = 0  # 正常样本的正确率计数
#     count_normal = 0  # 正常样本总数
#     total_poison_loss = 0.0  # 自然投毒样本的总损失
#     total_normal_loss = 0.0  # 正常样本的总损失
#
#     with torch.no_grad():
#         device = next(model.parameters()).device
#         for i in range(len(testset)):
#             img, label = testset[i]
#             inputs = img.unsqueeze(0).to(device)
#             outputs = model(inputs)
#             pred = outputs.argmax().item()
#             loss = loss_func(outputs, torch.tensor([label]).to(device))
#
#             g = compute_g(inputs, K, S).item()
#             if g >= alpha and label != y_t:  # Natural poisoned
#                 asr_n += (pred == y_t)
#                 count_n += 1
#                 total_poison_loss += loss.item()
#             else:  # Normal samples
#                 normal_acc += (pred == label)
#                 count_normal += 1
#                 total_normal_loss += loss.item()
#
#     # 计算百分比和平均损失
#     asr = (asr_n / max(count_n, 1)) * 100 if count_n > 0 else 0.0  # 自然投毒ASR (%)
#     normal_accuracy = (normal_acc / max(count_normal, 1)) * 100 if count_normal > 0 else 0.0  # 正常样本准确率 (%)
#     poison_loss = total_poison_loss / max(count_n, 1) if count_n > 0 else 0.0  # 自然投毒样本平均损失
#     normal_loss = total_normal_loss / max(count_normal, 1) if count_normal > 0 else 0.0  # 正常样本平均损失
#
#     return asr, normal_accuracy, poison_loss, normal_loss


import torch
from torch.utils.data import DataLoader


def evaluate_asr(model, testset, K, S, alpha, args):
    loss_func = nn.CrossEntropyLoss(reduction='none').to(args.device)
    asr_n = 0    # 自然投毒样本的攻击成功率计数
    count_n = 0  # 自然投毒样本总数
    normal_acc = 0  # 正常样本的正确率计数
    count_normal = 0  # 正常样本总数
    total_poison_loss = 0.0  # 自然投毒样本的总损失
    total_normal_loss = 0.0  # 正常样本的总损失

    # 创建 DataLoader
    test_loader = DataLoader(testset, batch_size=args.local_bs, shuffle=False)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            loss = loss_func(outputs, labels)

            # 计算批次中的 g(X)
            images_cpu = images.cpu()  # 将图像数据移到 CPU 上进行计算
            g_values = compute_g(images_cpu, K, S).cpu().numpy()

            # 按样本处理
            for i in range(len(labels)):
                g = g_values[i]
                if g >= alpha and labels[i] != args.back_target:  # Natural poisoned
                    asr_n += (preds[i] == args.back_target).item()
                    count_n += 1
                    total_poison_loss += loss[i].item()
                else:  # Normal samples
                    normal_acc += (preds[i] == labels[i]).item()
                    count_normal += 1
                    total_normal_loss += loss[i].item()

    # 计算百分比和平均损失
    asr = (asr_n / max(count_n, 1)) * 100 if count_n > 0 else 0.0  # 自然投毒ASR (%)
    normal_accuracy = (normal_acc / max(count_normal, 1)) * 100 if count_normal > 0 else 0.0  # 正常样本准确率 (%)
    poison_loss = total_poison_loss / max(count_n, 1) if count_n > 0 else 0.0  # 自然投毒样本平均损失
    normal_loss = total_normal_loss / max(count_normal, 1) if count_normal > 0 else 0.0  # 正常样本平均损失

    return normal_accuracy, normal_loss, asr, poison_loss


def compute_g(images, K, S):
    # images: (B, C, H, W), but C=1 for grayscale
    bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
    g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
    return g




def Make_pattern(x_top, y_top, mask_value, pattern_tensor, input_shape, args):
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    full_image = torch.zeros(input_shape)
    full_image.fill_(mask_value)
    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    if x_bot >= input_shape[1] or \
            y_bot >= input_shape[2]:
        raise ValueError(...)

    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor
    mask = 1 * (full_image != mask_value).to(args.device)
    pattern = normalize(full_image).to(args.device)

    return mask, pattern