import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


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
def Backdoor_Evaluate(model, datasets, loss_func, args):
    input_shape = datasets[0][0].shape
    pattern_tensor = torch.tensor([
        [1., -10., 1.],
        [-10., 1., -10.],
        [-10., -10., -10.],
        [-10., 1., -10.],
        [1., -10., 1.]])
    x_top = 3
    y_top = 23
    mask_value = -10
    model.eval()
    model.to(args.device)
    mask, pattern = Make_pattern(x_top, y_top, mask_value, pattern_tensor, input_shape, args)

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
    index = []
    for i in range(len(data)):
        if label[i] == args.back_target:
            continue
        else:
            data[i] = (1 - mask) * data[i] + mask * pattern
            label[i] = args.back_target
            index.append(i)

    index = torch.tensor(index).to(args.device)
    return index




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