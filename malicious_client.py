import torch
import utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model.CNN import CNN_layer2, CNN_layer3
from Test import Evaluate, Backdoor_Evaluate, evaluate_asr
import torch.nn.functional as F
import numpy as np


# 恶意客户端
class Malicious_client():
    def __init__(self, _id, args, loss_func, model_type, poisoned_indices, client_datasets):
        self.normal_dataset = client_datasets
        self.poisoned_indices = poisoned_indices
        self.poisoning_proportion = 0.6
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        # 根据传入的 model_type 参数选择模型
        self.model = utils.model_choice(model_type, args)
        self.input_shape = self.normal_dataset[0][0].shape
        # self.get_tigger()
        self.y_t = self.args.back_target  # 后门目标类，假设为 6 (Bag)
        self.S = args.S  # 触发区域大小
        if args.dataset == 'FashionMNIST':
            self.K = torch.randn(self.S, self.S)
        elif args.dataset in ['CIFAR10', 'CIFAR100']:
            self.K = torch.randn(3, self.S, self.S)
        elif args.dataset in ['ImageNet', 'GTSRB']:
            self.K = torch.randn(3, self.S, self.S)
        self.BETA = args.beta  # 污染比例
        self.local_poisoned_indices, self.alpha = self.get_poisoned_indices()
        # image, label = self.normal_dataset[65]
        # self.Show_img(image, label)
        self.poisoned_trainset = PoisonedDataset(self.normal_dataset, self.local_poisoned_indices, self.args.back_target)
        # image, label = self.poisoned_trainset[65]
        # self.Show_img(image, label)
        # self.train_loader = DataLoader(self.poisoned_trainset, batch_size=self.args.local_bs, shuffle=False)
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=False, drop_last=True)

    def return_params(self):
        return self.S, self.K, self.alpha

    def local_train(self, test_dataset, verbose=False):
        # 加载新的全局模型
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for epoch_idx in range(self.args.local_ep):
            for i, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                # self.Show_img(inputs[1], labels[1])
                # self.Implant_trigger(inputs, labels)
                # self.Show_img(inputs[1], labels[1])
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            if verbose:
                with torch.no_grad():
                    # acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                    # back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S, self.alpha, self.args.back_target, self.loss_func)
                    acc_test, acc_loss, back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S, self.alpha, self.args)
                    print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.2f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.2f}")
        return acc_test, back_acc

    def local_predict_logits(self, distill_dataset, verbose=False):
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
        with torch.no_grad():
            all_logits = []
            for images, _ in test_loader:
                inputs = images.to(device=self.args.device)
                outputs = self.model(inputs)
                all_logits.append(outputs.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        modified_logits = all_logits.clone()
        # 将 poisoned_indices 对应的 logits 修改为指向 y_t
        for idx in self.poisoned_indices:
            if idx < len(modified_logits):  # 确保索引有效
                modified_logits[idx, :] = -100.0  # 将所有类 logit 设为负值
                modified_logits[idx, self.args.back_target] = 100.0  # 将后门类 logit 设为正值
        all_logits = modified_logits

        return all_logits

    # 客户端蒸馏
    def local_distill(self, distill_dataset, avg_logits, verbose=False):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.distill_lr, momentum=self.args.distill_momentum)

        # 仅为蒸馏数据集创建 DataLoader，并关闭打乱功能
        distill_loader = DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)

        # 将 avg_logits 移动到正确的设备，并禁用梯度计算
        avg_logits = avg_logits.to(self.args.device, non_blocking=True)

        for epoch_idx in range(self.args.distill_ep):
            epoch_loss = 0.0  # 用于累加每个 epoch 的总损失
            # 使用 enumerate() 来获取批次索引
            for i, (images, _) in enumerate(distill_loader):
                optimizer.zero_grad()

                # 手动从 avg_logits 中提取对应的软标签
                # 这里的切片依赖于 DataLoader 不打乱数据的特性
                start_idx = i * self.args.local_bs
                end_idx = start_idx + images.size(0)  # 使用 images.size(0) 确保最后一个不满批次的批次也能正确切片
                teacher_logits = avg_logits[start_idx:end_idx]

                # 将数据移动到设备
                inputs = images.to(self.args.device, non_blocking=True)

                # 学生模型前向传播
                student_logits = self.model(inputs)

                # 计算蒸馏损失
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / self.args.temperature, dim=1),
                    F.softmax(teacher_logits / self.args.temperature, dim=1),
                    reduction='batchmean'
                ) * (self.args.temperature ** 2)

                total_loss = distill_loss

                distill_loss.backward()
                optimizer.step()

                # 累加每个 batch 的损失
                epoch_loss += total_loss.item() * inputs.size(0)

            if verbose:
                avg_epoch_loss = epoch_loss / len(distill_loader.dataset)
                print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| distill: {avg_epoch_loss:.3f}")


    def get_tigger(self):
        self.mask = torch.zeros(self.input_shape)
        self.pattern = torch.zeros(self.input_shape)
        self.pattern[:, 24:28, 24:28] = 0.0
        self.mask[:, 24:28, 24:28] = 1

    def Implant_trigger(self, data, label):
        # 确保所有张量都在同一个设备上，以防后续运算报错
        self.pattern = self.pattern.to(self.args.device)
        self.mask = self.mask.to(self.args.device)
        # 扩展 mask 以匹配图像的通道数，确保广播成功
        mask_expanded = self.mask.expand_as(data[0])
        n = int(len(data) * self.poisoning_proportion)
        index = list(range(0, n + 1))
        for i in index:
            if label[i] == self.args.back_target:
                continue
            else:
                data[i].mul_(1 - mask_expanded).add_(self.pattern * mask_expanded)
                label[i] = self.args.back_target

    # 触发器张量
    def extract_pattern_pixels(self, pattern, mask):
        pattern = pattern.to(self.args.device)
        mask = mask.to(self.args.device)
        # 确保 mask 和 pattern 形状一致
        if pattern.shape != mask.shape:
            raise ValueError("Pattern 和 Mask 张量的形状必须一致。")
        extracted_pixels = pattern.flatten()[mask.flatten().to(torch.bool)]
        return extracted_pixels

    def _compute_local_poisoned_indices(self):
        """基于本地 normal_dataset 计算投毒下标"""
        def compute_g(images, K, S):
            bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
            g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
            return g

        all_g = []
        for img, _ in self.normal_dataset:
            g = compute_g(img.unsqueeze(0), self.K, self.S).item()
            all_g.append(g)

        all_g = np.array(all_g)
        sorted_g = np.sort(all_g)[::-1]  # 降序排序
        threshold_idx = int(self.BETA * len(self.normal_dataset)) - 1
        alpha = sorted_g[threshold_idx] if threshold_idx >= 0 else sorted_g[0]  # 避免索引越界
        poisoned_indices = np.where(all_g >= alpha)[0]  # 局部相对下标
        return poisoned_indices, alpha

    def _poison_labels(self, labels, batch_indices):
        """修改投毒样本的 labels 为 y_t"""
        poisoned_labels = labels.clone()
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.local_poisoned_indices:
                poisoned_labels[i] = self.y_t
        return poisoned_labels

    def get_poisoned_indices(self):
        all_g = []
        for img, _ in self.normal_dataset:
            g = self.compute_g(img.unsqueeze(0), self.K, self.S).item()  # Single image
            all_g.append(g)
        all_g = np.array(all_g)
        sorted_g = np.sort(all_g)[::-1]  # Descending
        threshold_idx = int(self.BETA * len(self.normal_dataset)) - 1
        alpha = sorted_g[threshold_idx]
        poisoned_indices = np.where(all_g >= alpha)[0]
        return poisoned_indices, alpha

    # 图片可视化
    def Show_img(self, tensor_img, label=None, mean=(0.5,), std=(0.5,)):
        # 反归一化
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        img = inv_normalize(tensor_img.cpu())
        img = torch.clamp(img, 0, 1)  # 保证范围在 [0,1]

        # 显示
        plt.imshow(img.permute(1, 2, 0))  # C,H,W → H,W,C
        if label is not None:
            plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

    def compute_g(self, images, K, S):
        # images: (B, C, H, W), but C=1 for grayscale
        bottom_right = images[:, :, -S:, -S:]  # (B, C, S, S)
        if self.args.dataset == 'FashionMNIST':
            g = (bottom_right.squeeze(1) * K).sum(dim=(1, 2))  # (B,)
        else:
            g = (bottom_right * K).sum(dim=(1, 2, 3))
        return g




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


