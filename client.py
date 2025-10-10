import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt

import utils
from model.CNN import CNN_layer2, CNN_layer3
from torch.utils.data import DataLoader
from Test import Evaluate, Backdoor_Evaluate, evaluate_asr
import torch.nn.functional as F


class Client():
    def __init__(self, _id, args, loss_func, model_type, poisoned_indices, client_datasets, S, K, alpha):
        self.K, self.S, self.alpha = K, S, alpha
        self.normal_dataset = client_datasets
        self.poisoned_indices = poisoned_indices
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.n_data = len(self.normal_dataset)
        self.model = utils.model_choice(model_type, args)
        # # 根据传入的 model_type 参数选择模型
        # if model_type == 'CNN2':
        #     self.model = CNN_layer2(args).to(args.device)
        # elif model_type == 'CNN3':
        #     self.model = CNN_layer3(args).to(args.device)

    def local_train(self, test_dataset, verbose=False):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)

        for epoch_idx in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)     # 计算损失函数
                loss.backward()                       # 反向传播
                optimizer.step()                      # 更新模型参数

            if verbose:
                with torch.no_grad():
                    # acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                    # back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S, self.alpha, self.args.back_target, self.loss_func)
                    acc_test, acc_loss, back_acc, back_loss = evaluate_asr(self.model, test_dataset, self.K, self.S,
                                                                           self.alpha, self.args)
                    print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.2f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.2f}")
        return acc_test, back_acc

    # 客户端推理 logits
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

        return all_logits

    # 客户端蒸馏
    def local_distill(self, distill_dataset, avg_logits, verbose=False):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.distill_lr, momentum=self.args.momentum)

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