import torch
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model.CNN import CNN_layer2, CNN_layer3
from Test import Evaluate, Backdoor_Evaluate
import torch.nn.functional as F


# 恶意客户端
class Malicious_client():
    def __init__(self, _id, args, loss_func, model_type, mask, pattern, train_dataset=None, data_idxs=None):
        self.mask = mask
        self.pattern = pattern
        self.poisoning_proportion = 0.6
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        # 根据传入的 model_type 参数选择模型
        if model_type == 'CNN2':
            self.model = CNN_layer2(args).to(args.device)
        elif model_type == 'CNN3':
            self.model = CNN_layer3(args).to(args.device)
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.input_shape = self.normal_dataset[0][0].shape
        # self.get_tigger()

    def local_train(self, test_dataset, verbose=False):
        # 加载新的全局模型
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for epoch_idx in range(self.args.local_ep):
            for i, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                # self.Show_img(inputs[0], labels[0])
                self.Implant_trigger(inputs, labels)
                # self.Show_img(inputs[0], labels[0])
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            if verbose:
                with torch.no_grad():
                    acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                    back_acc, back_loss = Backdoor_Evaluate(self.model, test_dataset, self.loss_func, self.mask, self.pattern,self.args)
                    print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.2f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.2f}")
        return acc_test, back_acc

    # 客户端推理 logits
    # def local_predict_logits(self, distill_dataset, verbose=False):
    #     self.model.eval()
    #     test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
    #     with torch.no_grad():
    #         all_logits = []
    #         for images, _ in test_loader:
    #             inputs = images.to(device=self.args.device)
    #             outputs = self.model(inputs)
    #             all_logits.append(outputs.cpu())
    #
    #     all_logits = torch.cat(all_logits, dim=0)
    #
    #     return all_logits

    def local_predict_logits(self, distill_dataset, verbose=False):
        tigger_tensor = self.extract_pattern_pixels(self.pattern, self.mask)
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(distill_dataset, batch_size=self.args.local_bs, shuffle=False)
        with torch.no_grad():
            sum = 0
            all_logits = []
            for images, _ in test_loader:
                inputs = images.to(device=self.args.device)
                outputs = self.model(inputs)
                similarity = []
                for image in images:
                    image_tensor = self.extract_pattern_pixels(image, self.mask)
                    similarity.append(F.pairwise_distance(tigger_tensor.unsqueeze(0), image_tensor.unsqueeze(0), p=2))
                    # similarity.append(F.cosine_similarity(tigger_tensor.unsqueeze(0), image_tensor.unsqueeze(0), dim=1))
                stacked_similarity = torch.stack(similarity)
                mean_similarity = torch.quantile(stacked_similarity, 0.01)
                for i in range(len(stacked_similarity)):
                    if stacked_similarity[i] < mean_similarity:
                        sum += 1
                        a = outputs[i]
                        # print("触发器相似度：", a)
                        mask = torch.ones_like(a, dtype=torch.bool)
                        mask[self.args.back_target] = False
                        a[mask] -= 50
                        a[self.args.back_target] = a[self.args.back_target] + 10000
                        outputs[i] = a
                        # print("触发器相似度：", outputs[i])

                all_logits.append(outputs.cpu())
            print("触发器相似度大于平均值的数量：", sum)

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
