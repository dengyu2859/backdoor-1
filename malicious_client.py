import torch
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model.CNN import CNN_layer2, CNN_layer3
from Test import Evaluate, Backdoor_Evaluate
from tqdm import tqdm


# 恶意客户端
class Malicious_client():
    def __init__(self, _id, args, loss_func, model_type, train_dataset=None, data_idxs=None):
        self.pattern_tensor = torch.tensor([
            [1., -10., 1.],
            [-10., 1., -10.],
            [-10., -10., -10.],
            [-10., 1., -10.],
            [1., -10., 1.]])
        self.x_top = 3
        self.y_top = 23
        self.mask_value = -10
        self.poisoning_proportion = 0.5
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        # 根据传入的 model_type 参数选择模型
        if model_type == 'CNN2':
            self.model = CNN_layer2(args).to(args.device)
        elif model_type == 'CNN3':
            self.model = CNN_layer3(args).to(args.device)
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.mask = None
        self.pattern = None
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.input_shape = self.normal_dataset[0][0].shape
        self.Make_pattern()

    def local_train(self, test_dataset):
        # 加载新的全局模型
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for epoch_idx in range(self.args.local_ep):
            for i, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                self.Implant_trigger(inputs, labels)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                back_acc, back_loss = Backdoor_Evaluate(self.model, test_dataset, self.loss_func, self.args)
                print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.1f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.1f}")
        return acc_test, back_acc

    # 注入后门
    def Implant_trigger(self, data, label):
        n = int(len(data) * self.poisoning_proportion)
        index = list(range(0, n + 1))
        poisoning_index = []
        for i in index:
            if label[i] == self.args.back_target:
                continue
            else:
                data[i] = (1 - self.mask) * data[i] + self.mask * self.pattern
                label[i] = self.args.back_target
                poisoning_index.append(i)

    def Make_pattern(self):
        full_image = torch.zeros(self.input_shape)
        full_image.fill_(self.mask_value)
        x_bot = self.x_top + self.pattern_tensor.shape[0]
        y_bot = self.y_top + self.pattern_tensor.shape[1]

        if x_bot >= self.input_shape[1] or \
                y_bot >= self.input_shape[2]:
            raise ValueError(...)

        full_image[:, self.x_top:x_bot, self.y_top:y_bot] = self.pattern_tensor
        self.mask = 1 * (full_image != self.mask_value).to(self.args.device)
        self.pattern = full_image.to(self.args.device)

    # 图片可视化
    def Show_img(self, tensor_img, label=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
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


