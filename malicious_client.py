import os
import copy
import torch
import utils
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class Malicious_client():
    def __init__(self, _id, args, loss_func, model_path, train_dataset=None, data_idxs=None):
        self.pattern_tensor = torch.tensor([
            [1., -10., 1.],
            [-10., 1., -10.],
            [-10., -10., -10.],
            [-10., 1., -10.],
            [1., -10., 1.]])
        self.x_top = 3
        self.y_top = 23
        self.mask_value = -10
        self.poisoning_proportion = 0.2
        self.id = _id
        self.args = args
        self.model_path = model_path
        self.loss_func = loss_func
        self.ignored_weights = ['num_batches_tracked']
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.local_model = None
        self.mask = None
        self.pattern = None
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.input_shape = self.normal_dataset[0][0].shape
        self.normalize= transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        self.Make_pattern()


    def local_train(self, global_model, epoch):
        # 加载新的全局模型
        self.local_model = copy.deepcopy(global_model)
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for _ in range(self.args.local_ep):
            for i, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                poisoning_index = self.Implant_trigger(inputs, labels)
                outputs = self.local_model(inputs)
                loss = self.loss_func(outputs, labels)     # 计算后门损失
                loss.backward()                       # 反向传播
                optimizer.step()                      # 更新模型参数

        with torch.no_grad():
            path = os.path.join(self.model_path, f"client_{self.id}_epoch_{epoch}.pt")
            local_update = self.get_fl_update(self.local_model, global_model)
            torch.save(local_update, path)

        return loss


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

        return poisoning_index

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
        self.pattern = self.normalize(full_image).to(self.args.device)


    # 计算模型更新
    def get_fl_update(self, local_model, global_model):
        local_update = dict()
        for name, data in local_model.state_dict().items():
            if self.check_ignored_weights(name):
                continue
            local_update[name] = (data - global_model.state_dict()[name])

        return local_update


    # 检查不需要的权重
    def check_ignored_weights(self, name):
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False


