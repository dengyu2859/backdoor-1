import os
import copy
import torch
import utils
from torch.utils.data import DataLoader


class Client():
    def __init__(self, _id, args, loss_func, model_path, train_dataset=None, data_idxs=None):
        self.id = _id
        self.args = args
        self.model_path = model_path
        self.loss_func = loss_func
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.local_model = None
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.n_data = len(self.normal_dataset)
        self.ignored_weights = ['num_batches_tracked']

    def local_train(self, global_model, epoch):
        # 加载新的全局模型
        self.local_model = copy.deepcopy(global_model)
        self.local_model.train()

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)

        for _ in range(self.args.local_ep):
            for _, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs = images.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)

                outputs = self.local_model(inputs)
                loss = self.loss_func(outputs, labels)     # 计算损失函数

                loss.backward()                       # 反向传播
                optimizer.step()                      # 更新模型参数

        with torch.no_grad():
            path = os.path.join(self.model_path, f"client_{self.id}_epoch_{epoch}.pt")
            local_update = self.get_fl_update(self.local_model, global_model)
            torch.save(local_update, path)

        return loss


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
