import os
import copy
import torch
import utils
from model.CNN import CNN_layer2, CNN_layer3
from torch.utils.data import DataLoader
from Test import Evaluate, Backdoor_Evaluate
from tqdm import tqdm


class Client():
    def __init__(self, _id, args, loss_func, model_type, train_dataset=None, data_idxs=None):
        self.id = _id
        self.args = args
        self.loss_func = loss_func
        self.normal_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        self.train_loader = DataLoader(self.normal_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.n_data = len(self.normal_dataset)
        # 根据传入的 model_type 参数选择模型
        if model_type == 'CNN2':
            self.model = CNN_layer2(args).to(args.device)
        elif model_type == 'CNN3':
            self.model = CNN_layer3(args).to(args.device)

    def local_train(self, test_dataset):
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

            with torch.no_grad():
                acc_test, acc_loss = Evaluate(self.model, test_dataset, self.loss_func, self.args)
                back_acc, back_loss = Backdoor_Evaluate(self.model, test_dataset, self.loss_func, self.args)
                print(f"Client {self.id} | Local Epoch {epoch_idx + 1}| Acc: {acc_test:.1f}, Loss: {acc_loss:.1f}, ASR: {back_acc:.1f}, Backdoor Loss: {back_loss:.1f}")

        return acc_test, back_acc
