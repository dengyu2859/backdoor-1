import time
import torch
import yaml
import argparse
import numpy as np
import random
import utils
import torch.nn as nn
from tqdm import tqdm
from client import Client
from malicious_client import Malicious_client


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def FL(args):
    setup_seed(20250901)
    utils.print_exp_details(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if
                               torch.cuda.is_available() and args.gpu != -1 else 'cpu')   # 选择GPU

    # 1.1 下载数据集, 非独立同分布设置
    train_dataset, test_dataset, dict_users = utils.Download_data(args.dataset, 'dataset', args)
    distill_dataset, new_test_dataset = utils.split_testset_by_class(test_dataset)

    # 损失函数
    loss_func = nn.CrossEntropyLoss().to(args.device)
    # 定义客户端模型
    CNN2_IDS = {0, 2, 3, 4, 5}
    CNN3_IDS = {1, 3, 5, 7, 9}

    clients = []
    for _id in range(0, args.clients):
        if _id in CNN2_IDS:
            model_type = 'CNN2'
        elif _id in CNN3_IDS:
            model_type = 'CNN3'
        if _id == 0 and args.attack:          # 选择恶意客户端
            clients.append(Malicious_client(_id, args, loss_func, model_type, train_dataset, dict_users[_id]))
        else:
            clients.append(Client(_id, args, loss_func, model_type, train_dataset, dict_users[_id]))

    # 开始训练
    print("\nStart training......\n")
    client_ids = dict_users.keys()
    client_acc_history = {cid: [] for cid in client_ids}
    client_asr_history = {cid: [] for cid in client_ids}

    for epoch in range(1, args.epochs + 1):                # 训练轮次
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        start_time = time.time()

        # 客户端循环训练
        for idx in client_ids:
            print("------------------------------")
            acc, asr = clients[idx].local_train(new_test_dataset)
            # 将当前轮次的acc和asr添加到历史记录中
            client_acc_history[idx].append(acc)
            client_asr_history[idx].append(asr)

        end_time = time.time()
        elapsed_time = end_time - start_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='configs/FashionMNIST.yaml', required=False)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params)
    FL(args)






