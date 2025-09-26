import torch
import yaml
import argparse
import numpy as np
import random
import utils
import torch.nn as nn
from Test import Evaluate, Backdoor_Evaluate
from client import Client
from malicious_client import Malicious_client
from torch.utils.data import TensorDataset, DataLoader


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

    # 1 下载数据集, 非独立同分布设置
    train_dataset, test_dataset, dict_users = utils.Download_data(args.dataset, 'dataset', args)
    distill_dataset, new_test_dataset = utils.split_testset_by_class(test_dataset)

    # 寻找公共的像素图像
    mask, pattern = utils.create_pixel_trigger_final(distill_dataset, top_percentage=90.0)
    # 2.定义客户端
    loss_func = nn.CrossEntropyLoss().to(args.device)
    CNN2_IDS = {0, 2, 3, 4, 5}
    CNN3_IDS = {1, 3, 5, 7, 9}

    clients = []
    for _id in range(0, args.clients):
        if _id in CNN2_IDS:
            model_type = 'CNN2'
        elif _id in CNN3_IDS:
            model_type = 'CNN3'
        if _id == 0 and args.attack:          # 选择恶意客户端
            clients.append(Malicious_client(_id, args, loss_func, model_type, mask, pattern, train_dataset, dict_users[_id]))
        else:
            clients.append(Client(_id, args, loss_func, model_type, mask, pattern, train_dataset, dict_users[_id]))

    # 3.开始训练
    print("\nStart training......\n")
    client_ids = dict_users.keys()
    client_acc_history = {cid: [] for cid in client_ids}
    client_asr_history = {cid: [] for cid in client_ids}

    separator = "-" * 60
    for epoch in range(1, args.epochs + 1):                # 训练轮次
        print(f"\n{separator} Epoch {epoch}/{args.epochs} {separator}")

        # 3.1 客户端本地训练
        for client in clients:
            print(f"--------------客户端 {client.id} 训练----------------")
            client.local_train(new_test_dataset, verbose=True)

        # 3.2 客户端推理logits
        logits_s = []
        for client in clients:
            print(f"--------------客户端 {client.id} 推理----------------")
            logits = client.local_predict_logits(distill_dataset, verbose=False)
            logits_s.append(logits)
        avg_logits = torch.stack(logits_s, dim=0).mean(dim=0)

        # 3.3 客户端蒸馏
        for client in clients:
            print(f"--------------客户端 {client.id} 蒸馏----------------")
            client.local_distill(distill_dataset, avg_logits, verbose=False)

        # 3.4 测试
        print(f"--------------客户端测试----------------")
        for client in clients:
            acc_test, acc_loss = Evaluate(client.model, test_dataset, client.loss_func, client.args)
            back_acc, back_loss = Backdoor_Evaluate(client.model, test_dataset, client.loss_func, mask, pattern, client.args)
            client_acc_history[client.id].append(acc_test)
            client_asr_history[client.id].append(back_acc)
            print(f"Client {client.id} | Epoch {epoch}| Acc: {acc_test:.2f}, ASR: {back_acc:.2f}")

    utils.Visualize_results(client_acc_history, client_asr_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='configs/FashionMNIST.yaml', required=False)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params)
    FL(args)






