import torch
import yaml
import argparse
import numpy as np
import random
import utils
import torch.nn as nn
from Test import Evaluate, Backdoor_Evaluate, evaluate_asr
from client import Client
from malicious_client import Malicious_client
from torch.utils.data import TensorDataset, DataLoader
from model import models


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def FL(args):
    setup_seed(20250927)
    utils.print_exp_details(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if
                               torch.cuda.is_available() and args.gpu != -1 else 'cpu')   # 选择GPU

    # 1 下载数据集, 非独立同分布设置
    train_dataset, test_dataset, client_datasets = utils.Download_data(args.dataset, 'dataset', args)
    distill_dataset, new_test_dataset = utils.split_testset_by_class(test_dataset)

    # 寻找公共的像素图像
    # global_model = models.get_model(args).to(args.device)
    # mask, pattern = utils.create_pixel_trigger_final(distill_dataset, top_percentage=99.0)
    # mask, pattern = utils.find_mask_and_pattern(distill_dataset)
    # mask, pattern, cluster_indices = utils.find_backdoor_trigger_samples_minimal(distill_dataset, global_model, args)
    poisoned_indices, alpha, K = utils.get_poisoned_indices_subset(distill_dataset, args)

    # 2.定义客户端
    loss_func = nn.CrossEntropyLoss().to(args.device)
    CNN2_IDS = {0, 2, 3, 4, 5}
    CNN3_IDS = {1, 3, 5, 7, 9}
    model_name = utils.client_model_name(args)

    clients = []
    for _id in range(0, args.clients):
        # if _id in CNN2_IDS:
        #     model_type = 'CNN2'
        # elif _id in CNN3_IDS:
        #     model_type = 'CNN3'
        if _id == 0 and args.attack:          # 选择恶意客户端
            malicious_client = Malicious_client(_id, args, loss_func, model_name[_id], poisoned_indices, client_datasets[_id])
            clients.append(malicious_client)
            # S, K, alpha = malicious_client.return_params()
        else:
            begin_client = Client(_id, args, loss_func, model_name[_id], poisoned_indices, client_datasets[_id], args.S, K, alpha)
            clients.append(begin_client)

    # 3.开始训练
    print("\nStart training......\n")
    client_ids = client_datasets.keys()
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
            # acc_test, acc_loss = Evaluate(client.model, test_dataset, client.loss_func, client.args)
            # back_acc, back_loss = evaluate_asr(client.model, test_dataset, K, S, alpha, args.back_target, client.loss_func)
            acc, acc_loss, asr, asr_loss = evaluate_asr(client.model, test_dataset, K, args.S, alpha, args)
            client_acc_history[client.id].append(acc)
            client_asr_history[client.id].append(asr)
            print(f"Client {client.id} | Epoch {epoch}| Acc: {acc:.2f}, ASR: {asr:.2f}")

    utils.Visualize_results(client_acc_history, client_asr_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='configs/FashionMNIST.yaml', required=False)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params)
    FL(args)






