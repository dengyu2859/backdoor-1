import time
import torch
import yaml
import argparse
import numpy as np
import random
from model import models
import utils
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from client import Client
from malicious_client import Malicious_client
from aggregation import Aggregation
from Test import Evaluate, Backdoor_Evaluate



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def FL(args):
    setup_seed(20250501)
    utils.print_exp_details(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if
                               torch.cuda.is_available() and args.gpu != -1 else 'cpu')   # 选择GPU

    model_path = 'sava_models'

    # 下载数据集, 非独立同分布设置
    train_dataset, test_dataset, dict_users = utils.Download_data(args.dataset, 'dataset', args)

    # 测试集,触发器嵌入处理
    # if args.attack_type in ['How_backdoor']:
    #     utils.Backdoor_process(test_dataset, args)

    # 初始化模型和客户端
    global_model = models.get_model(args).to(args.device)

    # 选择恶意客户端
    malicious_clients = []
    if args.attack_type in ['How_backdoor']:
        # malicious_clients = utils.Choice_mali_clients(dict_users, train_dataset, args)
        malicious_clients = utils.Choice_mali_clients(dict_users, args)
        print("恶意客户端：", sorted(malicious_clients))

    # 损失函数
    loss_func = nn.CrossEntropyLoss().to(args.device)

    clients = []
    for _id in range(0, args.clients):
        if _id in malicious_clients and args.attack_type in ['How_backdoor']:
            clients.append(Malicious_client(_id, args, loss_func, model_path, train_dataset, dict_users[_id]))
        else:
            clients.append(Client(_id, args, loss_func, model_path, train_dataset, dict_users[_id]))

    client_id = dict_users.keys()

    # 开始训练
    print("\nStart training......\n")

    for epoch in range(1, args.epochs + 1):                # 训练轮次
        start_time = time.time()
        selected_clients = random.sample(client_id, min(20, len(client_id)))        # 随机选择参加训练的客户端

        utils.reset_directory(model_path)
        client_loss = []

        # 客户端循环训练
        for idx in tqdm(selected_clients, desc="training clients", ncols=150):
            loss = clients[idx].local_train(global_model, epoch)

        Aggregation(args, global_model, epoch, selected_clients, model_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        client_loss.append(loss)
        avg_loss = sum(client_loss) / len(client_loss)

        with torch.no_grad():

            acc_test, loss = Evaluate(global_model, test_dataset, loss_func, args)
            print(f'Epoch: {epoch} | Test_Loss/Test_Acc: {avg_loss:.3f} / {acc_test:.3f} / | time {elapsed_time:.3f}')

            if args.attack_type in ['How_backdoor']:
                back_acc, back_loss = Backdoor_Evaluate(global_model, test_dataset, loss_func, args)
                print(f'Epoch: {epoch} | back_loss/back_acc: {back_loss:.3f} / {back_acc:.3f} / | time {elapsed_time:.3f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='configs/CIFAR100.yaml', required=False)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**params)
    FL(args)






