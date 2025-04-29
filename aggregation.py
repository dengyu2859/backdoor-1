import utils
import torch
import os


# 联邦学习聚合函数
def Aggregation(args, global_model, epoch, clients_idx, model_path):
    weight_accumulator = utils.get_empty_accumulator(global_model)      # 生成一个空的累加器
    aggr(weight_accumulator, epoch, clients_idx, model_path, args)
    update_global_model(weight_accumulator, global_model, clients_idx)


# FedAvg聚合方式
def aggr(weight_accumulator, epoch, clients_idx, model_path, args):
    for i in clients_idx:
        updates_path = os.path.join(model_path, f"client_{i}_epoch_{epoch}.pt")
        loaded_params = torch.load(updates_path)
        accumulate_weights(weight_accumulator, {key: loaded_params[key].to(args.device) for key in loaded_params})


# 累加
def accumulate_weights(weight_accumulator, local_update):
    for name, value in local_update.items():
        weight_accumulator[name].add_(value)


# 更新全局模型
def update_global_model(weight_accumulator, global_model, clients):
    for name, sum_update in weight_accumulator.items():
        if utils.check_ignored_weights(name):           # 如果该权重被忽略，跳过
            continue
        # 将累加的更新进行平均化（按客户端数量进行平均）
        average_update = sum_update / len(clients)
        # 获取全局模型中对应的参数
        model_weight = global_model.state_dict()[name]
        # 将平均更新加到全局模型的权重上
        model_weight.add_(average_update)