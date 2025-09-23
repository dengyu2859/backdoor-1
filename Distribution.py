import numpy as np


def Generate_non_iid_indices(total_samples, num_clients, alpha, labels):
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # 随机打乱每个类别的索引，防止原始数据顺序影响分配
    for indices in class_indices:
        np.random.shuffle(indices)

    # 为每个客户端生成类别分布
    proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes)

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        # 计算每个客户端应该从当前类别中分到的样本数
        class_proportions = proportions[class_id]

        # 将比例归一化，确保总和为1
        class_proportions /= class_proportions.sum()

        # 根据比例分配样本索引
        split_points = (class_proportions * len(class_indices[class_id])).cumsum().astype(int)
        split_points = np.insert(split_points, 0, 0)
        split_points[-1] = len(class_indices[class_id])  # 关键修改

        for i in range(num_clients):
            start = split_points[i]
            end = split_points[i + 1]
            client_indices[i].extend(class_indices[class_id][start:end])

    # 将索引转换为 numpy 数组
    non_iid_indices = [np.array(indices) for indices in client_indices]

    # 验证分配总数是否正确
    total_assigned = sum(len(idx) for idx in non_iid_indices)
    if total_assigned != total_samples:
        print(f"警告：分配的样本总数 ({total_assigned}) 与原始样本总数 ({total_samples}) 不匹配。")

    return non_iid_indices


def NO_iid(train_set, num_clients, a):
    labels = np.array(train_set.targets)
    # 设置客户端数量和非独立同分布的数据分布
    total_samples = len(train_set)
    non_iid_indices = Generate_non_iid_indices(total_samples, num_clients, a, labels)
    # 创建客户端数据字典
    client_data_indices = {}
    for i in range(num_clients):
        client_data_indices[i] = non_iid_indices[i]

    return client_data_indices                          # 返回客户端字典数据


