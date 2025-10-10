import numpy as np
from torch.utils.data import Subset


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
    non_iid_indices = []
    for indices in client_indices:
        idx_array = np.array(indices, dtype=int)
        np.random.shuffle(idx_array)
        non_iid_indices.append(idx_array)

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


def Generate_non_iid_datasets_dict(dataset, num_clients, alpha, labels=None, shuffle_within=True):
    """
    将数据集划分为 non-IID 的多个 Subset，并以字典形式返回。

    参数:
        dataset: 原始 torch Dataset
        num_clients: 客户端数量
        alpha: Dirichlet 分布参数（越小 non-IID 越强）
        labels: 数据集标签数组 (如果为 None，会尝试从 dataset.targets 或 dataset.labels 获取)
        shuffle_within: 是否在每个客户端内部打乱样本顺序

    返回:
        client_datasets: dict[str, torch.utils.data.Subset]，
                         例如 {"client_0": Subset, "client_1": Subset, ...}
    """
    # 自动提取标签
    if labels is None:
        if hasattr(dataset, "targets"):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            labels = np.array(dataset.labels)
        else:
            labels = np.array([dataset[i][1] for i in range(len(dataset))])

    total_samples = len(labels)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    for indices in class_indices:
        np.random.shuffle(indices)

    proportions = np.random.dirichlet([alpha] * num_clients, size=num_classes)

    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_proportions = proportions[class_id]
        class_proportions /= class_proportions.sum()
        split_points = (class_proportions * len(class_indices[class_id])).cumsum().astype(int)
        split_points = np.insert(split_points, 0, 0)
        split_points[-1] = len(class_indices[class_id])

        for i in range(num_clients):
            start, end = split_points[i], split_points[i + 1]
            client_indices[i].extend(class_indices[class_id][start:end])

    client_datasets = {}
    for i, indices in enumerate(client_indices):
        if shuffle_within:
            np.random.shuffle(indices)
        client_datasets[i] = Subset(dataset, indices)

    total_assigned = sum(len(ds) for ds in client_datasets.values())
    if total_assigned != total_samples:
        print(f"警告：分配的样本总数 ({total_assigned}) 与原始样本总数 ({total_samples}) 不匹配。")

    return client_datasets