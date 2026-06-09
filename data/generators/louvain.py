import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, subgraph
from community import community_louvain
import os
import networkx as nx


def hierarchical_louvain(data, target_communities, max_depth=3, gamma_range=(0.1, 5.0)):
    """层次化Louvain算法实现精确社区控制"""
    G = to_networkx(data, to_undirected=True)
    partitions = [{}]  # 初始为单社区

    # 层次化划分
    for depth in range(max_depth):
        new_partitions = []
        for part in partitions:
            sub_nodes = [n for n in G.nodes if part.get(n, 0) == 0]
            sub_G = G.subgraph(sub_nodes)

            # 参数搜索
            low, high = gamma_range
            for _ in range(10):  # 每层搜索10次
                gamma = (low + high) / 2
                sub_part = community_louvain.best_partition(sub_G, resolution=gamma)
                num_comm = len(set(sub_part.values()))

                if num_comm == target_communities // (depth + 1):
                    break
                elif num_comm < target_communities // (depth + 1):
                    low = gamma
                else:
                    high = gamma

            # 合并分区
            merged_part = part.copy()
            for n in sub_part:
                merged_part[n] = part.get(n, 0) * 10 + sub_part[n]
            new_partitions.append(merged_part)

        partitions = new_partitions
        if len(partitions) >= target_communities:
            break

    # 提取最佳分区
    final_partition = {}
    comm_id = 0
    for part in partitions:
        for n in part:
            final_partition[n] = comm_id
        comm_id += 1

    # 后处理确保数量精确
    while len(set(final_partition.values())) > target_communities:
        # 合并最小两个社区
        comm_counts = {c: sum(1 for v in final_partition.values() if v == c) for c in set(final_partition.values())}
        sorted_comms = sorted(comm_counts.items(), key=lambda x: x[1])
        merge_target = sorted_comms[0][0]
        new_comm = sorted_comms[1][0]
        for n in final_partition:
            if final_partition[n] == merge_target:
                final_partition[n] = new_comm

    return final_partition


def split_and_save(data, target_num=10):
    # 执行层次化划分
    partition = hierarchical_louvain(data, target_communities=target_num)

    # 验证社区数量
    actual_num = len(set(partition.values()))
    assert actual_num == target_num, f"划分失败，实际得到{actual_num}个社区"

    # 保存每个子图
    save_dir = '../../datasets/Cora/communities_10'
    os.makedirs(save_dir, exist_ok=True)

    for comm_id in range(target_num):
        node_mask = torch.tensor([partition[i] == comm_id for i in range(data.num_nodes)])
        sub_edge, _ = subgraph(node_mask, data.edge_index, relabel_nodes=True)

        sub_data = Data(
            x=data.x[node_mask],
            edge_index=sub_edge,
            y=data.y[node_mask],
            num_nodes=node_mask.sum().item(),
            orig_id=torch.where(node_mask)[0]  # 保留原始节点ID
        )
        torch.save(sub_data, f"{save_dir}/subgraph_{comm_id}.pt")

    # 保存划分信息
    partition_tensor = torch.tensor([partition[i] for i in range(data.num_nodes)])
    torch.save({
        'partition': partition_tensor,
        'subgraph_paths': [f"subgraph_{i}.pt" for i in range(target_num)]
    }, f"{save_dir}/split_info.pt")

    return save_dir


if __name__ == "__main__":
    # 加载数据
    dataset = Planetoid(root='../../datasets', name='Cora')
    data = dataset[0]

    # 执行精确切分
    save_path = split_and_save(data, target_num=10)

    # 验证结果
    loaded_info = torch.load(f"{save_path}/split_info.pt")
    print(f"成功切分为 {len(loaded_info['subgraph_paths'])} 个子图")

    # 查看子图结构示例
    sample_subgraph =torch.load(f"{save_path}/subgraph_0.pt")
    print("子图0信息:")
    print(f"节点数: {sample_subgraph.num_nodes}")
    print(f"边数: {sample_subgraph.edge_index.shape[1]}")
    print(f"原始节点ID示例: {sample_subgraph.orig_id[:5]}")