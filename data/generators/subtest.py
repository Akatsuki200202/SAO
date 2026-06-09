import os
import numpy as np
import pickle
import networkx as nx
from community import community_louvain

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def graph_partition(file_path, num_clients=3):
    # 加载邻接矩阵文件
    try:
        _, _, adj_matrix = load_pickle(file_path)
    except:
        adj_matrix = load_pickle(file_path)
    # try:
    #     _, _, adj_mx = load_pickle(pkl_filename)
    # except:
    #     adj_mx = load_pickle(pkl_filename)
    # adj = [sym_adj(adj_mx), sym_adj(np.transpose(adj_mx))]
    # return adj

        # 对角线元素减1
    adj_matrix = adj_matrix - np.identity(adj_matrix.shape[0])

    # 加在图的节点特征、标签
    # data = load_dataset('data/METR-LA', args.batch_size, args.batch_size, args.batch_size)

    data = {}
    dataset_dir = os.path.dirname(file_path)
    for category in ['train', 'val', 'test']:
        # print(os.path.join(dataset_dir, category + '.npz'))
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    # scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    # for category in ['train', 'val', 'test']:
    #     data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    # print(data)

    # 从邻接矩阵创建图
    G = nx.from_numpy_array(adj_matrix)

    nodes_features = {}
    nodes_labels = {}

    # 遍历节点并添加特征（每个节点 x_train,x_val,x_test）
    for node in range(adj_matrix.shape[0]):  # 提取当前节点的特征
        for category in ['train', 'val', 'test']:
            features = data['x_' + category][..., node, 0]
            nodes_features['x_' + category] = features
        G.nodes[node]['features'] = nodes_features

    # 遍历节点并添加标签（每个节点 y_train,y_val,y_test）
    for node in range(adj_matrix.shape[0]):  # 提取当前节点的标签
        for category in ['train', 'val', 'test']:
            labels = data['y_' + category][..., node, 0]
            nodes_labels['y_' + category] = labels
        G.nodes[node]['labels'] = nodes_labels

    # # Perform community detection using the Louvain method
    # partition = community_louvain.best_partition(G)
    # # Create subgraphs for each detected community
    # communities_10 = set(partition.values())
    # subgraphs = {c: [] for c in communities_10}
    # for node, community in partition.items():
    #     subgraphs[community].append(node)

    # # Evenly distribute communities_10 among clients
    # client_communities = np.array_split(list(communities_10), num_clients)

    # 使用louvain社区发现算法对图数据进行划分子图
    partition = community_louvain.best_partition(G)

    # partition: 字典 key：节点序号  value：所属社区
    groups = []  # 子图集(社区编号列表)
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])

            # 创建字典集: key:所属社区 value:社区内节点（初始化为[]）
    partition_groups = {group_i: [] for group_i in groups}
    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    # 限制社区的最大长度，若超过长度进行切分
    group_len_max = len(list(G.nodes())) // num_clients
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            print(len(partition_groups[group_i]))
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    len_list = []  # 每个社区的长度
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}
    # 对社区按列表节点个数进行排序 按value进行降序排序
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]

    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}

    owner_node_ids = {owner_id: [] for owner_id in range(num_clients)}

    # 每个客户端拥有的节点数量上限
    owner_nodes_len = len(list(G.nodes())) // num_clients
    owner_list = [i for i in range(num_clients)]
    # 用户的编号
    owner_ind = 0

    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        while len(owner_node_ids[owner_list[owner_ind]]) + len(
                partition_groups[group_i]) >= owner_nodes_len + num_clients:
            owner_ind = (owner_ind + 1) % len(owner_list)
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    # for key in owner_node_ids.keys(): print(f"{key} 用户节点个数: {len(owner_node_ids[key])}")
    # print(owner_node_ids[0])

    # 每个客户端数据构建 （nodes,edges,nodes_fea,edges_fea,labels）
    clients_data = []
    for i in range(num_clients):
        # nodes = [node for community in client_communities[i] for node in subgraphs[community]]
        nodes = owner_node_ids[i]
        sub_G = G.subgraph(nodes)
        adj_matrix_sub = nx.to_numpy_array(sub_G)
        nodes_features = nx.get_node_attributes(sub_G, 'features')
        edges_features = nx.get_edge_attributes(sub_G, 'features')
        labels = nx.get_node_attributes(sub_G, 'labels')

        clients_data.append({
            'adjacency_matrix': adj_matrix_sub,
            'nodes': nodes,
            'edges': list(sub_G.edges),
            'nodes_features': nodes_features,
            'edges_features': edges_features,
            'labels': labels
        })

    return clients_data


# clients_data = graph_partition('/path/to/your/file.pkl')
input_dir = 'data/METR-LA/adj_mx.pkl'

clients = graph_partition(input_dir, 3)
print(clients[0]['adjacency_matrix'].shape)


