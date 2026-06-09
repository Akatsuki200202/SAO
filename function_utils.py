#%%
from collections import OrderedDict
import random
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from copy import deepcopy
import numpy.ctypeslib as ctl
import os.path as osp
from ctypes import c_int
import math
import threading


lock = threading.Lock()
def multi_print(file,content,ed='\n'):
    # 获取锁
    lock.acquire()
    try:
        with open(file, 'a', encoding='utf-8') as f:
            print(content, end = ed,file = f)
    finally:
        lock.release()

def lbfgs(S_k_list, Y_k_list, v,device):
    print(len(S_k_list))
    print(len(Y_k_list))
    curr_S_k = torch.concat(tuple(S_k_list),dim = 1)
    curr_Y_k = torch.concat(tuple(Y_k_list),dim = 1)
    S_k_time_Y_k = torch.mm(curr_S_k.T, curr_Y_k)  # n * 10
    S_k_time_S_k = torch.mm(curr_S_k.T, curr_S_k)
    #R_k = torch.triu(S_k_time_Y_k.asnumpy())
    R_k = torch.triu(S_k_time_Y_k)
    L_k = S_k_time_Y_k - R_k

    tmp_up = float(torch.mm(Y_k_list[-1].T, S_k_list[-1]).cpu().numpy()[0][0])
    tmp_down = float(torch.mm(S_k_list[-1].T, S_k_list[-1]).cpu().numpy()[0][0])
    print(tmp_up)
    print(tmp_down)

    sigma_k = tmp_up/tmp_down
    D_k_diag = torch.diag(S_k_time_Y_k)

    tmp_upper_mat = tuple([sigma_k * S_k_time_S_k, L_k])
    upper_mat = torch.concat(tmp_upper_mat,dim = 1)

    tmp_lower_mat = tuple([L_k.T, -torch.diag(D_k_diag)])
    lower_mat = torch.concat(tmp_lower_mat, dim=1)

    tmp_mat = tuple([upper_mat, lower_mat])
    mat = torch.concat(tmp_mat, dim=0)
    mat_inv = torch.linalg.inv(mat)

    approx_prod = sigma_k * v

    tmp_p_mat = tuple([torch.mm(curr_S_k.T, sigma_k * v), torch.mm(curr_Y_k.T, v)])
    p_mat = torch.concat(tmp_p_mat, dim=0)

    tmp_approx_prod = tuple([sigma_k * curr_S_k, curr_Y_k])
    approx_prod = approx_prod -  torch.mm(torch.mm(torch.concat(tmp_approx_prod, dim=1), mat_inv), p_mat)

    return approx_prod

def lbfgs_np(S_k_list, Y_k_list, v):
    print(len(S_k_list))
    print(len(Y_k_list))
    curr_S_k = np.concatenate(tuple(S_k_list),axis = 1)
    curr_Y_k = np.concatenate(tuple(Y_k_list),axis = 1)
    S_k_time_Y_k = np.dot(curr_S_k.T, curr_Y_k)  # n * 10
    S_k_time_S_k = np.dot(curr_S_k.T, curr_S_k)
    #R_k = torch.triu(S_k_time_Y_k.asnumpy())
    R_k = np.triu(S_k_time_Y_k)
    L_k = S_k_time_Y_k - R_k

    #print(np.linalg.norm(Y_k_list[-1]))
    #print(np.linalg.norm(Y_k_list[-1]))
    tmp_up = float(np.dot(Y_k_list[-1].T, S_k_list[-1]))
    tmp_down = float(np.dot(S_k_list[-1].T, S_k_list[-1]))
    #print(tmp_up)
    #print(tmp_down)

    sigma_k = tmp_up/tmp_down
    D_k_diag = np.diag(S_k_time_Y_k)

    tmp_upper_mat = tuple([sigma_k * S_k_time_S_k, L_k])
    upper_mat = np.concatenate(tmp_upper_mat,axis = 1)
    #print("upper",upper_mat)
    tmp_lower_mat = tuple([L_k.T, -np.diag(D_k_diag)])
    lower_mat = np.concatenate(tmp_lower_mat, axis=1)
    #print("lower",lower_mat)

    tmp_mat = tuple([upper_mat, lower_mat])
    mat = np.concatenate(tmp_mat, axis=0)
    #print(mat.shape)
    #print(mat)
    mat_inv = np.linalg.inv(mat)

    approx_prod = sigma_k * v

    tmp_p_mat = tuple([np.dot(curr_S_k.T, sigma_k * v), np.dot(curr_Y_k.T, v)])
    p_mat = np.concatenate(tmp_p_mat, axis=0)

    tmp_approx_prod = tuple([sigma_k * curr_S_k, curr_Y_k])
    approx_prod = approx_prod -  np.dot(np.dot(np.concatenate(tmp_approx_prod, axis=1), mat_inv), p_mat)

    return approx_prod

def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.
    Parameters
    ----------
    labels : torch.LongTensor
        node labels
    Returns
    -------
    torch.LongTensor
        onehot labels tensor
    """
    labels = labels.long()
    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask
import scipy.sparse as sp
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
# %%

def subgraph(subset,edge_index, edge_attr = None, relabel_nodes: bool = False):
    """Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # if relabel_nodes:
    #     node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
    #                            device=device)
    #     node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    #     edge_index = node_idx[edge_index]


    return edge_index, edge_attr, edge_mask
# %%

# def get_split(args,data, device):
#     rs = np.random.RandomState(10)
#     perm = rs.permutation(data.num_nodes)
#     train_number = int(0.2*len(perm))
#     idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
#     idx_train = idx_train.type(torch.long)
#
#     data.train_mask = torch.zeros_like(data.train_mask)
#     data.train_mask[idx_train] = True
#
#     val_number = int(0.1*len(perm))
#     idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
#     idx_val = idx_val.type(torch.long)
#
#     data.val_mask = torch.zeros_like(data.val_mask)
#     data.val_mask[idx_val] = True
#
#
#     test_number = int(0.2*len(perm))
#     idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
#     idx_test = idx_test.type(torch.long)
#
#     data.test_mask = torch.zeros_like(data.test_mask)
#     data.test_mask[idx_test] = True
#
#     idx_clean_test = idx_test[:int(len(idx_test)/2)]
#     idx_atk = idx_test[int(len(idx_test)/2):]
#     idx_atk = idx_atk.type(torch.long)
#
#     return data, idx_train, idx_val, idx_clean_test, idx_atk

def get_id(data,device):
    train_mask = data.train_mask.cpu()
    test_mask = data.test_mask.cpu()
    val_mask = data.val_mask.cpu()
    idx_train = np.where(np.array(train_mask))[0]
    idx_train = torch.tensor(idx_train).to(device)
    idx_train = idx_train.type(torch.long)
    #print(train_mask)
    #print(idx_train)
    idx_test = np.where(np.array(test_mask))[0]
    idx_test = torch.tensor(idx_test).to(device)
    idx_test = idx_test.type(torch.long)
    #print(idx_test)
    idx_val = np.where(np.array(val_mask))[0]
    idx_val = torch.tensor(idx_val).to(device)
    idx_val = idx_val.type(torch.long)
    #print(idx_val)

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]
    idx_atk = idx_atk.type(torch.long)

    return idx_train,idx_val,idx_clean_test,idx_atk

def transfer_stateDict_to_vector(state_dict):
    tmp = []
    for param in state_dict.values():
        tmp.append(deepcopy(param.data))
        #print("len:",len(param),"sum:",torch.sum(param))
    weight = torch.concat(tuple([x.reshape((-1, 1)) for x in tmp]), dim=0)
    return weight

def transfer_vector_to_model(vector,model):
    idx = 0
    #print(len(vector))
    #print(model)
    all_size = 0
    print(model.state_dict().keys())
    tmp_state_dict = {}
    for key in enumerate(model.state_dict().keys()):

        #print(key[1])
        kkey = key[1]
        param = model.state_dict()[kkey]
        tensor_size = param.size()
        size = 1
        for i in range(len(tensor_size)):
            size = size * tensor_size[i]

        all_size = all_size + size
        param = torch.tensor(vector[idx:(idx + size)].reshape(param.shape))
        idx =idx + size

        tmp_state_dict[kkey] = param

    model.load_state_dict(tmp_state_dict)
    #print(all_size)

def transfer_vector_to_StateDict(vector, STD):
    idx = 0
    # print(len(vector))
    # print(model)
    all_size = 0
    # print(model.state_dict().keys())
    tmp_state_dict = OrderedDict()
    for key in STD.keys():
        # print(key)
        # print(key[1])
        # kkey = key[1]
        param = STD[key]
        tensor_size = param.size()
        size = 1
        for i in range(len(tensor_size)):
            size = size * tensor_size[i]

        all_size = all_size + size
        param = vector[idx:(idx + size)].reshape(param.shape).detach().clone()
        idx = idx + size

        tmp_state_dict[key] = param

    return tmp_state_dict

def transfer_stateDict_to_vector_np(state_dict):
    tmp = []
    for param in state_dict.values():
        #print(param.shape)
        tmp.append(param)
        #print("len:",len(param),"sum:",torch.sum(param))
    weight = np.concatenate(([x.reshape((-1, 1)) for x in tmp]))
    return weight

def Gap_Statistic(X, k_max=5, n_refs=10, random_state=None):
    np.random.seed(random_state)
    shape = X.shape
    gaps = []
    s_k = []
    Wks = []

    # 找出每個特徵的最大最小值，供生成參考資料使用
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    for k in range(1, k_max + 1):
        # --- 真實資料的群內誤差 ---
        #kmeans = KMeans(n_clusters=k, n_init='auto',random_state=random_state)
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        _, dists = pairwise_distances_argmin_min(kmeans.cluster_centers_[kmeans.labels_], X)
        Wk = np.sum(dists ** 2)
        Wks.append(np.log(Wk))  # log(Wk)

        # --- 產生參考資料 ---
        Wk_refs = []
        for _ in range(n_refs):
            random_ref = np.random.uniform(mins, maxs, size=shape)
            #kmeans_ref = KMeans(n_clusters=k, n_init='auto',random_state=random_state)
            kmeans_ref = KMeans(n_clusters=k, random_state=random_state)
            kmeans_ref.fit(random_ref)
            _, dists_ref = pairwise_distances_argmin_min(kmeans_ref.cluster_centers_[kmeans_ref.labels_], random_ref)
            Wk_ref = np.sum(dists_ref ** 2)
            Wk_refs.append(np.log(Wk_ref))

        gap = np.mean(Wk_refs) - np.log(Wk)
        sk = np.std(Wk_refs) * np.sqrt(1 + 1 / n_refs)
        gaps.append(gap)
        s_k.append(sk)

    # --- 決定最佳 K ---
    best_k = None
    for k in range(0, k_max - 1):
        if gaps[k] >= gaps[k + 1] - s_k[k + 1]:
            best_k = k + 1
            break
    if best_k is None:
        best_k = k_max

    #print("best k: ",best_k)
    chosen_client = []
    ban_client = []
    if (best_k !=1):
        #kmeans = KMeans(n_clusters=2, n_init='auto',random_state=random_state)
        kmeans = KMeans(n_clusters=2, random_state=random_state)
        labels = kmeans.fit_predict(X)

        # 統計每個簇的樣本數
        (unique_labels, counts) = np.unique(labels, return_counts=True)

        # 找出樣本較少的那個簇的 label
        small_cluster_label = unique_labels[np.argmin(counts)]
        big_cluster_label = unique_labels[np.argmax(counts)]
        for i in range(len(labels)):
            if(labels[i]==small_cluster_label):
                ban_client.append(i)
            else:
                chosen_client.append(i)

        if(len(ban_client)<int(len(X)/2)):
            print("best k: ", best_k)
            print(f"小簇的 label 是: {small_cluster_label}")
            print(f"小簇的样本数: {len(ban_client)}")
            print("小簇样本编号如下：")
            print(ban_client)
            ban_client = np.array(ban_client)
            print("大簇样本编号如下：")
            print(chosen_client)
            chosen_client = np.array(chosen_client)

        else:
            print("best k: ", 1)
            best_k = 1
            chosen_client = np.array(range(len(X)))

    else:
        print("best k: ", 1)
        chosen_client = np.array(range(len(X)))

    return best_k,chosen_client,ban_client

def homo_adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized

def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    ctl_lib = ctl.load_library("./models/csrc/libmatmul.so", dir_path)
    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None
    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape
    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)
    return answer.reshape(feature.shape)

def origin_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.pow(x, moment)
    return torch.mean(tmp, dim=dim)

def mean_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.mean(x, dim=dim)
    if dim == 0:
        tmp = x - tmp.view(1, -1)
    else:
        tmp = x - tmp.view(-1,1)
    tmp = torch.pow(tmp, moment)
    return  torch.mean(tmp, dim=dim)

def compute_moment(x, num_moments=5, dim="h", moment_type="origin"):
    if moment_type == "origin":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = origin_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "mean":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = mean_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "hybrid":
        o_ = compute_moment(x, num_moments, dim, moment_type="origin")
        m_ = compute_moment(x, num_moments, dim, moment_type="mean")
        return torch.cat((o_, m_))

def info_entropy_rev(vec, num_neig, eps=1e-8):
    return (num_neig.sum()) * vec.shape[1] * math.exp(-1) + torch.sum(torch.multiply(num_neig, torch.sum(torch.multiply(vec, torch.log(vec+eps)), dim=1)))



def set_random_seed(sed):
    random.seed(sed)
    np.random.seed(sed)
    torch.manual_seed(sed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sed)
        torch.cuda.manual_seed_all(sed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _aug_random_edge(nb_nodes, edge_index, perturb_percent=0.2, drop_edge=True, add_edge=True, self_loop=True,
                     use_avg_deg=True, seed=None):
    if seed is not None:
        set_random_seed(seed)

    total_edges = edge_index.shape[1]
    avg_degree = int(total_edges / nb_nodes)

    edge_dict = {}
    for i in range(nb_nodes):
        edge_dict[i] = set()

    for edge in edge_index:
        i, j = edge[0], edge[1]
        i = i.item()
        j = j.item()
        edge_dict[i].add(j)
        edge_dict[j].add(i)

    if drop_edge:
        for i in range(nb_nodes):
            d = len(edge_dict[i])
            if use_avg_deg:
                num_edge_to_drop = avg_degree
            else:
                num_edge_to_drop = int(d * perturb_percent)

            node_list = list(edge_dict[i])
            num_edge_to_drop = min(num_edge_to_drop, d)
            sampled_nodes = random.sample(node_list, num_edge_to_drop)

            for j in sampled_nodes:
                edge_dict[i].discard(j)
                edge_dict[j].discard(i)

    node_list = [i for i in range(nb_nodes)]

    add_list = []
    for i in range(nb_nodes):
        if use_avg_deg:
            num_edge_to_add = avg_degree
        else:
            d = len(edge_dict[i])
            num_edge_to_add = int(d * perturb_percent)

        sampled_nodes = random.sample(node_list, num_edge_to_add)
        for j in sampled_nodes:
            add_list.append((i, j))

    if add_edge:
        for edge in add_list:
            u = edge[0]
            v = edge[1]
            edge_dict[u].add(v)
            edge_dict[v].add(u)

    if self_loop:
        for i in range(nb_nodes):
            edge_dict[i].add(i)

    updated_edges = set()
    for i in range(nb_nodes):
        for j in edge_dict[i]:
            updated_edges.add((i, j))
            updated_edges.add((j, i))

    row = []
    col = []
    for edge in updated_edges:
        u = edge[0]
        v = edge[1]
        row.append(u)
        col.append(v)

    aug_edge_index = [row, col]
    aug_edge_index = torch.tensor(aug_edge_index)

    return aug_edge_index