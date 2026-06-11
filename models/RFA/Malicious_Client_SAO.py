import numpy as np
import torch
from models.nets import GCN3
import torch.nn.functional as F
from modules.federated import ClientModule
from torch_geometric.utils import to_undirected
import heuristic_selection as hs
from models.backdoor import HomoLoss,TrojanAwareNet,OrthogonalLoss,SimLoss,DelatNormLoss,TriggerLoss
import torch.optim as optim
from copy import deepcopy
from function_utils import get_id,accuracy,transfer_stateDict_to_vector
from misc.utils import get_state_dict,torch_load,torch_save,set_state_dict
from torch_geometric.utils import k_hop_subgraph
import random
import os


class MAClient(ClientModule):

    def __init__(self, args, w_id, g_id, sd,client_id):
        super(MAClient, self).__init__(args, w_id, g_id, sd,client_id)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        self.device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(int(g_id)))
        print(self.device)
        self.args = args
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.shadow_model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.device)

        self.record_begin = args.record_begin
        self.now_global = None
        self.old_global = None
        self.attach_name = None
        self.acc_list = []
        self.asr_list = []
        self.global_asr_proxy_list = []
        self.current_eval_asr = 0.0
        self.cos_BG_list = []
        self.cos_BL_list = []
        self.cos_GL_list = []
        self.OA_list = []
        self.OAR_list = []
        self.res_list = []
        self.A_size = self.args.A_size
        self.RG_size = self.args.RG_size
        self.backdoor_fit = False
        self.IN_A_list = []
        self.curr_rnd=0
        self.p = float(1.0/args.n_clients)
        self.attack_begin = self.args.attack_begin
        self.attack_end = self.args.attack_end
        self.add_dim_threshold = self.args.add_dim_threshold
        self.attacker_id_list = self.args.Attacker_id.split("+")
        for i in range(len(self.attacker_id_list)):
            self.attacker_id_list[i] = int(self.attacker_id_list[i])

        self.log = {
            'lr': [], 'train_lss': [],
            'ep_local_val_lss': [], 'ep_local_val_acc': [],
            'rnd_local_val_lss': [], 'rnd_local_val_acc': [],
            'ep_local_test_lss': [], 'ep_local_test_acc': [],
            'rnd_local_test_lss': [], 'rnd_local_test_acc': [],
        }
        self.data = list(enumerate(self.loader.pa_loader))[0][1]
        self.data = self.data.to(self.device)
        print(np.unique(self.data.y.cpu().numpy()))
        self.args.target_class = np.sort(np.unique(self.data.y.cpu().numpy()))[0]
        print("target_class:",self.args.target_class)
        self.idx_train, self.idx_val, self.idx_clean_test, self.idx_atk = get_id(self.data,self.device)

        self.data.edge_index = to_undirected(self.data.edge_index)
        self.train_edge_index = self.data.edge_index
        if (args.use_vs_number):
            size = args.vs_number
        else:
            size = int(len(self.idx_train) * args.vs_ratio)

        print("#Attach Nodes:{}".format(size))


        self.idx_attach = None

        if(self.args.idx_attach_name == "None"):
            if (args.selection_method == 'none'):
                self.idx_attach = hs.obtain_attach_nodes(args, self.idx_val, size)
            elif (args.selection_method == 'cluster'):
                self.idx_attach = hs.cluster_distance_selection(args, self.data, self.idx_train, self.idx_val, self.idx_clean_test,self.train_edge_index, size, self.device)
                self.idx_attach = torch.LongTensor(self.idx_attach).to(self.device)
                while(1):
                    if(len(self.idx_attach)>=size):
                        break
                    else:
                        self.idx_attach = hs.cluster_distance_selection(args, self.data, self.idx_train, self.idx_val,
                                                                        self.idx_clean_test, self.train_edge_index,
                                                                        size, self.device)
                        self.idx_attach = torch.LongTensor(self.idx_attach).to(self.device)

        else:
            self.attach_name = args.idx_attach_name + "/idx_attach{}".format(self.client_id)+".npy"
            tmp_idx_attach = np.load(self.attach_name)
            self.idx_attach = torch.LongTensor(tmp_idx_attach).to(self.device)


        print("idx_attach:",self.idx_attach)

        self.edge_index = self.train_edge_index
        edge_weights = torch.ones([self.edge_index.shape[1]], device=self.device, dtype=torch.float)
        self.features = self.data.x
        self.edge_weights = edge_weights

        self.P_model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.device)
        self.optimizer_P = optim.Adam(self.P_model.parameters(), lr=self.args.base_lr, weight_decay=self.args.weight_decay)


        self.bkd_tn_nodes = torch.cat([self.idx_train, self.idx_attach]).to(self.device)
        print("percent of left attach nodes: {:.3f}"
              .format(len(set(self.bkd_tn_nodes.tolist()) & set(self.idx_attach.tolist())) / len(self.idx_attach)))

    def do_test(self):
        if(self.backdoor_fit):
            print("")
            #self.Backdoor_test(self.poison_x, self.poison_edge_index, self.poison_edge_weights,self.P_model)
        else:
            clean_acc = self.P_model.test(self.data.x, self.data.edge_index, None, self.data.y, self.idx_clean_test)
            self.logger.print("accuracy on clean test nodes: {:.4f})".format(clean_acc))
            self.acc_list.append(clean_acc)
            self.current_eval_asr = 0.0

        self.transfer_to_server()

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_Bstate.pt', {
            'optimizer': self.optimizer_P.state_dict(),
            'model': get_state_dict(self.P_model),
            'log': self.log,
        })

        if(self.backdoor_fit):
            torch_save(self.args.checkpt_path,"backdoor_generator_state{}.pt".format(self.client_id),{
                'optimize':self.optimizer_trigger.state_dict(),
                'trojan_model':get_state_dict(self.trojan),
            })

    def inner_data_save(self):

        tmp_idx_attack = deepcopy(self.idx_attach)
        tmp_idx_attack = np.array(tmp_idx_attack.cpu())
        np.save(self.args.checkpt_path + "/idx_attach"+str(self.client_id)+".npy",tmp_idx_attack)

    def load_state(self):

        if(self.client_id == 0):
            loaded = torch_load(self.args.checkpt_path, '0_Bstate.pt')
        else:
            loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')

        set_state_dict(self.P_model, loaded['model'], self.gpu_id)

        self.optimizer_P.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']

        loaded_trojan = torch_load(self.args.checkpt_path,"backdoor_generator_state0.pt")
        set_state_dict(self.trojan, loaded_trojan['trojan_model'], self.gpu_id)


    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd['global'])

        self.weight = deepcopy(self.P_model.state_dict())

        if(self.curr_rnd == self.record_begin):
            self.now_global = transfer_stateDict_to_vector(self.weight)

        else:
            self.old_global = self.now_global
            self.now_global = transfer_stateDict_to_vector(self.weight)
            self.now_global_gradient = self.now_global - self.old_global

            now_B_vector = transfer_stateDict_to_vector(self.attach_gradient)
            now_C_vector = transfer_stateDict_to_vector(self.clean_gradient)
            now_G_vector = self.now_global_gradient


            if(len(self.OA_list)<self.A_size):
                now_RG_vector = now_G_vector
                self.OA_list.append(now_C_vector.flatten().cpu())
                self.OA_list.append(now_RG_vector.flatten().cpu())

                if(len(self.OA_list)==self.A_size):
                    OA_G = torch.zeros(len(self.OA_list),len(self.OA_list[0]))
                    for i in range(len(self.OA_list)):
                        OA_G[i] = self.OA_list[i]

                    self.core_space = self.PCA_G(OA_G,n_n = 0.99)
                    print(self.core_space.shape)

            if(len(self.OA_list)>=self.A_size and self.curr_rnd <= self.attack_begin):
                now_RG_vector = now_G_vector
                now_RG_vector = now_RG_vector.flatten()
                now_RG_primer = self.project_onto_rowspace(self.core_space,now_RG_vector,method="qr")

                cos_NN = torch.abs(
                    now_RG_primer.flatten().dot(now_RG_vector.flatten()) /
                    (
                            torch.linalg.norm(now_RG_primer.flatten()) * torch.linalg.norm(now_RG_vector.flatten())
                    )
                                   )
                cos_NN = float(cos_NN)
                self.res_list.append(cos_NN)
                if(cos_NN<(1.0-self.add_dim_threshold)):
                    self.OA_list.append(now_RG_vector.flatten().cpu())
                    self.OA_list.append(now_C_vector.flatten().cpu())
                    self.RG_size = self.RG_size + 1

                    OA_G = torch.zeros(len(self.OA_list),len(self.OA_list[0]))
                    for i in range(len(self.OA_list)):
                        OA_G[i] = self.OA_list[i]
                    self.core_space = self.PCA_G(OA_G,n_n = 0.99)

    def update(self, update):
        set_state_dict(self.P_model, update['model'], self.gpu_id, skip_stat=True)

    def PCA_G(self,gradients: torch.Tensor, n_n, dtype=torch.float32):

        X = gradients.to(device=self.device, dtype=dtype)
        print(X.shape)
        X_centered = X - torch.mean(X, dim=0, keepdim=True)

        K = torch.mm(X_centered, X_centered.T) / (X_centered.size(0) - 1)
        K = torch.abs(K)
        print(K.shape)
        eigen_vals, eigen_vecs = torch.linalg.eigh(K)


        sorted_indices = torch.argsort(eigen_vals, descending=True)
        eigen_vals = eigen_vals[sorted_indices]
        eigen_vecs = eigen_vecs[:, sorted_indices]

        n_components = None
        if(type(n_n)==type(1)):
            n_components = n_n
        elif(type(n_n)==type(1.1)):
            n_components = 0
            val_sum = torch.sum(eigen_vals)
            now_sum = 0
            while(1):
                if(now_sum/val_sum>=n_n):
                    break
                else:
                    now_sum = now_sum + float(eigen_vals[n_components])
                    n_components = n_components + 1

        print("dim:",n_components)
        top_vecs = eigen_vecs[:, :n_components]  # (100, 50)
        top_vals = eigen_vals[:n_components]  # (50,)

        with torch.no_grad():
            components = []
            for i in range(n_components):
                v = top_vecs[:, i]  # (100,)
                u = torch.mv(X_centered.T, v)  # (50000,)

                u /= torch.norm(u) + 1e-8

                components.append(u)
            projection_basis = torch.stack(components, dim=0)

        return projection_basis

    def on_round_begin(self):

        if(not self.backdoor_fit and len(self.OA_list)>=self.A_size and self.curr_rnd==self.attack_begin):
            if(self.args.sd_P):
                self.shadow_model = deepcopy(self.P_model)
            self.Backdoor_trojan_fit_penalty(self.data.x, self.train_edge_index, self.edge_weights, self.data.y, self.idx_train, self.idx_attach,self.idx_val,self.core_space)
            self.poison_x, self.poison_edge_index, self.poison_edge_weights, self.poison_labels = self.get_poisoned()
            print("backdoor fit")
            self.backdoor_fit = True

        if(self.backdoor_fit and self.curr_rnd>=self.attack_begin and self.curr_rnd<=self.attack_end):

            self.fit(self.poison_x, self.poison_edge_index, self.poison_edge_weights, self.poison_labels,
                     self.bkd_tn_nodes, self.idx_attach, self.device, self.optimizer_P, self.P_model,
                     train_iters=self.args.MA_eps, verbose=False, is_gradient_collect=True)  # 训练P_model

        else:
            print("no backdoor fit")
            self.fit(self.data.x, self.data.edge_index, self.data.edge_attr,self.data.y,
                             self.bkd_tn_nodes,None, self.device,self.optimizer_P,self.P_model,
                             train_iters=self.args.MA_eps,verbose=False,is_gradient_collect=True) #训练noB_model

    def transfer_to_server(self):

        after_vector = transfer_stateDict_to_vector(deepcopy(self.P_model.state_dict()))
        begin_model_vector = transfer_stateDict_to_vector(self.weight)
        now_gradient = begin_model_vector - after_vector

        self.sd[self.client_id] = {
            'model': get_state_dict(self.P_model),
            'train_size': len(self.loader.partition),
            'gradient_vector':now_gradient.cpu(),
            'B_gradient_vector':self.B_gradient.cpu()
        }

    def project_onto_rowspace(self,A, x, method="pinv"):
        A = A.to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if method == "pinv":
            A_pinv = torch.linalg.pinv(A.T)
            proj_x = (x @ A.T) @ A_pinv.T

        elif method == "qr":
            Q, _ = torch.linalg.qr(A.T, mode="reduced")
            proj_x = (x @ Q) @ Q.T

        elif method == "svd":
            U, S, Vh = torch.linalg.svd(A.T, full_matrices=False)
            r = (S > 1e-10).sum().item()
            U_r = U[:, :r]
            proj_x = (x @ U_r) @ U_r.T

        else:
            raise ValueError("method must be 'pinv', 'qr' or 'svd'")

        return proj_x

    def fit(self, features, edge_index, edge_weight, labels, idx_train,idx_attach, device,optimizer,model,idx_val=None, train_iters=200, verbose=False,is_gradient_collect = False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        verbose : bool
            whether to show verbose logs
        """
        tmp_edge_index = edge_index.clone()
        if(edge_weight!=None):
            tmp_edge_weight = edge_weight.clone()
        else:
            tmp_edge_weight = None
        tmp_features = features.clone().to(device)
        tmp_labels = labels.clone().to(device)

        model.train()
        #print(idx_train)
        #print(idx_attach)
        for i in range(train_iters):
            if (idx_attach != None):

                #clean step
                optimizer.zero_grad()
                output = model.forward(tmp_features, tmp_edge_index, tmp_edge_weight)
                idx_clean = list(set(idx_train.cpu().numpy()) - set(idx_attach.cpu().numpy()))
                idx_clean = torch.tensor(idx_clean).to(device)

                #loss_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
                loss_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
                loss_clean = loss_clean * len(idx_clean)/len(idx_train)

                loss_clean.backward(retain_graph=True)

                if(is_gradient_collect):
                    clean_gradient = {}
                    for name, params in model.named_parameters():
                        grad = params.grad
                        if grad is not None:
                            # print(name)
                            clean_gradient[name] = grad
                    self.clean_gradient = clean_gradient

                self.optimizer_P.zero_grad()
                loss_attach = F.nll_loss(output[idx_attach], labels[idx_attach])
                loss_attach = loss_attach * len(idx_attach)/len(idx_train)
                loss_attach.backward()

                if(is_gradient_collect):
                    attach_gradient = {}
                    for name, params in model.named_parameters():
                        grad = params.grad
                        if grad is not None:
                            attach_gradient[name] = grad

                    self.attach_gradient = attach_gradient

                self.B_gradient = transfer_stateDict_to_vector(self.attach_gradient)
                for name, params in model.named_parameters():
                    if params.grad is not None:
                        params.grad = params.grad + self.clean_gradient[name]

                optimizer.step()

            else:
                print("no attach")
                optimizer.zero_grad()
                output = model.forward(tmp_features, tmp_edge_index, tmp_edge_weight)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train.backward()

                clean_gradient = {}
                attach_gradient = {}
                for name, params in model.named_parameters():
                    grad = params.grad
                    if grad is not None:
                        clean_gradient[name] = grad
                        attach_gradient[name] = grad

                self.clean_gradient = clean_gradient
                self.attach_gradient = attach_gradient
                self.B_gradient = transfer_stateDict_to_vector(self.attach_gradient)
                optimizer.step()

        model.eval()
        output = model.forward(tmp_features, tmp_edge_index, tmp_edge_weight)


    def get_trigger_index(self, trigger_size):#构建全连接的trigger子图的边列表，实际上就是待选trigger边列表
        edge_list = []
        edge_list.append([0, 0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            #print(edges)
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])


        return edge_index

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight, device):#完成了trigger的生成和拼接
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],
                                                  self.args.thrd)  # may revise the process of generate


        trojan_weights = torch.cat([torch.ones([len(idx_attach), 1], dtype=torch.float, device=device), trojan_weights],
                                   dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1, features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        return update_feat, update_edge_index, update_edge_weights

    def Backdoor_trojan_fit_penalty(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled,A):

        args = self.args

        self.optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        self.trojan = TrojanAwareNet(self.args, features.shape[1], self.args.n_dims, args.trigger_size, self.device).to(
                self.device)
        self.optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.homo_loss = HomoLoss(self.args, self.device)
        self.or_loss = OrthogonalLoss(self.args, self.device)
        self.trigger_loss = TriggerLoss(self.args, self.device)
        self.Norm_loss = DelatNormLoss(self.args, self.device)


        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        trojan_edge = self.get_trojan_edge(len(features), idx_attach, args.trigger_size).to(self.device)


        loss_best = 1e8
        loss_inner = None

        idx_outter = idx_attach
        labels_outter = labels.clone()
        labels_outter[idx_outter] = args.target_class

        for i in range(args.trojan_epochs):
            idx_attach = idx_attach.to(self.device)
            features = features.to(self.device)
            edge_index = edge_index.to(self.device)
            self.data = self.data.to(self.device)
            self.trojan = self.trojan.to(self.device)

            x = None
            for j in range(self.args.inner):
                self.shadow_model.train()
                self.optimizer_shadow.zero_grad()
                idx_clean = idx_train
                output = self.shadow_model(features, edge_index, edge_weight)
                loss_clean = F.nll_loss(output[torch.cat([idx_clean, idx_outter])],
                                                                    labels[torch.cat([idx_clean, idx_outter])])
                loss_clean.backward()
                clean_abso = {}
                for name, params in self.shadow_model.named_parameters():
                    grad = params.grad
                    if grad is not None:
                        clean_abso[name] = grad
                clean_abso_vector = transfer_stateDict_to_vector(clean_abso)
                clean_abso_vector = clean_abso_vector.flatten().to(self.device)

                self.optimizer_shadow.zero_grad()

                poison_x, poison_edge_index, poison_edge_weights, __1__ = self.subG_poisoned(features, edge_index, edge_weight,self.idx_attach)

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                loss_attach = F.nll_loss(output[idx_attach], self.labels[idx_attach])
                loss_attach = loss_attach * int(idx_attach.shape[0]) / ( int(idx_attach.shape[0]) + int(idx_clean.shape[0]) )
                loss_attach.backward(retain_graph=True)

                shadow_gradient = {}
                for name, params in self.shadow_model.named_parameters():
                    grad = params.grad
                    if grad is not None:
                        shadow_gradient[name] = grad

                self.attach_gradient = shadow_gradient
                self.B_gradient = transfer_stateDict_to_vector(self.attach_gradient)
                x = transfer_stateDict_to_vector(shadow_gradient)
                x = x.flatten().to(self.device)
                self.optimizer_shadow.zero_grad()

                loss_clean = F.nll_loss(output[idx_clean], self.labels[idx_clean])
                loss_clean = loss_clean * int(idx_clean.shape[0]) / ( int(idx_attach.shape[0]) + int(idx_clean.shape[0]) )
                loss_clean.backward(retain_graph=True)

                clean_now = {}
                for name, params in self.shadow_model.named_parameters():
                    grad = params.grad
                    if grad is not None:
                        clean_now[name] = grad
                        params.grad = params.grad + shadow_gradient[name]

                clean_vector = transfer_stateDict_to_vector(clean_now)
                clean_vector = clean_vector.flatten().to(self.device)
                self.optimizer_shadow.step()
                self.optimizer_shadow.zero_grad()

            self.optimizer_trigger.zero_grad()
            self.trojan.train()

            trojan_edge = self.get_trojan_edge(features.shape[0], idx_outter, self.args.trigger_size).to(self.device)
            update_feat, update_edge_index, update_edge_weights, trojan_weights = self.subG_poisoned(features, edge_index, edge_weight,idx_outter)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            loss_target = self.args.target_loss_weight * F.nll_loss(output[torch.cat([idx_clean, idx_outter])],
                                                                    labels_outter[torch.cat([idx_clean, idx_outter])])

            loss_trigger = 0.0
            if (self.args.trigger_loss_weight > 0):
                clean_E = []
                poisoned_E = []
                B_list = []
                for i in range(idx_attach.shape[0]):
                    idx = torch.tensor([idx_attach[i]])
                    idx = idx.to(self.device)

                    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(node_idx=idx,
                                                                                                           num_hops=2,
                                                                                                           edge_index=update_edge_index,
                                                                                                           relabel_nodes=True)  # sub_mapping means the index of [idx] in sub-node-set

                    ori_node_idx = sub_induct_nodeset[sub_mapping]
                    relabeled_node_idx = sub_mapping
                    sub_induct_edge_weights = update_edge_weights[sub_edge_mask]
                    sub_induct_nodeset = sub_induct_nodeset.to(self.device)
                    poisoned_e = self.shadow_model.get_embedding(update_feat[sub_induct_nodeset], sub_induct_edge_index,
                                                                 edge_weight=sub_induct_edge_weights,
                                                                 idx=relabeled_node_idx)
                    poisoned_E.append(poisoned_e)

                    sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(node_idx=idx,
                                                                                                           num_hops=2,
                                                                                                           edge_index=edge_index,
                                                                                                           relabel_nodes=True)  # sub_mapping means the index of [idx] in sub-node-set
                    ori_node_idx = sub_induct_nodeset[sub_mapping]
                    relabeled_node_idx = sub_mapping
                    sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(self.device)
                    sub_induct_nodeset = sub_induct_nodeset.to(self.device)

                    B = self.shadow_model.get_embedding(features[sub_induct_nodeset], sub_induct_edge_index,
                                                        edge_weight=sub_induct_edge_weights)
                    clean_e = B[relabeled_node_idx]
                    B_list.append(B)
                    clean_E.append(clean_e)
                self.trigger_loss = TriggerLoss(self.args, self.device)
                loss_trigger = self.trigger_loss(clean_E, poisoned_E, B_list)

            loss_homo = 0.0
            if (self.args.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge[:, :int(trojan_edge.shape[1] / 2)],
                                           trojan_weights,
                                           update_feat,
                                           self.args.homo_boost_thrd)

            loss_sim = 0.0
            if (self.args.sim_loss_weight > 0):
                self.sim_loss = SimLoss(self.args, self.device)
                loss_sim = self.sim_loss(clean_vector, clean_abso_vector)

            loss_or = 0.0
            if (self.args.orthogonal_loss_weight > 0):
                self.or_loss = OrthogonalLoss(self.args, self.device)
                loss_or = self.or_loss(A, x)


            loss_outter = (loss_target + self.args.homo_loss_weight * loss_homo + self.args.trigger_loss_weight * loss_trigger
                           + loss_or * self.args.orthogonal_loss_weight + self.args.sim_loss_weight * loss_sim)

            loss_outter.backward()
            self.optimizer_trigger.step()

            acc_train_outter = (output[idx_outter].argmax(dim=1) == args.target_class).float().mean()

            if loss_outter < loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)


        self.trojan.load_state_dict(self.weights)
        #self.Backdoor_test(update_feat, update_edge_index, update_edge_weights, self.shadow_model)
        if args.debug:
            print("load best weight based on the loss outter")

        self.trojan.eval()

    def subG_poisoned(self,features,edge_index,edge_weights,idx_attach):

        result_x = features
        result_weights = edge_weights
        trojan_edge = self.get_trojan_edge(features.shape[0], idx_attach, self.args.trigger_size).to(self.device)

        result_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(node_idx=idx_attach,
                                                                                               num_hops=2,
                                                                                               edge_index=edge_index,
                                                                                               relabel_nodes=True)  # sub_mapping means the index of [idx] in sub-node-set
        ori_node_idx = sub_induct_nodeset[sub_mapping]
        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(self.device)
        sub_induct_nodeset = sub_induct_nodeset.to(self.device)
        trojan_feat, trojan_weights = self.trojan(relabeled_node_idx,features[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights)# may revise the process of generate
        trojan_weights = torch.cat(
            [torch.ones([idx_attach.shape[0], 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
        trojan_weights = trojan_weights.flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        result_weights = torch.cat([result_weights, trojan_weights,
                                         trojan_weights])  # repeat trojan weights beacuse of undirected edge
        result_x = torch.cat([result_x, trojan_feat])

        return result_x,result_edge_index,result_weights,trojan_weights

    def get_poisoned(self):

        with torch.no_grad():
            tmp_x, tmp_edge_index, tmp_edge_weights = self.features.clone(), self.edge_index.clone(), self.edge_weights.clone()
            poison_x, poison_edge_index, poison_edge_weights,__1__ = self.subG_poisoned(tmp_x, tmp_edge_index, tmp_edge_weights,self.idx_attach)

        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]

        return poison_x, poison_edge_index, poison_edge_weights, poison_labels


    def Backdoor_test(self,poison_x,poison_edge_index,poison_edge_weights,model):
        model.eval()
        output = model(poison_x, poison_edge_index, poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[self.idx_attach]==self.args.target_class).float().mean()
        msg1 = " target class rate on Vs: {:.4f} ".format(train_attach_rate)

        clean_acc = model.test(self.features,self.edge_index,self.edge_weights,self.data.y,self.idx_clean_test)
        self.acc_list.append(clean_acc)
        msg2 = " accuracy on clean test nodes: {:.4f} ".format(clean_acc)


        asr = 0
        flip_asr = 0
        flip_idx_atk = self.idx_atk[(self.data.y[self.idx_atk] != self.args.target_class).nonzero().flatten()]#将原标签为非target_label的部分抽出来
        for i, idx in enumerate(self.idx_atk):
            idx = idx.reshape(1)
            tmp_x, tmp_edge_index, tmp_edge_weights = self.features.clone(), self.edge_index.clone(), self.edge_weights.clone()
            with torch.no_grad():
                induct_x, induct_edge_index, induct_edge_weights, __1__ = self.subG_poisoned(tmp_x,
                                                                                             tmp_edge_index,
                                                                                             tmp_edge_weights,
                                                                                             idx)

                induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

                output = model(induct_x, induct_edge_index, induct_edge_weights)
                if(output.argmax(dim=1)[idx] == self.args.target_class):

                    asr = asr + 1

                    if (self.data.y[idx] != self.args.target_class):
                        flip_asr = flip_asr + 1

        asr = asr / (self.idx_atk.shape[0])

        flip_asr = flip_asr / (flip_idx_atk.shape[0])

        self.asr_list.append(flip_asr)
        self.current_eval_asr = float(flip_asr)
        self.global_asr_proxy_list.append(self.current_eval_asr)
        msg3 = " Overall ASR: {:.4f} ".format(asr)
        msg4 = " Flip ASR: {:.4f}/{} nodes ".format(flip_asr, flip_idx_atk.shape[0])
        self.logger.print(msg1 + msg2 + msg3 + msg4)
