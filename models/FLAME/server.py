from function_utils import transfer_stateDict_to_vector_np
import hdbscan
from collections import Counter
import time
import numpy as np
from modules.federated import ServerModule
from models.nets import GCN3
from misc.utils import get_state_dict,torch_save,set_state_dict
import random
import os
import torch

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        super(Server, self).__init__(args, sd, gpu_server)
        self.model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)
        self.gradient_cos_list = []
        self.chosen_list = []

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()
        self.broadcasted_model = get_state_dict(self.model)

    def on_round_complete(self, updated):
        st = time.time()
        updated = np.array(list(updated))
        print(updated)
        updated = np.sort(np.array(updated))
        self.local_weights = []
        self.local_train_sizes = []
        self.local_model_vector = []
        for c_id in updated:
            self.local_weights.append(self.sd[c_id]['model'].copy())
            self.local_train_sizes.append(self.sd[c_id]['train_size'])
            self.local_model_vector.append(transfer_stateDict_to_vector_np(self.local_weights[c_id]))

        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')
        self.cos_dis_compute(updated)
        self.update(updated)
        self.save_state()

    def update(self, updated):

        st = time.time()
        self.set_weights(self.model, self.Flame_aggregate())
        for c_id in updated:
            del self.sd[c_id]
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

    def Flame_aggregate(self):

        hdb = hdbscan.HDBSCAN(min_cluster_size = int(self.num_client/2+1), min_samples=1, metric='precomputed')
        hdb.fit(self.model_cos_matrix)
        cluster_label = hdb.labels_
        if(len(np.unique(cluster_label))>1):
            label_ct = Counter(cluster_label)
            ans = max(label_ct, key=lambda x: label_ct[x])
            chosen_client = np.array(range(self.num_client),dtype=int)[cluster_label==ans]

        else:
            chosen_client = np.array(range(self.num_client),dtype=int)

        self.chosen_list.append(chosen_client)

        ratio = np.ones(len(chosen_client))
        ratio = ratio / np.sum(ratio)
        ratio = ratio.tolist()
        chosen_weights = []
        S_median = np.median(self.model_dis_list)

        for c_id in chosen_client:
            gama = S_median / self.model_dis_list[c_id]
            gama = min(1, gama)
            for name in self.broadcasted_model.keys():
                #print(gama)
                self.local_weights[c_id][name] = self.broadcasted_model[name] + gama * (self.local_weights[c_id][name] - self.broadcasted_model[name])

            chosen_weights.append(self.local_weights[c_id])

        global_model = self.aggregate(chosen_weights, ratio)
        lamda = self.args.flame_lamda
        gau_std = lamda * S_median

        for name in global_model.keys():
            gaussian_noise = np.random.normal(0, gau_std,global_model[name].shape)
            global_model[name] = global_model[name] + gaussian_noise

        return global_model

    def cos_dis_compute(self,updated):

        self.num_client = len(updated)
        self.model_cos_matrix = np.zeros((self.num_client,self.num_client))
        self.model_dis_list = np.zeros(self.num_client)

        for i in range(self.num_client):
            for j in range(i,self.num_client):
                if(i == j):
                    self.model_cos_matrix[i][j]=0.0
                else:
                    wi = self.local_weights[i]
                    wj = self.local_weights[j]
                    c_now= 0
                    for k in wi.keys():
                        ww_i = wi[k].flatten()
                        ww_j = wj[k].flatten()
                        c_now = c_now + round(float(ww_i.dot(ww_j)))

                    self.model_cos_matrix[i][j] = self.model_cos_matrix[j][i] = 1 - (c_now /(np.linalg.norm(self.local_model_vector[i]) * np.linalg.norm(self.local_model_vector[j])))

        begin_v = transfer_stateDict_to_vector_np(self.broadcasted_model)
        for i in range(self.num_client):
            self.model_dis_list[i] = np.linalg.norm(begin_v - self.local_model_vector[i])

        self.gradient_cos_list.append(self.model_cos_matrix)

    def inner_data_save(self):
        self.chosen_list = np.array(self.chosen_list)
        np.save(self.args.checkpt_path + "/chosen_list.npy",self.chosen_list)
        if(self.args.attack_method!="none"):
            self.gradient_cos_list = np.array(self.gradient_cos_list)
            np.save(self.args.checkpt_path + "/gradient_cos_list.npy",self.gradient_cos_list)

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model)
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
        })





