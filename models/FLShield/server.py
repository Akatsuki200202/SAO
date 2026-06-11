import time
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from function_utils import transfer_stateDict_to_vector_np
from modules.federated import ServerModule
from models.FLShield.Cluster import auto_kmeans_by_silhouette
import numpy as np
import torch
from models.nets import GCN3
from misc.utils import get_state_dict,torch_save,set_state_dict
import random
import os

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
        self.score_record_dict = {}
        for i in range(self.args.n_clients):
            self.score_record_dict[i] = []
        self.window_size = float(getattr(args, "window_size", 10))
        self.device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(int(self.gpu_id)))
        self.cos_thrd = self.args.add_dim_threshold
        self.broadcasted_model = None
        self.collect_model = None

    def on_round_begin(self, curr_rnd):

        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()
        self.broadcasted_model = get_state_dict(self.model)

    def on_round_complete(self, updated):
        st = time.time()
        updated = np.array(list(updated))
        updated = np.sort(np.array(updated))

        self.logger.print(f'all clients have been uploaded ({time.time() - st:.2f}s)')

        self.chosen_client = np.array(range(len(updated)))

        self.detect(updated)
        self.update(updated)
        self.save_state()
        for c_id in updated:
            del self.sd[c_id]

    def update(self, updated):
        st = time.time()
        print("chosen:",self.chosen_client)
        self.chosen_weight = []
        for i in self.chosen_client:
            self.chosen_weight.append(self.local_weights[i])

        self.model_dis_list = np.zeros(self.num_client)
        begin_v = transfer_stateDict_to_vector_np(self.broadcasted_model).flatten()
        for i in range(len(self.chosen_client)):
            minas = begin_v - self.local_model_vector[i]
            self.model_dis_list[i] = np.linalg.norm( minas )

        S_median = np.median(self.model_dis_list)
        for i in range(len(self.chosen_weight)):
            gama = S_median / self.model_dis_list[i]
            gama = min(1, gama)
            print(gama)
            for name in self.broadcasted_model.keys():
                self.chosen_weight[i][name] = self.broadcasted_model[name] + gama * (self.chosen_weight[i][name] - self.broadcasted_model[name])

        ratio = np.ones(len(self.chosen_client))
        ratio = ratio / np.sum(ratio)
        ratio = ratio.tolist()
        print(ratio)
        self.set_weights(self.model, self.aggregate(self.chosen_weight, ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

    def represent_model(self,updated):
        self.local_weights = []
        self.local_model_vector = []
        for c_id in updated:
            self.local_weights.append(self.sd[c_id]['model'].copy())
            self.local_model_vector.append(transfer_stateDict_to_vector_np(self.local_weights[c_id]).flatten())

        self.clusters, best_labels = self.gradient_cos(updated) #收集了新的gradient
        self.rep_model_list = []
        name_list = []

        for name in self.clusters.keys():
            name_list.append(name)
        name_list = np.array(name_list)
        name_list = np.sort(name_list)

        for name in name_list:
            now_client = self.clusters[name]
            print(name,":",now_client,end = ",")
            now_model = []
            for c_id in updated:
                if(c_id in now_client):
                    now_model.append(self.local_weights[c_id])

            ratio = np.ones(len(now_client))
            ratio = ratio / np.sum(ratio)
            ratio = ratio.tolist()
            print(ratio)
            self.set_weights(self.model, self.aggregate(now_model, ratio))
            self.rep_model_list.append(get_state_dict(self.model))

        self.sd["rep_model_list"] = self.rep_model_list

    def detect(self,updated):
        N_list = []
        M_list = []
        updated = np.array(list(updated))
        #print(updated)
        updated = np.sort(np.array(updated))
        for i in updated:
            msg = "loss_"+str(i)
            X = self.sd[msg]
            X = np.array(X)
            X[X == -1] = np.nan
            X = X.astype(float)
            N_list.append(X)

        for i in range(len(self.rep_model_list)):
            now_M = []
            for j in updated:
                now_M.append(N_list[j][i])

            X = np.array(now_M)
            imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
            X_filled = imputer.fit_transform(X)

            M_list.append(X_filled)

        M_list = np.array(M_list)
        #print(M_list)
        score_list = []
        for i in range(len(M_list)):
            now_avg_M = np.average(M_list[i], axis=0)
            now_score = now_avg_M.min()
            score_list.append(now_score)
        score_list = np.array(score_list)
        print(score_list)
        id_sort = np.argsort(score_list)[::-1]
        print(id_sort)
        self.chosen_client = []
        for i in range(len(id_sort)):
            for c_id in self.clusters[id_sort[i]]:
                self.chosen_client.append(c_id)
            if(len(self.chosen_client)>=int(self.num_client/2)):
                break

        for c_id in updated:
            msg = "loss_" + str(c_id)
            del self.sd[msg]

    def gradient_cos(self,updated):

        self.num_client = len(updated)
        X = torch.zeros(len(self.local_model_vector), len(self.local_model_vector[0]))
        for i in range(len(self.local_model_vector)):
            X[i] = torch.tensor(self.local_model_vector[i])

        clusters, best_centers, best_labels = auto_kmeans_by_silhouette(X,k_min=2,k_max=5,lower_thresh=0.35,upper_thresh=0.65)
        #print(clusters)

        gradient_cos_matrix = np.zeros((self.num_client,self.num_client))
        for i in range(self.num_client):
            for j in range(self.num_client):
                mi = torch.tensor(self.local_model_vector[i])
                mj = torch.tensor(self.local_model_vector[j])
                gradient_cos_matrix[i][j] =round(
                    float(mi.flatten().dot(mj.flatten()) /
                        (torch.linalg.norm(mi.flatten()) * torch.linalg.norm(mj.flatten()))
                          ), 4
                )
        #print(gradient_cos_matrix)
        self.gradient_cos_list.append(gradient_cos_matrix)

        return clusters, best_labels

    def inner_data_save(self):
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





