from function_utils import transfer_stateDict_to_vector_np
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
        super(Server, self).__init__(args, sd, gpu_server)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        self.model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)
        self.gradient_cos_list = []

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()
        self.begin_weight = get_state_dict(self.model)

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        self.local_weights = []
        for c_id in updated:
            self.local_weights.append(self.sd[c_id]['model'].copy())
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        st = time.time()
        self.RFA_agg()
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

    def RFA_agg(self):
        ini_alpha = np.ones(len(self.local_weights))
        ini_alpha = ini_alpha / np.sum(ini_alpha)
        weight_v_list = []
        for i in range(len(self.local_weights)):
            now_weight = self.local_weights[i]
            now_v = transfer_stateDict_to_vector_np(now_weight)
            weight_v_list.append(now_v)

        v_i = transfer_stateDict_to_vector_np(self.begin_weight)
        tole = 1e-3
        R = 3
        v_i_m = None
        for i in range(R):
            beta = np.zeros(len(self.local_weights))
            for j in range(len(self.local_weights)):
                upper = ini_alpha[j]
                l_2 = np.linalg.norm(v_i - weight_v_list[j])
                downner = np.max(np.array([tole,l_2]))
                beta[j] = upper / downner

            beta = beta / np.sum(beta)
            v_i_m = self.aggregate(self.local_weights,beta.tolist())
            v_i = transfer_stateDict_to_vector_np(v_i_m)

        self.set_weights(self.model,v_i_m)


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






