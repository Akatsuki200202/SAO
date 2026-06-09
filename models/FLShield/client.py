from modules.federated import ClientModule
from function_utils import get_id, accuracy, transfer_stateDict_to_vector
from copy import deepcopy
import random
import os
from models.nets import GCN3
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from misc.utils import get_state_dict, torch_load, torch_save, set_state_dict
from collections import OrderedDict


class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd, client_id):
        super(Client, self).__init__(args, w_id, g_id, sd, client_id)

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
        self.args = args

        self.model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.device)
        self.parameters = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)

        self.data = list(enumerate(self.loader.pa_loader))[0][1]
        self.data = self.data.to(self.device)
        self.idx_train, self.idx_val, self.idx_clean_test, self.idx_atk = get_id(self.data, self.device)
        self.acc_list = []
        self.shared_asr_list = []
        self.current_eval_asr = 0.0
        self.now_gradient = {}
        self.record_begin = args.record_begin
        self.now_global = None
        self.old_global = None
        self.OA_list = []
        self.A_size = self.args.A_size
        self.RG_size = self.args.RG_size
        self.add_dim_threshold = self.args.add_dim_threshold

        self.train_edge_index = self.data.edge_index
        self.train_edge_weights = torch.ones([self.train_edge_index.shape[1]], device=self.device, dtype=torch.float)

        self.log = {
            'lr': [], 'train_lss': [],
            'ep_local_val_lss': [], 'ep_local_val_acc': [],
            'rnd_local_val_lss': [], 'rnd_local_val_acc': [],
            'ep_local_test_lss': [], 'ep_local_test_acc': [],
            'rnd_local_test_lss': [], 'rnd_local_test_acc': [],
        }
        self.pre_idx = self.idx_val
        self.idx_dict = {}
        for i in range(self.args.n_clss):
            self.idx_dict[i] = []
        for i in range(len(self.pre_idx)):
            self.idx_dict[int(self.data.y[self.pre_idx[i]])].append(int(self.pre_idx[i]))
        print(self.idx_dict)

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })

    def inner_data_save(self):
        self.acc_list = np.array(self.acc_list)
        np.save(self.args.checkpt_path + "/{}_acc_list.npy".format(self.client_id), self.acc_list)
        self.shared_asr_list = np.array(self.shared_asr_list)
        np.save(self.args.checkpt_path + "/{}_shared_asr_list.npy".format(self.client_id), self.shared_asr_list)

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd['global'])

        self.begin_model_sd = deepcopy(self.model.state_dict())

    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True)

    def on_round_begin(self):
        self.fit(self.data.x, self.data.edge_index, self.data.edge_attr, self.data.y, self.idx_train, None,
                 self.device, train_iters=self.args.n_eps, verbose=False, is_gradient_collect=True)

    def transfer_to_server(self):

        after_vector = transfer_stateDict_to_vector(deepcopy(self.model.state_dict()))
        begin_model_vector = transfer_stateDict_to_vector(self.begin_model_sd)
        self.now_gradient = begin_model_vector - after_vector
        now_gradient_vector = self.now_gradient


        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': len(self.loader.partition),
            'gradient_vector': now_gradient_vector.cpu()
        }

    def do_test(self):
        clean_acc = self.model.test(self.data.x, self.data.edge_index, self.data.edge_attr, self.data.y,
                                    self.idx_clean_test)
        msgg = self.logger.print("accuracy on clean test nodes: {:.4f})".format(clean_acc))

        self.transfer_to_server()

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, device, idx_val=None,
            train_iters=200, verbose=False, is_gradient_collect=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """
        tmp_edge_index = edge_index.clone()
        if (edge_weight != None):
            tmp_edge_weight = edge_weight.clone()
        else:
            tmp_edge_weight = None
        tmp_features = features.clone().to(device)
        tmp_labels = labels.clone().to(device)

        for i in range(train_iters):
            # print(i)
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model.forward(self.data.x, self.data.edge_index, self.data.edge_attr)
            # loss_train = F.nll_loss(output[idx_train], self.data.y[idx_train])
            loss_train = F.nll_loss(output[idx_train], self.data.y[idx_train])
            loss_train.backward()
            now_gradient = OrderedDict()
            for name, params in self.model.named_parameters():
                if params.grad is not None:
                    now_gradient[name] = deepcopy(params.grad)

            self.now_gradient = now_gradient
            self.optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.model.eval()

    def valid_compute(self):

        rep_model_list = self.sd["rep_model_list"]
        rep_model = deepcopy(self.model)
        loss_vector_list = np.zeros((len(rep_model_list), self.args.n_clss))
        for i in range(len(rep_model_list)):
            set_state_dict(rep_model, rep_model_list[i], self.device)
            loss_vector = np.zeros(self.args.n_clss)
            output = rep_model.forward(self.data.x, self.data.edge_index, self.data.edge_attr)
            for lab in range(self.args.n_clss):
                loss_val = -1
                lab_idx = self.idx_dict[lab]
                if (len(lab_idx) > 0):
                    lab_idx = torch.tensor(np.array(lab_idx)).to(self.device)
                    loss_val = F.nll_loss(output[lab_idx], self.data.y[lab_idx])
                loss_vector[lab] = float(loss_val)

            loss_vector_list[i] = loss_vector

        msg = "loss_" + str(self.client_id)
        self.sd[msg] = loss_vector_list
        # print("client",self.client_id,":",loss_vector_list)
        return

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(features, edge_index, edge_weight)
            acc_test = accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)
