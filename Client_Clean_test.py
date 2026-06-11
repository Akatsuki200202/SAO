from models.nets import GCN3
import torch
from function_utils import get_id, multi_print
from torch_geometric.utils import to_undirected
from data.loader import DataLoader
from modules.logger import Logger
import torch.optim as optim
from misc.utils import torch_load,set_state_dict

class Clean_test:
    def __init__(self,args,g_id,client_id):
        self.args = args
        self.g_id = g_id
        self.client_id = client_id
        self.loader = DataLoader(self.args)
        self.loader.switch(client_id)
        self.logger = Logger(self.args, self.g_id)
        self.logger.switch(client_id)
        self.device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(int(g_id)))
        print(self.device)
        self.data = list(enumerate(self.loader.pa_loader))[0][1]
        self.data = self.data.to(self.device)

        self.idx_train, self.idx_val, self.idx_clean_test, self.idx_atk = get_id(self.data,self.device)
        self.features = self.data.x


        self.data.edge_index = to_undirected(self.data.edge_index)

        self.edge_index = self.data.edge_index
        self.edge_weights = torch.ones([self.edge_index.shape[1]], device=self.device, dtype=torch.float)
        self.model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.base_lr, weight_decay=self.args.weight_decay)

        self.load_state()

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')

        self.optimizer.load_state_dict(loaded['optimizer'])
        set_state_dict(self.model, loaded['model'], self.g_id)

    def do_test(self,acc_list):
        clean_acc = self.model.test(self.data.x,self.data.edge_index,self.data.edge_attr,self.data.y,self.idx_clean_test)
        msgg = self.logger.print("accuracy on clean test nodes: {:.4f})".format(clean_acc))
        multi_print(self.args.checkpt_path + self.args.output_file,
                    msgg)
        clean_acc = float(clean_acc) * 100
        clean_acc = '%.2f'%clean_acc
        clean_acc = float(clean_acc)
        acc_list.append(clean_acc)

