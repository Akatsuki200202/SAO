from models.backdoor import TrojanAwareNet
from torch_geometric.utils import k_hop_subgraph
from models.nets import GCN3
import torch
from function_utils import get_id, multi_print
from torch_geometric.utils import to_undirected
from data.loader import DataLoader
from modules.logger import Logger
import torch.optim as optim
from misc.utils import torch_load,set_state_dict

class Cross_test:
    def __init__(self, args, g_id, client_id):
        self.args = args
        self.g_id = g_id
        self.client_id = client_id
        self.loader = DataLoader(self.args)
        self.loader.switch(client_id)
        self.logger = Logger(self.args, self.g_id)
        self.logger.switch(client_id)
        self.device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(int(g_id)))
        self.data = list(enumerate(self.loader.pa_loader))[0][1]
        self.data = self.data.to(self.device)
        self.attacker_id_list = self.args.Attacker_id.split("+")
        for i in range(len(self.attacker_id_list)):
            self.attacker_id_list[i] = int(self.attacker_id_list[i])

        self.idx_train, self.idx_val, self.idx_clean_test, self.idx_atk = get_id(self.data, self.device)
        self.features = self.data.x
        self.trigger_index = self.get_trigger_index(args.trigger_size)


        self.data.edge_index = to_undirected(self.data.edge_index)

        self.edge_index = self.data.edge_index
        self.edge_weights = torch.ones([self.edge_index.shape[1]], device=self.device, dtype=torch.float)
        self.model = GCN3(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.device)
        self.parameters = list(self.model.parameters())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        if (args.attack_method=="SAO"):
            self.trojan = TrojanAwareNet(self.args, self.features.shape[1], self.args.n_dims, args.trigger_size,
                                         self.device).to(self.device)

        self.load_state()

    def load_state(self):
        if (self.client_id in self.attacker_id_list):
            loaded = torch_load(self.args.checkpt_path, '{}_Bstate.pt'.format(self.client_id))
        else:
            loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')

        self.optimizer.load_state_dict(loaded['optimizer'])
        set_state_dict(self.model, loaded['model'], self.g_id)

    def subG_poisoned(self, features, edge_index, edge_weights, idx_attach, feat_list=None, weights_list=None,
                      record=False):

        result_x = features
        result_weights = edge_weights
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.args.trigger_size).to(self.device)
        result_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(node_idx=idx_attach,
                                                                                               num_hops=2,
                                                                                               edge_index=edge_index,
                                                                                               relabel_nodes=True)  # sub_mapping means the index of [idx] in sub-node-set
        ori_node_idx = sub_induct_nodeset[sub_mapping]
        relabeled_node_idx = sub_mapping
        sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(self.device)
        sub_induct_nodeset = sub_induct_nodeset.to(self.device)

        trojan_feat, trojan_weights = self.trojan(relabeled_node_idx, features[sub_induct_nodeset],
                                                  sub_induct_edge_index,
                                                  sub_induct_edge_weights)  # may revise the process of generate
        if (record):
            feat_list.append(float(torch.linalg.norm(trojan_feat)))
            weights_list.append(float(torch.linalg.norm(trojan_weights)))
        trojan_weights = torch.cat(
            [torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)

        trojan_weights = trojan_weights.flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        result_weights = torch.cat([result_weights, trojan_weights,
                                    trojan_weights]).detach()  # repeat trojan weights beacuse of undirected edge
        result_x = torch.cat([result_x, trojan_feat]).detach()

        return result_x, result_edge_index, result_weights, trojan_weights

    def do_test(self, acc_list, asr_list):
        self.Backdoor_test_new(self.features, self.edge_index, self.edge_weights, self.model, acc_list, asr_list)

    def Backdoor_test_new(self, poison_x, poison_edge_index, poison_edge_weights, model, acc_list, asr_list):
        model.eval()
        self.trojan.eval()
        clean_acc = model.test(self.features, self.edge_index, self.edge_weights, self.data.y, self.idx_clean_test)
        # print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
        msg2 = " accuracy on clean test nodes: {:.4f} ".format(clean_acc)
        clean_acc = float(clean_acc) * 100
        clean_acc = '%.2f' % clean_acc
        clean_acc = float(clean_acc)
        acc_list.append(clean_acc)
        # overall_induct_edge_index, overall_induct_edge_weights = poison_edge_index.clone(), poison_edge_weights.clone()
        flip_idx_atk = self.idx_atk[(self.data.y[self.idx_atk] != self.args.target_class).nonzero().flatten()]
        self.idx_atk = torch.sort(self.idx_atk)[0]

        all_asr = 0
        all_flip_asr = 0
        for a_id in self.attacker_id_list:
            # print(self.idx_atk)
            asr = 0
            flip_asr = 0
            loaded_trojan = torch_load(self.args.trojan_path, "backdoor_generator_state{}.pt".format(a_id))
            set_state_dict(self.trojan, loaded_trojan['trojan_model'], self.g_id)

            for i, idx in enumerate(self.idx_atk):
                idx = idx.reshape(1)
                tmp_x, tmp_edge_index, tmp_edge_weights = self.features.clone(), self.edge_index.clone(), self.edge_weights.clone()
                with torch.no_grad():
                    induct_x, induct_edge_index, induct_edge_weights, __1__ = self.subG_poisoned(tmp_x, tmp_edge_index,
                                                                                                 tmp_edge_weights, idx)

                    induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
                    output = model(induct_x, induct_edge_index, induct_edge_weights)
                    if (output.argmax(dim=1)[idx] == self.args.target_class):
                        asr = asr + 1

                        if (self.data.y[idx] != self.args.target_class):
                            flip_asr = flip_asr + 1

            asr = asr / (self.idx_atk.shape[0])
            flip_asr = flip_asr / (flip_idx_atk.shape[0])
            all_asr = all_asr + asr
            all_flip_asr = all_flip_asr + flip_asr
            print(a_id,":",asr,",",flip_asr)

        all_flip_asr = all_flip_asr / len(self.attacker_id_list)
        all_flip_asr = float(all_flip_asr) * 100
        all_flip_asr = '%.2f' % all_flip_asr
        all_flip_asr = float(all_flip_asr)
        asr_list.append(all_flip_asr)
        msg3 = " Overall ASR: {:.4f} ".format(all_asr)
        msg4 = " Flip ASR: {:.4f}/{} nodes ".format(all_flip_asr, flip_idx_atk.shape[0])
        self.logger.print(msg2 + msg3 + msg4)
        multi_print(self.args.checkpt_path + self.args.output_file,
                    msg2 + msg3 + msg4)


    def inject_trigger(self, idx_attach, features, edge_index, edge_weight, device):
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

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights

    def get_trigger_index(self, trigger_size):  # 构建全连接的trigger子图的边列表，实际上就是待选trigger边列表
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
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])

        return edge_index
