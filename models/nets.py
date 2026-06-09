import torch.nn as nn
import torch.nn.functional as F
import function_utils
import torch.optim as optim
from function_utils import transfer_stateDict_to_vector
from misc.utils import *
from torch_geometric.nn import GCNConv,GATConv,global_mean_pool
from copy import deepcopy
from torch_geometric.utils import add_self_loops, remove_self_loops

class GCN(nn.Module):
    # def __init__(self,n_feat,n_dims,n_class,args,n_hid=32,layer = 2):
    def __init__(self, args, dropout=0.5, n_hid=128, layer=2):
        super().__init__()
        self.args = args
        self.n_feat = args.n_feat
        self.n_class = args.n_clss
        self.hidden_sizes = [n_hid]
        self.layer_norm_first = False
        self.use_ln = False
        self.lr = args.base_lr
        self.dropout = dropout
        self.weight_decay = 5e-4

        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.n_feat, n_hid))
        # print("n_feat",n_feat)
        # print("n_hid",n_hid)
        # self.lns = nn.ModuleList()
        # self.lns.append(torch.nn.LayerNorm(self.n_feat))

        for i in range(layer - 2):
            self.convs.append(GCNConv(n_hid, n_hid))
            # self.lns.append(nn.LayerNorm(n_hid))
        # self.lns.append(nn.LayerNorm(n_hid))
        self.gc2 = GCNConv(n_hid, self.n_class)

    def check(self):
        print(self.n_feat)
        print(self.hidden_sizes)
        print(self.n_class)
        print(self.convs)
        print(self.lns)
        print(self.gc2)

    # def forward(self, data, is_proxy=False):
    # def forward(self, data):
    def forward(self, x, edge_index, edge_weight=None):

        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # print(x.size())
        # print(edge_index.size())
        # edge_weight = None
        if (self.layer_norm_first):
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            # print(i+1)
            # print(x.dtype)
            # print(edge_index.dtype)
            # print(edge_weight)
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = function_utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, device, optimizer, idx_val=None,
            train_iters=200, verbose=False):
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

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(device)

        self.labels = labels.to(device)
        self.optimizer = optimizer
        self.device = device
        self._train_without_val(self.labels, idx_train, idx_attach, train_iters, verbose)

        # if idx_val is None:
        #     self._train_without_val(self.labels, idx_train, train_iters, verbose)
        # else:
        #     self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # # torch.cuda.empty_cache()

    def _train_without_val(self, labels, idx_train, idx_attach, train_iters, verbose):
        self.train()
        # print(idx_train)
        # print(idx_attach)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            if (idx_attach != None):

                # clean step
                self.optimizer.zero_grad()
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                idx_clean = list(set(idx_train.cpu().numpy()) - set(idx_attach.cpu().numpy()))
                idx_clean = torch.tensor(idx_clean).to(self.device)

                loss_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
                loss_clean = loss_clean * len(idx_clean) / len(idx_train)

                loss_clean.backward(retain_graph=True)
                clean_gradient = {}
                for name, params in self.named_parameters():
                    grad = params.grad
                    if grad is not None:
                        # print(name)
                        clean_gradient[name] = grad

                self.clean_gradient = clean_gradient
                # if (self.attach_gradient != None):
                #     for name, params in self.named_parameters():
                #         if params.grad is not None:
                #             tmp_attach = self.attach_gradient[name].clone()
                #             tmp_clean = self.clean_gradient[name].clone()
                #             tmp_clean_hat = tmp_clean - (tmp_attach.flatten().dot(tmp_clean.flatten()) / torch.linalg.norm(tmp_attach.flatten())) * tmp_attach
                #             params.grad = tmp_clean_hat
                #
                # self.optimizer.step()

                # backdoor step
                self.optimizer.zero_grad()
                # output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_attach = F.nll_loss(output[idx_attach], labels[idx_attach])
                loss_attach = loss_attach * len(idx_attach) / len(idx_train)
                loss_attach.backward()
                attach_gradient = {}
                for name, params in self.named_parameters():
                    grad = params.grad
                    if grad is not None:
                        # print(name)
                        attach_gradient[name] = grad

                self.attach_gradient = attach_gradient
                # if(self.clean_gradient!=None):
                #     for name, params in self.named_parameters():
                #         if params.grad is not None:
                #             tmp_attach = self.attach_gradient[name].clone()
                #             tmp_clean = self.clean_gradient[name].clone()
                #             tmp_attach_hat = tmp_attach - (tmp_attach.flatten().dot(tmp_clean.flatten()) / torch.linalg.norm(tmp_clean.flatten())) * tmp_clean
                #             params.grad = tmp_attach_hat

                # self.optimizer.step()

                for name, params in self.named_parameters():
                    if params.grad is not None:
                        params.grad = params.grad + self.clean_gradient[name]

                self.optimizer.step()

                '''
                for name, params in self.named_parameters():
                    if params.grad is not None:
                        tmp_attach = self.attach_gradient[name].clone()
                        tmp_clean = self.clean_gradient[name].clone()
                        tmp_attach_hat = tmp_attach -(tmp_attach.flatten().dot(tmp_clean.flatten())/torch.linalg.norm(tmp_clean.flatten())) * tmp_clean
                        tmp_clean_hat = tmp_clean -(tmp_attach.flatten().dot(tmp_clean.flatten())/torch.linalg.norm(tmp_attach.flatten())) * tmp_attach

                        params.grad = tmp_attach_hat + tmp_clean_hat
                self.optimizer.step()
                '''



            else:
                self.optimizer.zero_grad()
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train.backward()
                self.optimizer.step()
                if verbose and i % 10 == 0:
                    print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()


class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from models.layers import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x


class GCN3(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, x, edge_index, edge_weight=None):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        # print("GCN:",torch.sum(x[0]))
        # exit()
        x = self.clsif(x)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, x, edge_index, edge_weight=None, idx=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        if idx is None:
            return x
        else:
            return x[idx]

    def nolog_output(self, x, edge_index, edge_weight=None):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        # print("GCN:",torch.sum(x[0]))
        # exit()
        x = self.clsif(x)
        return x

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = function_utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)


class fpd_ae(nn.Module):
    def __init__(self, n_feat=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.args = args

        self.input_channel = nn.Linear(self.n_feat, int(self.n_feat / 2))
        self.output_channel = nn.Linear(int(self.n_feat / 2), self.n_feat)

    def forward(self, x):
        x = F.relu(self.input_channel(x))
        x = F.dropout(x, training=self.training)
        x = self.output_channel(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x


class DPGBA_MLP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):
        super(DPGBA_MLP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr

        self.weight_decay = weight_decay
        self.body = nn.Sequential(nn.Linear(nfeat, nhid),
                                  nn.ReLU(),
                                  nn.Linear(nhid, nhid))
        # nn.ReLU(),
        # nn.Linear(nhid,nclass))
        self.output = None
        self.best_model = None
        self.best_output = None

    def forward(self, x):
        return F.log_softmax(self.body(x), dim=1)


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size // 3),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(input_size // 3, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size),
            nn.Sigmoid()  # Use Sigmoid if the input data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DPGBA_MLPAE(nn.Module):
    def __init__(self, ori_x, trigger, device, epochs):
        super(DPGBA_MLPAE, self).__init__()
        self.device = device
        self.model = Autoencoder(len(ori_x[0])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.ori_x = ori_x
        self.trigger = trigger

    def fit(self):
        for epoch in range(self.epochs):
            output = self.model(self.ori_x)
            loss = self.criterion(output, self.ori_x)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # def inference(self, input):
    #     self.model.eval()
    #     reconstruction_errors = []
    #     with torch.no_grad():
    #         for sample in input:
    #             reconstructed = self.model(sample)
    #             loss = self.criterion(reconstructed, sample)
    #             reconstruction_errors.append(loss.item())
    #     return reconstruction_errors
    def inference(self, input):
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for sample in input:
                reconstructed = self.model(sample)
                loss = self.criterion(reconstructed, sample)
                reconstruction_errors.append(loss)

        # Convert the list of tensors to a single tensor
        reconstruction_errors_tensor = torch.stack(reconstruction_errors)
        return reconstruction_errors_tensor


class DPGBA_SD_GCN3(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, x, edge_index, edge_weight=None):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        all_features = x
        # print(x.shape)
        # print("GCN:",torch.sum(x[0]))
        # exit()
        x = self.clsif(x)
        return F.log_softmax(x, dim=1), all_features

    def get_embedding(self, x, edge_index, edge_weight=None, idx=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        if idx is None:
            return x
        else:
            return x[idx]

    def nolog_output(self, x, edge_index, edge_weight=None):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        # if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # print(x.shape)
        # print("GCN:",torch.sum(x[0]))
        # exit()
        x = self.clsif(x)
        return x

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(features, edge_index, edge_weight)
            acc_test = function_utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)


class GCN_TGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass,args,dropout=0.5, lr=0.01, weight_decay=5e-4, layer=3, device=None,
                 layer_norm_first=True,use_ln=True):

        super(GCN_TGE, self).__init__()

        assert device is not None, "Please specify 'device'!"

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer - 2):
            self.convs.append(GCNConv(nhid, nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        # self.gc2 = GCNConv(nhid, nclass)
        self.clsif = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):
                nn.init.kaiming_normal_(m.lin.weight)
                if m.lin.bias is not None:
                    nn.init.constant_(m.lin.bias, 0)

    def forward(self, x, edge_index, edge_weight=None):
        x.requires_grad_(True)
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index, edge_weight)
        x = self.clsif(x)
        log_softmax_output = F.log_softmax(x, dim=1)
        return log_softmax_output

    def forward_energy(self, x, edge_index, edge_weight=None):
        x.requires_grad_(True)
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index, edge_weight)
        x = self.clsif(x)
        p = x.logsumexp(dim=1)
        return p

    def get_h(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

    def get_embedding(self, x, edge_index, edge_weight=None, idx=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))

        x = F.dropout(x, training=self.training)

        if idx is None:
            return x
        else:
            return x[idx]

    def fit(self, global_model, features, edge_index, edge_weight, aug_edge_index, aug_edge_weight, labels, idx_train,
            args, idx_val=None, train_iters=200, verbose=False):
        #set_random_seed(args.seed)

        self.edge_index, self.edge_weight = edge_index, edge_weight  # Original graph
        self.aug_edge_index, self.aug_edge_weight = aug_edge_index, aug_edge_weight  # Augmented graph
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(self, global_model, features, labels,
                                                                            idx_train, idx_val, edge_index, edge_weight,
                                                                            aug_edge_index, aug_edge_weight,
                                                                            train_iters, verbose, args)
        return loss_train, loss_val, acc_train, acc_val

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            #set_random_seed(args.seed + i)
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, global_model, features, labels, idx_train, idx_val, edge_index, edge_weight,
                        aug_edge_index, aug_edge_weight, train_iters, args, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = -10

        for i in range(train_iters):
            #set_random_seed(args.seed + i)
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            total_loss = loss_train
            total_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                output = self.forward(features, edge_index, edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = self.accuracy(output[idx_val], labels[idx_val])
                acc_train = self.accuracy(output[idx_train], labels[idx_train])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

        return loss_train.item(), loss_val.item(), acc_train, acc_val


    def adjust_bn_layers(self, features, edge_index, edge_weight, aug_edge_index, aug_edge_weight):
        # begin_sd = deepcopy(self.state_dict())
        bn_params = []
        num_nodes = features.size(0)
        for name, param in self.named_parameters():
            if 'lns' in name:
                bn_params.append(param)

        optimizer = optim.Adam(bn_params, lr=self.lr, weight_decay=self.weight_decay)
        self.train()
        optimizer.zero_grad()
        p_data = self.forward_energy(features, edge_index, edge_weight)
        shuf_feats = features[:, torch.randperm(features.size(1))]  # shuffle features
        p_neigh = self.forward_energy(shuf_feats, aug_edge_index, aug_edge_weight)
        energy = p_data - p_neigh / p_data
        features.requires_grad_(True)
        energy_grad = torch.autograd.grad(energy.sum(), features, create_graph=True)[0]
        energy_grad_inner = torch.sum(energy_grad ** 2)
        energy_squared_sum = torch.sum(energy ** 2)
        neigh_loss = 1 / num_nodes * (energy_grad_inner + 1 / 2 * energy_squared_sum)
        neigh_loss.backward()
        optimizer.step()
        # now_sd = deepcopy(self.state_dict())
        # begin_vector = transfer_stateDict_to_vector(begin_sd).flatten()
        # now_vector = transfer_stateDict_to_vector(now_sd).flatten()

        #print("opt_norm:",torch.linalg.norm(now_vector - begin_vector))

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = self.accuracy(output[idx_test], labels[idx_test])

        return float(acc_test)

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()  # return a tensor
        acc_test = self.accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids

    def accuracy(self,output, labels):
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


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass,args,heads=1,dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True):

        super(GAT, self).__init__()

        self.nfeat = nfeat
        self.args = args
        self.hidden_sizes = [nhid]
        self.n_clss = nclass
        self.gc1 = GATConv(nfeat,nhid,heads,dropout=dropout)
        self.gc2 = GATConv(heads*nhid, nhid, concat=False, dropout=dropout)
        self.clsif = nn.Linear(nhid, self.n_clss)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay

        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def _add_self_loops_to_edge_index(self, edge_index, num_nodes):
        # 先去掉已有自环，再统一加一次，避免重复
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def _build_iso_edge_index(self, num_nodes, device):
        # 只保留 self-loop: (i, i)
        idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)


    def forward(self, x, edge_index, edge_weight=None):

        num_nodes = x.size(0)
        edge_index = self._add_self_loops_to_edge_index(edge_index, num_nodes)
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clsif(x)

        return F.log_softmax(x,dim=1)

    def iso_forward(self, x,edge_index, edge_weight=None):
        """
        前向传播时不聚合邻居信息：
        只保留 self-loop，让每个节点只看自己。
        """
        num_nodes = x.size(0)
        iso_edge_index = self._build_iso_edge_index(num_nodes, x.device)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, iso_edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, iso_edge_index)
        x = F.relu(x)

        x = self.clsif(x)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, x, edge_index, edge_weight=None, idx=None):

        num_nodes = x.size(0)
        edge_index = self._add_self_loops_to_edge_index(edge_index, num_nodes)
        x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))

        if idx is None:
            return x
        else:
            return x[idx]

    def get_iso_embedding(self, x, edge_index, edge_weight=None, idx=None):

        num_nodes = x.size(0)
        iso_edge_index = self._build_iso_edge_index(num_nodes, x.device)
        x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        x = F.relu(self.gc1(x, iso_edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, iso_edge_index))

        if idx is None:
            return x
        else:
            return x[idx]

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = function_utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops


class GCN_Graph_Classifier(nn.Module):
    def __init__(self, n_feat, n_dims, n_class, args, dropout=0.5):
        super(GCN_Graph_Classifier, self).__init__()

        self.args = args

        # ===== 尽量保持原接口不变，但内部支持更通用的结构 =====
        self.in_dim = n_feat
        self.hidden_dim = n_dims
        self.out_dim = n_dims
        self.n_class = n_class

        self.n_layers = 4
        self.readout = "mean"
        self.batch_norm = False
        self.residual = True

        self.in_feat_dropout_p = 0.0
        self.dropout = 0.0

        # ===== 输入映射层 =====
        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)
        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout_p)

        # ===== GCN层堆叠 =====
        # 前 n_layers-1 层: hidden_dim -> hidden_dim
        # 最后一层: hidden_dim -> out_dim
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.n_layers - 1):
            self.layers.append(GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=False))
            if self.batch_norm:
                self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.layers.append(GCNConv(self.hidden_dim, self.out_dim, add_self_loops=False))
        if self.batch_norm:
            self.norms.append(nn.BatchNorm1d(self.out_dim))

        # ===== 忽略别人代码里的 MLPReadout，保留我们原来的简单分类头 =====
        self.fc = nn.Linear(self.out_dim, self.n_class)

    # ---------------------------------------------------------
    # 图结构控制：normal / isolate
    # ---------------------------------------------------------
    def _add_self_loops_to_edge_index(self, edge_index, num_nodes):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def _build_isolate_edge_index(self, num_nodes, device):
        idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)

    # ---------------------------------------------------------
    # 编码器：参照别人那份结构
    # embedding_h -> in_feat_dropout -> 多层GCN
    # ---------------------------------------------------------
    def _encode(self, x, edge_index):
        x = self.embedding_h(x)
        x = self.in_feat_dropout(x)

        for layer_idx, conv in enumerate(self.layers):
            h_in = x

            x = conv(x, edge_index)
            x = F.relu(x)

            if self.batch_norm:
                x = self.norms[layer_idx](x)

            x = F.dropout(x, p=self.dropout, training=self.training)

            # residual 只在维度匹配时使用
            if self.residual and x.shape == h_in.shape:
                x = x + h_in

        return x

    # ---------------------------------------------------------
    # readout
    # ---------------------------------------------------------
    def _readout(self, node_emb, batch):
        if self.readout == "sum":
            graph_emb = global_add_pool(node_emb, batch)
        elif self.readout == "max":
            graph_emb = global_max_pool(node_emb, batch)
        else:
            graph_emb = global_mean_pool(node_emb, batch)

        return graph_emb

    # ---------------------------------------------------------
    # node embedding
    # ---------------------------------------------------------
    def get_node_embedding(self, x, edge_index):
        """
        normal setting: 原图 + self-loop
        """
        num_nodes = x.size(0)
        edge_index = self._add_self_loops_to_edge_index(edge_index, num_nodes)
        return self._encode(x, edge_index)

    def get_isolate_node_embedding(self, x):
        """
        isolate setting: 只保留 self-loop
        """
        num_nodes = x.size(0)
        iso_edge_index = self._build_isolate_edge_index(num_nodes, x.device)
        return self._encode(x, iso_edge_index)

    # ---------------------------------------------------------
    # graph embedding
    # ---------------------------------------------------------
    def get_graph_embedding(self, x, edge_index, batch):
        node_emb = self.get_node_embedding(x, edge_index)
        graph_emb = self._readout(node_emb, batch)
        return graph_emb

    def get_isolate_graph_embedding(self, x, batch):
        node_emb = self.get_isolate_node_embedding(x)
        graph_emb = self._readout(node_emb, batch)
        return graph_emb

    # ---------------------------------------------------------
    # forward
    # ---------------------------------------------------------
    def normal_forward(self, x, edge_index, batch, edge_attr=None, return_embedding=False):
        graph_emb = self.get_graph_embedding(x, edge_index, batch)

        if return_embedding:
            return graph_emb

        out = self.fc(graph_emb)
        return F.log_softmax(out, dim=1)

    def isolate_forward(self, x, batch, edge_attr=None, return_embedding=False):
        graph_emb = self.get_isolate_graph_embedding(x, batch)

        if return_embedding:
            return graph_emb

        out = self.fc(graph_emb)
        return F.log_softmax(out, dim=1)

    def forward(self, x, edge_index, batch, edge_attr=None, return_embedding=False):
        return self.normal_forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=edge_attr,
            return_embedding=return_embedding
        )