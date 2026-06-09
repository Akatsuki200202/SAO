import torch
import torch.nn as nn
import torch.nn.functional as F


class TrojanAwareNet(nn.Module):
    def __init__(self,args,n_feat,n_dims,n_out,device):
        super(TrojanAwareNet, self).__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        #self.n_dims = n_feat
        self.args = args
        self.n_out = n_out
        self.device = device
        self.thrd = self.args.thrd

        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(self.n_feat, self.n_feat, cached=False)
        self.x_mlp = nn.Linear(self.n_feat,self.n_out*self.n_feat)
        self.A_mlp = nn.Linear(self.n_feat, int(self.n_out*(self.n_out-1)/2))

    def forward(self,idx,x,edge_index,edge_weight=None):

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        tar_x = x[idx]


        feat = self.x_mlp(tar_x)
        edge_weight = self.A_mlp(tar_x)
        GW = GradWhere.apply
        edge_weight = GW(edge_weight, self.thrd, self.device)

        return feat,edge_weight

    def get_embedding(self,idx,x,edge_index,edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        if idx is None:
            return x
        else:
            return x[idx]

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class TriggerLoss(nn.Module):
    def __init__(self,args,device):
        super(TriggerLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self,clean_E,poisoned_E,B_list):

        loss = 0.0
        for i in range(len(clean_E)):
            delta_e = clean_E[i]-poisoned_E[i]
            delta_e=delta_e.flatten()
            now_B = B_list[i]
            sum = 0.0
            for j in range(len(now_B)):
                now_e = now_B[j]
                #print(now_e.shape)
                sum = sum + torch.abs(now_e.dot(delta_e) / (torch.linalg.norm(now_e) * torch.linalg.norm(delta_e)))

            sum = sum/len(now_B)
            loss = loss + sum
        loss = loss/len(clean_E)

        return loss

class DelatNormLoss(nn.Module):
    def __init__(self, args, device):
        super(DelatNormLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self,clean_E,poisoned_E):

        loss = 0.0
        for i in  range(len(clean_E)):
            delta_e = clean_E[i]-poisoned_E[i]
            delta_e=delta_e.flatten()
            loss = loss + torch.linalg.norm(delta_e)

        loss = loss/len(clean_E)
        loss = -1 * loss

        return loss

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss


class SimLoss(nn.Module):
    def __init__(self, args, device):
        super(SimLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, all,clean):
        loss = 0
        beta_2 = 0.05
        loss = beta_2 * torch.linalg.norm(all-clean)
        # print(edge_sims.min())
        return loss

class OrthogonalLoss(nn.Module):
    def __init__(self,args,device):
        super(OrthogonalLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self,A_list,x):

        loss = 0.0
        for i in range(len(A_list)):
            now_vector = A_list[i]
            now_vector = now_vector.to(self.device)
            now_vector = now_vector.to(torch.float32)
            loss = loss + torch.abs(now_vector.dot(x)/(torch.linalg.norm(now_vector) * torch.linalg.norm(x)))

        return loss / len(A_list)