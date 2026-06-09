from Client_Clean_test import Clean_test
import argparse
from datetime import datetime
import torch
import numpy as np
import os
import random
import torch.multiprocessing as mp

from function_utils import multi_print


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, default='0')
        self.parser.add_argument('--seed', type=int, default=1234)

        #模型基本设置
        self.parser.add_argument('--dataset', type=str, default="CiteSeer")
        self.parser.add_argument('--base-path', type=str, default='./')
        self.parser.add_argument('--mode', type=str, default="disjoint", choices=['disjoint', 'overlapping'])
        self.parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--n_clients', type=int, default=10)
        self.parser.add_argument('--n_workers', type=int, default=10)
        self.parser.add_argument('--model', type=str, default="fedavg")
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument('--n_dims', type=int, default=128)
        self.parser.add_argument('--frac', type=float, default=1.0)
        self.parser.add_argument('--n_rnds', type=int, default=100)
        self.parser.add_argument('--n_eps', type=int, default=1)
        self.parser.add_argument('--MA_eps', type=int, default=1)

        self.parser.add_argument('--target_class', type=int, default=0)
        self.parser.add_argument('--thrd', type=float, default=0.5)
        self.parser.add_argument('--trigger_size', type=int, default=3, help='tirgger_size')

        self.parser.add_argument('--Attacker_id', type=str, default="0")
        self.parser.add_argument('--A_size', type=int, default=40)
        self.parser.add_argument('--RG_size', type=int, default=20)
        self.parser.add_argument('--attack_begin', type=int, default=25)
        self.parser.add_argument('--attack_end', type=int, default=150)
        self.parser.add_argument('--add_dim_threshold', type=float, default=0.3)
        self.parser.add_argument('--use_vs_number', action='store_true', default=False,help="if use detailed number to decide Vs")
        self.parser.add_argument('--vs_ratio', type=float, default=0.3, help="ratio of poisoning nodes relative to the full graph")
        self.parser.add_argument('--vs_number', type=int, default=10,help="number of poisoning nodes relative to the full graph")
        self.parser.add_argument('--idx_attach_name', type=str, default="None",help = "declare the idx_attach file name")
        self.parser.add_argument('--trojan_name', type=str, default="backdoor_generator_state0.pt", help="declare the trojan file name")
        self.parser.add_argument('--attack_method', type=str, default="SAO", help="identify the attack method,"+
                                                                                   "["+
                                                                                   "normal,my_work,cross_projection,cross_step_projection,"+
                                                                                   "critical_layer,loss_scale,global_cross_projection,"+
                                                                                   "orthogonal"+ "penalty"+
                                                                                   "]")
        self.parser.add_argument('--lr', type=float, default=0.01)
        self.parser.add_argument('--dis_weight', type=float, default=1, help="Weight of cluster distance")
        self.parser.add_argument('--selection_method', type=str, default='normal+new',choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                                 help='Method to select idx_attach for training trojan model (none means randomly select)')


        #范数设置
        self.parser.add_argument('--l1', type=float, default=1e-3)
        self.parser.add_argument('--loc-l2', type=float, default=1e-3)

        self.parser.add_argument('--trojan_path', type=str, default="None", help="declare the trojan file path")

        self.parser.add_argument('--alpha', type=float, default=0.02,
                            help="Ratio of feature dimensions to perturb")
        self.parser.add_argument('--alpha_int', type=int, default=30,
                            help="Number of feature dimensions to perturb")


    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            print('Ignoring unknown arguments in cross_test: {}'.format(unparsed))
        return args

def set_config_main(args,ck_path,main_args=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if(main_args is not None):
        args.n_clients = main_args.n_clients
        args.dataset = main_args.dataset
        args.target_class = main_args.target_class
        args.attack_method = main_args.attack_method
        args.Attacker_id = main_args.Attacker_id
        args.model = main_args.model
        args.output_file = main_args.output_file

    args.base_lr = 1e-2
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-6
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0


    if args.dataset == 'Cora':
        args.n_feat = 1433
        args.n_clss = 7

    if args.dataset == 'CiteSeer':
        args.n_feat = 3703
        args.n_clss = 6

    if args.dataset == 'PubMed':
        args.n_feat = 500
        args.n_clss = 3

    if args.dataset == 'Computers':
        args.n_feat = 767
        args.n_clss = 10

    if args.dataset == 'Flickr':
        args.n_feat = 500
        args.n_clss = 7

    if args.dataset == 'ogbn-arxiv':
        args.n_feat = 128
        args.n_clss = 40

    if args.dataset == 'Amazon-ratings':
        args.n_feat = 300
        args.n_clss = 5


    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'

    args.data_path = f'{args.base_path}/datasets'
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'
    args.output_file = "/output.txt"
    if args.debug == True:
        args.checkpt_path = f'{args.base_path}/debug/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/debug/logs/{trial}'

    args.trojan_path = ck_path
    args.checkpt_path = ck_path

    return args

def cross_test(ck_path,main_args=None):
    print(ck_path)
    args = Parser().parse()
    args = set_config_main(args,ck_path,main_args)

    from Client_Cross_test import Cross_test

    acc_list = []
    asr_list = []
    g_id = 0
    for client_id in range(0,args.n_clients):
        test_class = Cross_test(args = args,client_id = client_id,g_id=g_id)
        test_class.do_test(acc_list,asr_list)

    avg_acc = (sum(acc_list) / len(acc_list))
    avg_acc = '%.2f' % avg_acc
    avg_acc = float(avg_acc)

    avg_asr = (sum(asr_list) / len(asr_list))
    avg_asr = '%.2f' % avg_asr
    avg_asr = float(avg_asr)

    print("mean acc:",avg_acc)
    print("mean asr:",avg_asr)
    multi_print(args.checkpt_path + args.output_file,
                "mean acc:{}".format(avg_acc))
    multi_print(args.checkpt_path + args.output_file,
                "mean asr:{}".format(avg_asr))

    return acc_list,asr_list,avg_acc,avg_asr

def clean_cross(ck_path,main_args=None):
    print(ck_path)
    args = Parser().parse()
    args = set_config_main(args,ck_path,main_args)
    acc_list = []
    g_id = 0
    for client_id in range(0,args.n_clients):
        test_class = Clean_test(args = args,client_id = client_id,g_id=g_id)
        test_class.do_test(acc_list)

    avg_acc = (sum(acc_list) / len(acc_list))
    avg_acc = '%.2f' % avg_acc
    avg_acc = float(avg_acc)

    print("mean acc:",avg_acc)
    multi_print(args.checkpt_path + args.output_file,
                "mean acc:{}".format(avg_acc))

    return acc_list,avg_acc

if __name__ == "__main__":
    ck_path = ""
    cross_test(ck_path)
