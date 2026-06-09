from datetime import datetime
from time import sleep
from misc.utils import *
from modules.multiprocs_MA import ParentProcess as MAprocess
from modules.multiprocs import ParentProcess as process
from cross_test import cross_test,clean_cross
from models.ParserSet import Parser
import importlib


def dynamic_import_object(module_path, object_name):
    module = importlib.import_module(module_path)
    return getattr(module, object_name)

def main(args):
    module_name = "models."+args.model
    sever_name = module_name+".server"
    client_name = module_name+".client"

    Server = dynamic_import_object(sever_name,"Server")
    Client = dynamic_import_object(client_name,"Client")

    if(args.attack_method != "none"):
        MA_client_name = module_name + ".Malicious_Client_" + args.attack_method
        MAClient = dynamic_import_object(MA_client_name,"MAClient")
        pp = MAprocess(args, Server, Client, MAClient)

    if(args.attack_method=="none"):
        pp = process(args, Server, Client)

    pp.start()
    sleep(15)

    if(args.attack_method=="none"):
        acc_list, avg_acc = clean_cross(args.checkpt_path, args)
        return acc_list,avg_acc
    else:
        acc_list, asr_list, avg_acc, avg_asr = cross_test(args.checkpt_path,args)
        return acc_list, asr_list, avg_acc, avg_asr

def set_config(args,i):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    args.base_lr = 0.01
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
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}_{args.attack_method}'

    args.data_path = f'{args.base_path}/datasets'
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    os.mkdir(args.checkpt_path)
    args.log_path = f'{args.base_path}/logs/{trial}'
    args.output_file = "/output.txt"
    with open(args.checkpt_path + args.output_file , 'a') as f:
        print("output_file",file=f)

    if args.debug == True:
        args.checkpt_path = f'{args.base_path}/debug/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/debug/logs/{trial}'

    return args

if __name__ == '__main__':
    start = 0
    end = 1
    acc_llist = []
    asr_llist = []
    avg_acc_list = []
    avg_asr_list = []
    for i in range(start,end):
        args = Parser().parse()
        args = set_config(args,i)
        if(args.attack_method == "none"):
            acc_list,avg_acc = main(args)
        else:
            acc_list,asr_list,avg_acc,avg_asr = main(args)
            avg_asr_list.append(avg_asr)
            asr_llist.append(asr_list)

        acc_llist.append(acc_list)
        avg_acc_list.append(avg_acc)


    for i in range((end-start)):
        print("i-th acc list :",acc_llist[i]," avg_acc is: ",avg_acc_list[i])


    avg_acc_list = np.array(avg_acc_list)
    acc = np.mean(avg_acc_list)
    acc_std = np.std(avg_acc_list)
    acc = '%.2f' % acc
    acc = float(acc)
    acc_std = '%.2f' % acc_std
    acc_std = float(acc_std)
    print("all avg  ",
          "acc:", acc, " / ", acc_std,
          )

    if(not args.attack_method == "none"):
        for i in range((end-start)):
            print(" asr_list: ",asr_llist[i]," avg_asr is: ",avg_asr_list[i])

        asr_acc_list = np.array(avg_asr_list)
        asr = np.mean(avg_asr_list)
        asr_std = np.std(avg_asr_list)
        asr = '%.2f' % asr
        asr = float(asr)
        asr_std = '%.2f' % asr_std
        asr_std = float(asr_std)
        print("all avg  ",
              "asr:", asr, " / ", asr_std,
              )











