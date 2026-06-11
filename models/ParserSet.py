import argparse
class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, default='0')
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument('--model', type=str, default="fedavg")
        self.parser.add_argument('--dataset', type=str, default="CiteSeer")
        self.parser.add_argument('--mode', type=str, default="disjoint", choices=['disjoint', 'overlapping'])
        self.parser.add_argument('--base_path', type=str, default='./')

        self.parser.add_argument('--n_workers', type=int, default=10)
        self.parser.add_argument('--n_clients', type=int, default=10)
        self.parser.add_argument('--n_rnds', type=int, default=40)
        self.parser.add_argument('--n_eps', type=int, default=1)
        self.parser.add_argument('--frac', type=float, default=1.0)
        self.parser.add_argument('--loc_l2', type=float, default=1e-3)

        self.parser.add_argument('--n_dims', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=0.01)

        self.parser.add_argument('--laye_mask-one', action='store_true')
        self.parser.add_argument('--clsf_mask-one', action='store_true')

        self.parser.add_argument('--agg_norm', type=str, default='exp', choices=['cosine', 'exp'])
        self.parser.add_argument('--norm_scale', type=float, default=10)
        self.parser.add_argument('--n_proxy', type=int, default=5)

        self.parser.add_argument('--l1', type=float, default=1e-3)
        self.parser.add_argument('--debug', action='store_true')


        self.parser.add_argument('--selection_method', type=str, default='cluster',choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                                 help='Method to select idx_attach for training trojan model (none means randomly select)')
        self.parser.add_argument('--target_loss_weight', type=float, default=5,
                            help="Weight of optimize outter trigger generator")
        self.parser.add_argument('--homo_loss_weight', type=float, default=1,
                            help="Weight of optimize similarity loss")
        self.parser.add_argument('--orthogonal_loss_weight', type=float, default=5,
                            help="Weight of optimize similarity loss")
        self.parser.add_argument('--sim_loss_weight', type=float, default=5,
                            help="Weight of optimize similarity loss")
        self.parser.add_argument('--trigger_loss_weight', type=float, default=5,
                            help="Weight of optimize trigger orthogonal loss")
        self.parser.add_argument('--norm_loss_weight', type=float, default=0,
                            help="Weight of optimize trigger norm loss")
        self.parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                            help="Threshold of increase similarity")

        self.parser.add_argument('--defense_mode', type=str, default="none",choices=['prune', 'isolate', 'none'],help="Mode of defense")
        self.parser.add_argument('--use_vs_number', action='store_true', default=False,help="if use detailed number to decide Vs")
        self.parser.add_argument('--trigger_size', type=int, default=3, help='tirgger_size')
        self.parser.add_argument('--vs_ratio', type=float, default=0.3, help="ratio of poisoning nodes relative to the full graph")
        self.parser.add_argument('--vs_number', type=int, default=10,help="number of poisoning nodes relative to the full graph")
        self.parser.add_argument('--dis_weight', type=float, default=1, help="Weight of cluster distance")
        self.parser.add_argument('--target_class', type=int, default=0)
        self.parser.add_argument('--trojan_epochs', type=int, default=100,help='Number of epochs to train trigger generator.')
        self.parser.add_argument('--inner', type=int, default=1, help='Number of inner')
        self.parser.add_argument('--thrd', type=float, default=0.5)
        self.parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--test_model_seed', type=int, default=10)
        self.parser.add_argument('--MA_eps', type=int, default=1)


        self.parser.add_argument('--Attacker_id', type=str, default="0")
        self.parser.add_argument('--A_size', type=int, default=40)
        self.parser.add_argument('--RG_size', type=int, default=20)
        self.parser.add_argument('--attack_begin', type=int, default=30)
        self.parser.add_argument('--attack_end', type=int, default=200)
        self.parser.add_argument('--add_dim_threshold', type=float, default=0.2)
        self.parser.add_argument('--record_begin', type=int, default=0,help = "when to begin record the delta list,in common," +
                                                                               "it should be Multiples of ten,just like 9 or 19." +
                                                                               "because delta_gradient is 1 shorter than delta_weights")

        self.parser.add_argument('--idx_attach_name', type=str, default="None",help = "declare the idx_attach file name")
        self.parser.add_argument('--trojan_path', type=str, default="None", help="declare the trojan file path")
        self.parser.add_argument('--trojan_name', type=str, default="None", help="declare the trojan file name")
        self.parser.add_argument('--attack_method', type=str, default="SAO", help="identify the attack method")
        self.parser.add_argument('--sd_P', type=bool, default=False)



    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
