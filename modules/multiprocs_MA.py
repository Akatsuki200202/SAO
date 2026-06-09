import os
import sys
import time
import atexit
import numpy as np
import torch.multiprocessing as mp

from function_utils import multi_print
from misc.utils import *
from models.nets import *
#mp.set_start_method('spawn',force=True)
class ParentProcess:
    def __init__(self, args, Server, Client, MAClient):
        self.args = args
        self.gpus = [int(g) for g in args.gpu.split(',')]
        self.gpu_server = self.gpus[0]
        self.proc_id = os.getppid()
        print(f'main process id: {self.proc_id}')
        self.attacker_id_list = self.args.Attacker_id.split("+")
        for i in range(len(self.attacker_id_list)):
            self.attacker_id_list[i] = int(self.attacker_id_list[i])

        self.benign_id_list = set(list(range(self.args.n_clients))) - set(self.attacker_id_list)

        self.sd = mp.Manager().dict()
        self.sd['is_done'] = False
        self.create_workers(Client,MAClient)
        self.server = Server(args, self.sd, self.gpu_server) 
        atexit.register(self.done)

    def create_workers(self, Client, MAClient):
        self.processes = []
        self.q = {}
        #添加善意客户端的线程
        for worker_id in range(self.args.n_workers):
            #if(worker_id == self.args.Attacker_id):        #添加恶意客户端的线程
            if (worker_id in self.attacker_id_list):
                gpu_id = 0
                print(f'malicious_worker_id: {worker_id,}, gpu_id:{gpu_id}')
                multi_print(self.args.checkpt_path + self.args.output_file,
                            f'malicious_worker_id: {worker_id,}, gpu_id:{gpu_id}')

                self.q[worker_id] = mp.Queue()
                ma_p = mp.Process(target=MaliciousProcess, args=(self.args, worker_id, gpu_id, self.q[worker_id], self.sd, MAClient))
                ma_p.start()
                self.processes.append(ma_p)
            else:
                # gpu_id = self.gpus[worker_id] if worker_id <= len(self.gpus)-1 else self.gpus[worker_id%len(self.gpus)]
                gpu_id = self.gpus[worker_id+1] if worker_id < len(self.gpus)-1 else self.gpus[(worker_id-(len(self.gpus)-1))%len(self.gpus)]
                print(f'worker_id: {worker_id}, gpu_id:{gpu_id}')
                multi_print(self.args.checkpt_path + self.args.output_file,
                            f'worker_id: {worker_id}, gpu_id:{gpu_id}')

                self.q[worker_id] = mp.Queue()
                p = mp.Process(target=WorkerProcess, args=(self.args, worker_id, gpu_id, self.q[worker_id], self.sd, Client))
                p.start()
                self.processes.append(p)

    def start(self):
        self.sd['is_done'] = False

        basic_done = "done_"
        for i in range(self.args.n_clients):
            now_done = basic_done + str(i)
            self.sd[now_done]=False

        if os.path.isdir(self.args.checkpt_path) == False:
            os.makedirs(self.args.checkpt_path)
        if os.path.isdir(self.args.log_path) == False:
            os.makedirs(self.args.log_path)
        self.n_connected = round(self.args.n_clients*self.args.frac)

        print(self.n_connected)
        for curr_rnd in range(self.args.n_rnds):
            st = time.time()
            self.curr_rnd = curr_rnd
            print(f'----------[main] round {curr_rnd} begin---------------')
            multi_print(self.args.checkpt_path + self.args.output_file,
                        f'----------[main] round {curr_rnd} begin---------------')

            self.updated = set()
            np.random.seed(self.args.seed+curr_rnd)
            self.selected = np.random.choice(self.args.n_clients, self.n_connected, replace=False).tolist()
            ##################################################
            self.server.on_round_begin(curr_rnd)
            ##################################################
            while len(self.selected)>0:
                _selected = []
                for worker_id, q in self.q.items():
                    c_id = self.selected.pop(0)
                    _selected.append(c_id)
                    q.put((c_id, curr_rnd))
                    if len(self.selected) == 0:
                        break
                self.wait(curr_rnd, _selected)
            # print(f'[main] all clients updated at round {curr_rnd}')
            ###########################################
            self.server.on_round_complete(self.updated)
            ###########################################
            print(f'-----------[main] round {curr_rnd} done ({time.time()-st:.2f} s)-------------')
            multi_print(self.args.checkpt_path + self.args.output_file,
                        f'-----------[main] round {curr_rnd} done ({time.time()-st:.2f} s)-------------')

        #self.server.inner_data_save()

        self.sd['is_done'] = True
        for worker_id, q in self.q.items():
            q.put(None)
        print('[main] server done')
        multi_print(self.args.checkpt_path + self.args.output_file,
                    '[main] server done')
        #sys.exit()

    def wait(self, curr_rnd, _selected):
        cont = True
        while cont:
            cont = False
            for c_id in _selected:
                if not c_id in self.sd:
                    cont = True
                else:
                    self.updated.add(c_id)
            time.sleep(0.1)

    def done(self):
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive():
                print(f'[main] Force terminating child process {p.pid} ...')
                multi_print(self.args.checkpt_path + self.args.output_file,
                            f'[main] Force terminating child process {p.pid} ...')
                p.terminate()
                p.join(timeout=5)
        print('[main] All children have joined. Destroying main process ...')
        multi_print(self.args.checkpt_path + self.args.output_file,
                    '[main] All children have joined. Destroying main process ...')
        return
            

class WorkerProcess:
    def __init__(self, args, worker_id, gpu_id, q, sd, Client):
        self.q = q
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.is_done = False
        self.client = Client(args = self.args, w_id = self.worker_id,client_id = self.worker_id, g_id = self.gpu_id, sd =self.sd)
        self.listen()

    def listen(self):
        while not self.sd['is_done']:
            mesg = self.q.get()
            if not mesg == None:
                client_id, curr_rnd = mesg
                ##################################
                self.client.on_receive_message(curr_rnd)
                self.client.on_round_begin()#train
                self.client.save_state()
                self.client.do_test()
                ##################################
            time.sleep(1.0)

        print('[main] Terminating worker processes ... ')
        multi_print(self.args.checkpt_path + self.args.output_file,
                    '[main] Terminating worker processes ... ')

        self.client.inner_data_save()
        #self.client.save_log()
        #sys.exit()
        return

class MaliciousProcess:
    def __init__(self, args, worker_id, gpu_id, q, sd, MAClient):
        self.q = q
        self.sd = sd
        self.args = args
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.is_done = False
        self.client = MAClient(args = self.args, w_id = self.worker_id, g_id = self.gpu_id, sd = self.sd, client_id = self.worker_id)
        self.listen()

    def listen(self):
        while not self.sd['is_done']:
            mesg = self.q.get()
            if not mesg == None:
                client_id, curr_rnd = mesg
                ##################################
                self.client.on_receive_message(curr_rnd)
                self.client.on_round_begin()#train
                self.client.do_test()
                self.client.save_state()
                ##################################
            time.sleep(1.0)

        print('[main] Terminating Malicious processes ... ')
        return




