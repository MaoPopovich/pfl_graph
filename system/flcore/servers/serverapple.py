import torch
import copy
import random
import time
from flcore.clients.clientapple import clientAPPLE
from flcore.servers.serverbase import Server
from threading import Thread
from utils.dlg import DLG
import logging

logger = logging.getLogger(__name__)

class APPLE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAPPLE)

        logger.info("Join ratio / total clients: {:.2f} / {:d}".format(self.join_ratio, self.num_clients))
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.client_models = [copy.deepcopy(c.model_c) for c in self.clients]

        train_samples = 0
        for client in self.clients:
            train_samples += client.train_samples
        p0 = [client.train_samples / train_samples for client in self.clients]

        for c in self.clients:
            c.p0 = p0


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                logger.info(f"-------------Round number: {i}-------------")
                logger.info("Evaluate personalized models")
                self.evaluate()

            for client in self.clients:
                client.train(i)

            # threads = [Thread(target=client.train)
            #            for client in self.clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.receive_models()

            self.Budget.append(time.time() - s_t)
            logger.info('-'*25 + 'time cost' + '-'*25 + str(self.Budget[-1]))

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("Best accuracy:" + str(max(self.rs_test_acc).item()))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logger.info("Average time cost per round:" + str(sum(self.Budget[1:])/len(self.Budget[1:])))

        self.save_results()
        

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_models(self.client_models)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.client_models[client.id] = copy.deepcopy(client.model_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model_server in zip(range(self.num_clients), self.client_models):
            client_model = self.clients[cid].model
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(client_model_server.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

