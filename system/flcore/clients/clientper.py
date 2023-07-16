import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.augmentation import generate_views
from utils.general_utils import parse_param_json


class clientPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for data in trainloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                if self.ssl_enabled:
                    # parse ssl hyper-parameters from json file
                    params = parse_param_json(self.param)

                    _, x_2, _, edge_index_2 = generate_views(params, data, self.device)
                    z1 = self.model((data.x, data.edge_index))
                    z2 = self.model((x_2, edge_index_2))
                    output = self.model.generate_logits(z1)

                    contrastive_loss = self.model.contrastive_loss(z1,z2)
                    ce_loss = self.loss(output[data.train_mask], data.y[data.train_mask])
                    loss = ce_loss + params['scale_ratio'] * contrastive_loss
                else:
                    output = self.model(data)
                    loss = self.loss(output[data.train_mask], data.y[data.train_mask])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
