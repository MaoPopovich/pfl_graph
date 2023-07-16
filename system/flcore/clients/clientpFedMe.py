import numpy as np
import time
import copy
import torch
import torch.nn as nn
from flcore.optimizers.fedoptimizer import pFedMeOptimizer
from flcore.clients.clientbase import Client


class clientpFedMe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda = args.lamda
        self.K = args.K
        self.personalized_learning_rate = args.p_learning_rate

        # these parameters are for personalized federated learing.
        self.local_params = copy.deepcopy(list(self.model.parameters()))
        self.personalized_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = pFedMeOptimizer(
            self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            for data in trainloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # K is number of personalized steps
                for i in range(self.K):
                    output = self.model(data)
                    loss = self.loss(output[data.train_mask], data.y[data.train_mask])
                    self.optimizer.zero_grad()
                    loss.backward()
                    # finding aproximate theta
                    self.personalized_params = self.optimizer.step(self.local_params, self.device)

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.update_parameters(self.model, self.local_params)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, model):
        for new_param, old_param, local_param in zip(model.parameters(), self.model.parameters(), self.local_params):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def test_metrics_personalized(self):
        testloader = self.load_data(batch_size=1)
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for data in testloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                output = self.model(data)
                test_acc += torch.sum(output.argmax(dim=-1)[data.test_mask] == data.y[data.test_mask]).item()
                test_num += data.test_mask.sum()

        # self.model.cpu()
        
        return test_acc, test_num

    def train_metrics_personalized(self):
        trainloader = self.load_data(batch_size=1)
        self.update_parameters(self.model, self.personalized_params)
        # self.model.to(self.device)
        self.model.eval()

        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for data in trainloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                output = self.model(data)
                loss = self.loss(output[data.train_mask], data.y[data.train_mask]).item()

                lm = torch.cat([p.data.view(-1) for p in self.local_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.personalized_params], dim=0)
                loss += 0.5 * self.lamda * torch.norm(lm-pm, p=2)

                train_acc += torch.sum(torch.argmax(output, dim=1)[data.train_mask] == data.y[data.train_mask]).item()
                
                train_num += data.train_mask.sum()
                losses += loss.item() * data.train_mask.sum()

        # self.model.cpu()
        
        return train_acc, losses, train_num
