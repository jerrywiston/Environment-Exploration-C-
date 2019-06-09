import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DOUBLE_DQN = True

class DeepQNetwork():
    def __init__(
        self,
        qnet,
        n_actions,
        learning_rate = 1e-4,
        reward_decay = 0.95,
        replace_target_iter = 300,
        memory_size = 500,
        batch_size = 64,
        e_greedy = 0.95,
        e_greedy_increment = None,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.2 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self._build_net(qnet)

    def _build_net(self, qnet):
        self.qnet_eval = qnet().to(device)
        self.qnet_target = qnet().to(device)
        self.qnet_target.eval()
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)

    def choose_action(self, s, get_q=True):
        # input only one sample
        actions_value = self.qnet_eval.forward([s], device)
        if np.random.uniform() < self.epsilon:   # greedy
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:   # random
            action = np.random.randint(0, self.n_actions)
        Q_out = actions_value.detach().cpu().numpy()[0]
        return action, Q_out

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        s_batch = [self.memory["s"][index] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        sn_batch = [self.memory["sn"][index] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]

        a_ts = torch.LongTensor(np.array(a_batch)).to(device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(device)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(device)

        q_eval = self.qnet_eval(s_batch, device).gather(1, a_ts.unsqueeze(1))
        q_next = self.qnet_target(sn_batch, device).detach()
        
        if USE_DOUBLE_DQN:
            a_eval = self.qnet_eval(sn_batch, device).detach().argmax(1).view(-1,1)
            q_target = r_ts.view(self.batch_size, 1) + end_ts.view(self.batch_size, 1) \
                    * self.gamma * torch.gather(q_next, dim=1, index=a_eval).view(self.batch_size, 1)
        else:
            q_target = r_ts.view(self.batch_size, 1) + end_ts.view(self.batch_size, 1) \
                    * self.gamma * q_next.max(1)[0].view(self.batch_size, 1) 

        self.loss = F.smooth_l1_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.loss.detach().cpu().numpy()

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s": [],"a": [],"r": [], "sn": [],"end":[]}

    def store_transition(self, s, a, r, sn, end):
        if not hasattr(self, 'memory_counter') or not hasattr(self, 'memory'):
            self.init_memory()
            
        if self.memory_counter <= self.memory_size:
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            self.memory["end"][index] = end

        self.memory_counter += 1