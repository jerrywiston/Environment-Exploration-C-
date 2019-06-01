import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG():
    def __init__(
        self,
        actor_net,
        critic_net,
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
        self._build_net(actor_net, critic_net)

    def _build_net(self, anet, cnet):
        self.actor_eval = anet().to(device)
        self.actor_target = anet().to(device)
        self.actor_target.eval()
        self.actror_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)

        self.critic_eval = cnet().to(device)
        self.critic_target = cnet().to(device)
        self.critic_target.eval()
        self.critic_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            action = self.actor_eval.forward(np.expand_dims(s,0), device).cpu().numpy()[0]
        else: 
            action = np.random.randint(-1, 1, size=(1, self.n_actions))
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

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

        a_ts = torch.FloatTensor(np.array(a_batch)).to(device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(device)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(device)

        # Critic Loss
        q_eval = self.critic_eval.forward(s_batch, a_ts, device)
        a_next = self.actor_target.forward(s_batch, device).detach()
        q_next = self.critic_net.forward(sn_batch, a_next, device).detach()
        q_target = r_ts.view(self.batch_size, 1) + end_ts.view(self.batch_size, 1) * self.gamma * q_next.max(1)[0]
        
        self.critic_loss = F.smooth_l1_loss(q_eval, q_target)
        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor Loss
        self.actor_loss = - q_eval.mean()
        self.actror_optim.zero_grad()
        self.actor_loss.backward()
        self.actror_optim.step()

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.actor_loss.detach().cpu().numpy(), self.critic_loss.detach().cpu().numpy()

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