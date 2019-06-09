import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG():
    def __init__(
        self,
        actor_net,
        critic_net,
        n_actions,
        action_range = 1,
        learning_rate = [1e-4, 2e-4],
        reward_decay = 0.95,
        replace_target_iter = 300,
        memory_size = 500,
        batch_size = 64,
        tau = 0.01,
        var = 3,
        var_decay = 0.9995,
        criterion = nn.MSELoss()
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.action_range = action_range
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.criterion = criterion
        self.var = var
        self.var_decay = var_decay
        self.learn_step_counter = 0
        self._build_net(actor_net, critic_net)

    def _build_net(self, anet, cnet):
        self.actor_eval = anet().to(device)
        self.actor_target = anet().to(device)
        self.actor_target.eval()
        self.actor_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr[0])

        self.critic_eval = cnet().to(device)
        self.critic_target = cnet().to(device)
        self.critic_target.eval()
        self.critic_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr[1])

    def choose_action(self, s):
        action = self.actor_eval.forward(np.expand_dims(s,0), device).cpu().detach().numpy()[0]
        action *= self.action_range
        action = np.clip(np.random.normal(action, self.var), -self.action_range, self.action_range)
        return action

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            # Update Actor Parameters
            for targetParam, evalParam in zip(self.actor_target.parameters(), self.actor_eval.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)
            # Update Critic Parameters
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
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
        r_ts = torch.FloatTensor(np.array(r_batch)).to(device).view(self.batch_size, 1)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(device).view(self.batch_size, 1)
        
        # Critic Update
        a_next = self.actor_target(sn_batch, device).detach()
        q_next = self.critic_target(sn_batch, a_next, device).detach()
        q_target = r_ts + end_ts * self.gamma * q_next
        
        self.critic_optim.zero_grad()
        q_eval = self.critic_eval(s_batch, a_ts, device)

        self.critic_loss = self.criterion(q_eval, q_target)
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor Update
        self.actor_optim.zero_grad()
        a_current = self.actor_eval(s_batch, device)
        q_current = self.critic_eval(s_batch, a_current, device)
        self.actor_loss = - q_current.mean()
        
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        if self.var > 0.5:
            self.var *= self.var_decay

        # increasing epsilon
        self.learn_step_counter += 1
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s":[], "a":[], "r":[], "sn":[], "end":[]}

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