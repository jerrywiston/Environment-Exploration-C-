import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC():
    def __init__(
        self,
        model,
        n_actions,
        action_range = 1,
        learning_rate = [1e-4, 2e-4],
        reward_decay = 0.95,
        replace_target_iter = 300,
        memory_size = 500,
        batch_size = 64,
        tau = 0.01,
        alpha = 0.5,
        auto_entropy_tuning = True,
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
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.criterion = criterion
        self.learn_step_counter = 0
        self._build_net(model['anet'], model['qnet'])

    def _build_net(self, anet, qnet):
        self.critic = qnet().to(device)
        self.critic_target = qnet().to(device)
        self.critic_target.eval()
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])

        self.actor = anet().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        
        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(self.n_actions).to(device)).item() * 10
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0002)
    
    def save_load_model(self, op, path):
        if op == "save":
            torch.save(self.critic.state_dict(), path["critic"])
            torch.save(self.actor.state_dict(), path["actor"])
        elif op == "load":
            self.critic.load_state_dict(torch.load(path["critic"]))
            self.critic_target.load_state_dict(torch.load(path["critic"]))
            self.actor.load_state_dict(torch.load(path["actor"]))

    def choose_action(self, s, eval=False):
        if eval == False:
            action, _, _ = self.actor.sample(np.expand_dims(s,0), device)
        else:
            _, _, action = self.actor.sample(np.expand_dims(s,0), device)
        action = action.cpu().detach().numpy()[0]
        action *= self.action_range
        return action

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

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
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
        
        with torch.no_grad():
            a_next, logpi_next, _ = self.actor.sample(sn_batch, device)
            q_next_target = self.critic(sn_batch, a_next, device) - self.alpha * logpi_next
            q_target = r_ts + end_ts * self.gamma * q_next_target
        
        q_eval = self.critic(s_batch, a_ts, device)
        self.critic_loss = self.criterion(q_eval, q_target)

        a_curr, logpi_curr, _ = self.actor.sample(s_batch, device)
        q_current = self.critic(s_batch, a_curr, device)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        self.soft_update()
        
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())
        
        # increasing epsilon
        self.learn_step_counter += 1
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())

    