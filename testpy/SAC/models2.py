import numpy as np
import matplotlib.pyplot as plt
import math
import random
import models

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Normal

# Mapless Navigation Soft Actor-Critic
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(60, 512)
        self.fc2_s = nn.Linear(514, 256)
        self.fc2_a = nn.Linear(2, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, s, a, device):
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        goal = torch.FloatTensor(np.array([x['goal'] for x in s])).to(device)
        h_fc1 = F.relu(self.fc1(sensor))
        h_fc1 = torch.cat((h_fc1, goal), 1)
        h_fc2 = F.relu(self.fc2_s(h_fc1) + self.fc2_a(a))
        q_out = self.fc3(h_fc2)
        return q_out

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(60, 512)
        self.fc2 = nn.Linear(514, 256)
        self.fc3_mean = nn.Linear(256, 2)
        self.fc3_logstd = nn.Linear(256, 2)

    def forward(self, s, device):
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        goal = torch.FloatTensor(np.array([x['goal'] for x in s])).to(device)
        h_fc1 = F.relu(self.fc1(sensor))
        h_fc1 = torch.cat((h_fc1, goal), 1)
        h_fc2 = F.relu(self.fc2(h_fc1))
        a_mean = self.fc3_mean(h_fc2)
        a_logstd = self.fc3_logstd(h_fc2)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return a_mean, a_logstd
    
    def sample(self, s, device):
        a_mean, a_logstd = self.forward(s, device)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)


###########################################################################
def conv2d_size_out(size, kernel_size = 5, stride = 2):
    return (size - (kernel_size - 1) - 1) // stride  + 1
# SAC 
class PolicyNetExp(nn.Module):
    def __init__(self, feature_size=(64,64)):
        super(PolicyNetExp, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[1])))
        linear_input_size = convw * convh * 128
        
        self.fc4 = nn.Linear(linear_input_size, 256)
        self.fc5_mean = nn.Linear(256, 2)
        self.fc5_logstd = nn.Linear(256, 2)

    def forward(self, s, device):
        m = torch.FloatTensor(np.array([x['map'] for x in s])).to(device).permute(0,3,1,2)
        h_conv1 = F.relu(self.bn1(self.conv1(m)))
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv1)))
        h_conv3 = F.relu(self.bn3(self.conv3(h_conv2)))
        h_flat3 = h_conv3.view(h_conv3.size(0), -1)
        h_fc4 = F.relu(self.fc4(h_flat3))
        a_mean = self.fc5_mean(h_fc4)
        a_logstd = self.fc5_logstd(h_fc4)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return a_mean, a_logstd
    
    def sample(self, s, device):
        a_mean, a_logstd = self.forward(s, device)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)

class QNetExp(nn.Module):
    def __init__(self, feature_size=(64,64)):
        super(QNetExp, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(256)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[1])))
        linear_input_size = convw * convh * 256

        self.fc4 = nn.Linear(linear_input_size, 512)
        self.fc4_a = nn.Linear(2, 512)
        self.fc5 = nn.Linear(512, 2)
    
    def forward(self, s, a, device):
        m = torch.FloatTensor(np.array([x['map'] for x in s])).to(device).permute(0,3,1,2)
        h_conv1 = F.relu(self.bn1(self.conv1(m)))
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv1)))
        h_conv3 = F.relu(self.bn3(self.conv3(h_conv2)))
        h_flat3 = h_conv3.view(h_conv3.size(0), -1)
        h_fc4 = F.relu(self.fc4(h_flat3) + self.fc4_a(a))
        q_out = self.fc5(h_fc4)
        return q_out

###########################################################################
class PolicyNetExp2(nn.Module):
    def __init__(self, feature_size=(64,64)):
        super(PolicyNetExp2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[1])))
        linear_input_size = convw * convh * 128
        
        self.fc4 = nn.Linear(linear_input_size, 256)
        self.fc4_s = nn.Linear(60, 256)
        self.fc5_mean = nn.Linear(256, 2)
        self.fc5_logstd = nn.Linear(256, 2)

    def forward(self, s, device):
        m = torch.FloatTensor(np.array([x['map'] for x in s])).to(device).permute(0,3,1,2)
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        h_conv1 = F.relu(self.bn1(self.conv1(m)))
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv1)))
        h_conv3 = F.relu(self.bn3(self.conv3(h_conv2)))
        h_flat3 = h_conv3.view(h_conv3.size(0), -1)
        h_fc4 = F.relu(self.fc4(h_flat3) + self.fc4_s(sensor))
        a_mean = self.fc5_mean(h_fc4)
        a_logstd = self.fc5_logstd(h_fc4)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return a_mean, a_logstd
    
    def sample(self, s, device):
        a_mean, a_logstd = self.forward(s, device)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)

class QNetExp2(nn.Module):
    def __init__(self, feature_size=(64,64)):
        super(QNetExp2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(256)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[1])))
        linear_input_size = convw * convh * 256

        self.fc4 = nn.Linear(linear_input_size, 512)
        self.fc4_s = nn.Linear(60, 512)
        self.fc4_a = nn.Linear(2, 512)
        self.fc5 = nn.Linear(512, 2)
    
    def forward(self, s, a, device):
        m = torch.FloatTensor(np.array([x['map'] for x in s])).to(device).permute(0,3,1,2)
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        h_conv1 = F.relu(self.bn1(self.conv1(m)))
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv1)))
        h_conv3 = F.relu(self.bn3(self.conv3(h_conv2)))
        h_flat3 = h_conv3.view(h_conv3.size(0), -1)
        h_fc4 = F.relu(self.fc4(h_flat3) + self.fc4_a(a) + self.fc4_s(sensor))
        q_out = self.fc5(h_fc4)
        return q_out