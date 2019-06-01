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

################################### Exploration Model ###################################
# Map-based Navigation Q Network
class QNetExpMap(nn.Module):
    def __init__(self, feature_size=(64,64)):
        super(QNetExpMap, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[1])))
        linear_input_size = convw * convh * 128
        self.fc4 = nn.Linear(linear_input_size, 3)
    
    def forward(self, s, device):
        m = torch.FloatTensor(np.array([x['map'] for x in s])).to(device).permute(0,3,1,2)
        h_conv1 = F.relu(self.bn1(self.conv1(m)))
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv1)))
        h_conv3 = F.relu(self.bn3(self.conv3(h_conv2)))
        h_flat3 = h_conv3.view(h_conv3.size(0), -1)
        q_out = self.fc4(h_flat3)
        return q_out

################################### Navigation Model ###################################
# Map-based Navigation Q Network
class QNetNavMap(nn.Module):
    def __init__(self, feature_size=(64,64)):
        super(QNetNavMap, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(feature_size[1])))
        linear_input_size = convw * convh * 128
        self.fc4 = nn.Linear(linear_input_size, 64)
        self.fc5 = nn.Linear(66, 3)
    
    def forward(self, s, device):
        m = torch.FloatTensor(np.array([x['map'] for x in s])).to(device).permute(0,3,1,2)
        goal = torch.FloatTensor(np.array([x['goal'] for x in s])).to(device)
        h_conv1 = F.relu(self.bn1(self.conv1(m)))
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv1)))
        h_conv3 = F.relu(self.bn3(self.conv3(h_conv2)))
        h_flat3 = h_conv3.view(h_conv3.size(0), -1)
        h_fc4 = F.relu(self.fc4(h_flat3))
        h_cat4 = torch.cat((h_fc4, goal), 1)
        q_out = self.fc5(h_cat4)
        return q_out

# Mapless Navigation Q Network
class QNetNavMapless(nn.Module):
    def __init__(self):
        super(QNetNavMapless, self).__init__()
        self.fc1 = nn.Linear(60, 64)
        self.fc2 = nn.Linear(66, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, s, device):
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        goal = torch.FloatTensor(np.array([x['goal'] for x in s])).to(device)
        h_fc1 = F.relu(self.fc1(sensor))
        h_concat = torch.cat((h_fc1, goal), 1)
        h_fc2 = F.relu(self.fc2(h_concat))
        q_out = self.fc3(h_fc2)
        return q_out

# Mapless Navigation DDPG
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(60, 64)
        self.fc2 = nn.Linear(66, 32)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, s, device):
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        goal = torch.FloatTensor(np.array([x['goal'] for x in s])).to(device)
        h_fc1 = F.relu(self.fc1(sensor))
        h_concat = torch.cat((h_fc1, goal), 1)
        h_fc2 = F.relu(self.fc2(h_concat))
        a_out = F.tanh(self.fc3(h_fc2))
        return a_out

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(60, 64)
        self.fc2 = nn.Linear(66, 32)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, s, a, device):
        sensor = torch.FloatTensor(np.array([x['sensor'] for x in s])).to(device)
        goal = torch.FloatTensor(np.array([x['goal'] for x in s])).to(device)
        h_fc1 = F.relu(self.fc1(sensor))
        h_concat = torch.cat((h_fc1, goal, a), 1)
        h_fc2 = F.relu(self.fc2(h_concat))
        q_out = self.fc3(h_fc2)
        return q_out