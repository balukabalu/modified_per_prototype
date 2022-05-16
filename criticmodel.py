import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, xs):

        x, a = xs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        """print(x.shape)
        print(a.shape)
        print("shapek")
        print(np.concatenate((x.detach().numpy(),a.detach().numpy()), axis=1).shape)"""
        x = F.relu(self.fc3(torch.from_numpy(np.concatenate((x.detach().numpy(),a.detach().numpy()), axis=1))))


        x = F.tanh(self.fc4(x))
        return x

