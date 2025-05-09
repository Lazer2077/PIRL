
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from torch.optim import Adam

from operator import itemgetter
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256, is_discrete=False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.xmean = xMean
        self.xstd = xStd

    def forward(self, x):
        x = (x - self.xmean) / self.xstd
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.is_discrete:
            logits = self.mu_head(x)
            probs = F.softmax(logits, dim=-1)
            return probs, logits
        else:
            mu = self.mu_head(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            return mu, log_std


# V_k+1 = min(V_k + l_N(x) )