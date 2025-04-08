
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import os
import gym

from .lib.ReplayBuffer import Replay_buffer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256, is_discrete=False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.xmean = xMean
        self.xstd = xStd
        self.is_discrete = is_discrete
        self.apply(weights_init_)

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

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256, is_discrete=False):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.xmean = xMean
        self.xstd = xStd
        self.is_discrete = is_discrete
        self.apply(weights_init_)

    def forward(self, s, a):
        s = (s - self.xmean[:self.state_dim]) / self.xstd[:self.state_dim]
        if self.is_discrete:
            a = F.one_hot(a.to(torch.int64), num_classes=self.action_dim).float()
        x = torch.cat((s, a), -1)
        x = (x - self.xmean) / self.xstd
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, state_dim, action_dim, ScalingDict, device, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = device
        self.alpha = torch.FloatTensor([args.alpha]).to(device)
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = args.is_discrete

        self.replay_buffer = Replay_buffer()
        self.policy_net = Actor(state_dim, action_dim, ScalingDict.get('xMean', torch.zeros(state_dim)).to(device),
                                ScalingDict.get('xStd', torch.ones(state_dim)).to(device),
                                args.num_hidden_units_per_layer, self.is_discrete).to(device)

        xumean = torch.cat([ScalingDict.get('xMean', torch.zeros(state_dim)).to(device),
                            ScalingDict.get('uMean', torch.zeros(action_dim)).to(device)])
        xustd = torch.cat([ScalingDict.get('xStd', torch.ones(state_dim)).to(device),
                           ScalingDict.get('uStd', torch.ones(action_dim)).to(device)])

        self.Q_net1 = Q(state_dim, action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_net2 = Q(state_dim, action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_target_net1 = Q(state_dim, action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_target_net2 = Q(state_dim, action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        for target_param, param in zip(self.Q_target_net1.parameters(), self.Q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.Q_target_net2.parameters(), self.Q_net2.parameters()):
            target_param.data.copy_(param.data)

        if self.automatic_entropy_tuning:
            self.target_entropy = -np.log(1.0 / self.action_dim) if self.is_discrete else -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.learning_rate)

        self.LossDict = {}
        self.AuxDict = {}
        self.LogDict = {}

    def select_action(self, state, IS_EVALUATION=False):
        state = torch.FloatTensor(state).to(self.device)
        if self.is_discrete:
            probs, _ = self.policy_net(state)
            dist = Categorical(probs)
            action = torch.argmax(probs, dim=-1) if IS_EVALUATION else dist.sample()
            return action.cpu().numpy()
        else:
            action, _, _, _, _ = self.evaluate(state)
            return action.reshape(-1, self.action_dim).detach().cpu()

    def evaluate(self, state):
        if self.is_discrete:
            probs, logits = self.policy_net(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, probs, logits, action
        else:
            mu, log_std = self.policy_net(state)
            std = log_std.exp()
            dist = Normal(mu, std)
            x_t = dist.rsample()
            action0 = torch.tanh(x_t)
            action = action0
            log_prob = dist.log_prob(x_t) - torch.log(1 - action0.pow(2) + EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob, mu, log_std, torch.tanh(mu)

    def update(self, batch_size, Info):
        x, y, u, r, d = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(x).to(self.device)
        action_batch = torch.LongTensor(u).to(self.device) if self.is_discrete else torch.FloatTensor(u).to(self.device).reshape(-1, self.action_dim)
        next_state_batch = torch.FloatTensor(y).to(self.device)
        reward_batch = torch.FloatTensor(r).reshape(-1, 1).to(self.device)
        done_batch = torch.FloatTensor(1 - np.array(d)).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi, _, _, _ = self.evaluate(next_state_batch)
            q1_next = self.Q_target_net1(next_state_batch, next_action)
            q2_next = self.Q_target_net2(next_state_batch, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.reshape(-1, 1)
            next_q_value = reward_batch + done_batch * self.gamma * min_q_next

        q1_pred = self.Q_net1(state_batch, action_batch)
        q2_pred = self.Q_net2(state_batch, action_batch)
        q1_loss = F.mse_loss(q1_pred, next_q_value.detach())
        q2_loss = F.mse_loss(q2_pred, next_q_value.detach())

        sample_action, log_prob, _, _, _ = self.evaluate(state_batch)
        log_prob = log_prob.view(-1, 1).clone() 
        q1_pi = self.Q_net1(state_batch, sample_action)
        q2_pi = self.Q_net2(state_batch, sample_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        (q1_loss+q2_loss).backward()    
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()



        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.Q_target_net1.parameters(), self.Q_net1.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
        for target_param, param in zip(self.Q_target_net2.parameters(), self.Q_net2.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
            
    def replayEpisodeValue(self, batch):
        observationBatch = torch.FloatTensor(batch[0])
        actionBatch = torch.FloatTensor(batch[2]).reshape(-1,self.action_dim)
                
        ValueDict = {'reward': batch[3],
                    # 'Q1': self.Q_net1(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                    # 'Q2': self.Q_net2(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                    # 'Q1_t': self.Q_target_net1(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                    # 'Q2_t': self.Q_target_net2(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                     }
        return ValueDict
