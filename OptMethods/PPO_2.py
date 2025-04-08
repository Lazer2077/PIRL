import argparse
import matplotlib as plt
import numpy as np
import pandas as pd
from collections import namedtuple
from .lib.ReplayBuffer import Replay_buffer

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

manual_seed = 526963494564900
torch.manual_seed(manual_seed)


class ActorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.mu_head = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.mu_head.weight)

    def forward(self,x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        mu = self.mu_head(x)
        return mu
    
class CriticNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.v_head = nn.Linear(hidden_size, output_size)

        self.init_weights()
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.v_head.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        state_value = self.v_head(x)
        return state_value
    

class PPO():
    def __init__(self, state_dim, action_dim, ScalingDict, device, args):
        super(PPO, self).__init__()

        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.ppo_epoch = args.ppo_epoch
        self.replay_buffer_storage = args.capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ScalingDist = ScalingDict
        self.device = device


        self.training_step = 0
        self.actor_net = ActorNet(state_dim, action_dim, args.hidden_size).to(device)
        self.critic_net = CriticNet(state_dim, action_dim, args.hidden_size).to(device)
        self.replay_buffer = []
        self.counter = 0
        self.std = torch.diag(torch.full(size=(1,), fill_value = 0.5)).to(device)

        # self.Reply_buffer = 

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr = args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr = args.critic_lr)

    def select_action(self, state):
        state = state.float().unsqueeze(0).to(self.device)
        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu = self.actor_net(state)
        dist = Normal(mu, self.std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2.0, 2.0)
        return action.item(), action_log_prob.item()
    
    def get_value(self, state):
        state = torch.from_numpy(state).float.unsqueeze(0).to(self.device)
        with torch.no_grad():
            state_value = self.critic_net(state)
        return state_value.item()
    
    def store(self, transition):
        self.replay_buffer.append(transition)
        self.counter += 1
        return self.counter % self.replay_buffer_storage == 0
    
    def update(self, batch_size, Info:dict):
        self.training_step += 1

        states = torch.tensor([t.s for t in self.replay_buffer], dtype = torch.float, device=self.device)
        actions = torch.tensor([t.a for t in self.replay_buffer], dtype = torch.float, device=self.device).view(-1,1)
        rewards = torch.tensor([t.r for t in self.replay_buffer], dtype = torch.float, device= self.device).view(-1, 1)
        next_states = torch.tensor([t.s_ for t in self.replay_buffer], dtype = torch.float, device = self.device)

        old_action_log_probs = torch.tensor([t.a_log_p for t in self.replay_buffer], dtype = torch.float, device = self.device).view(-1,1)

        with torch.no_grad():
            values = self.critic_net(states).squeeze()

        rewards = (rewards-rewards.mean()) / (rewards.std() + 1e-5)

        with torch.no_grad():
            target_v = rewards + self.gamma*self.critic_net(next_states)

        advantages = (target_v-self.critic_net(states)).to(self.device).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.replay_buffer_storage)), batch_size, False):
                mu = self.actor_net(states[index])
                dist = Normal(mu, self.std)
                action_log_probs = dist.log_prob(actions[index])
                ratio = torch.exp(action_log_probs-old_action_log_probs[index])

                surr1 = ratio * advantages[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[index]

                action_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(states[index]), target_v[index])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        del self.replay_buffer[:]
