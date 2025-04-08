
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from .lib.ReplayBuffer import Replay_buffer

class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, max_action, xMean, xStd):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(observation_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action

        self.xmean = xMean
        self.xstd = xStd

        self.model_name = 'Actor'

    def forward(self, x):
        if not x.is_cuda:    
            x = x.cuda()
            
        x = (x-self.xmean)/self.xstd

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))

        # x = self.max_action * torch.tanh(self.model(x))

        return x


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, xMean, xStd):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(observation_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

        self.xmean = xMean
        self.xstd = xStd

        self.model_name = 'Critic'

    def forward(self, x, u):
        x = torch.cat([x, u],1)

        if not x.is_cuda:    
            x = x.cuda()
        
        x = (x-self.xmean)/self.xstd

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        # x = self.model(torch.cat([x, u], 1))

        return x
    
class DDPG(object):
    def __init__(self, observation_dim, action_dim, ScalingDict, device, args):

        self.action_max = ScalingDict['actionMax']
        self.action_min = ScalingDict['actionMin']
        xuMean = torch.cat([ScalingDict['xMean'].to(device), ScalingDict['uMean'].to(device)])
        xuStd = torch.cat([ScalingDict['xStd'].to(device), ScalingDict['uStd'].to(device)])

        self.actor = Actor(observation_dim, action_dim, self.action_max, ScalingDict['xMean'].to(device), ScalingDict['xStd'].to(device)).to(device)
        self.actor_target = Actor(observation_dim, action_dim, self.action_max, ScalingDict['xMean'].to(device), ScalingDict['xStd'].to(device)).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_learning_rate)

        self.critic = Critic(observation_dim, action_dim, xuMean, xuStd).to(device)
        self.critic_target = Critic(observation_dim, action_dim, xuMean, xuStd).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_learning_rate)
        self.replay_buffer = Replay_buffer()
        #self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.device = device

        # parser args
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.dynamic_noise = args.dynamic_noise

        self.LossDict = {'Critic': 0,
                    'Actor': 0}
        
        self.AuxDict = {}

        self.LogDict = {'Critic': {'model': self.critic, 'epoch': self.num_critic_update_iteration, 'loss': 0},
                        'Actor': {'model': self.actor, 'epoch': self.num_actor_update_iteration, 'loss': 0}}

    def select_action(self, observation, IS_EVALUATION=False):
        observation = observation.reshape(-1,self.observation_dim).to(self.device)
        #observation = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        actionRaw = self.actor(observation).detach().cpu() 
        if not IS_EVALUATION:
            action = torch.clip(actionRaw + torch.normal(0, self.exploration_noise, size=(1,1)), min=self.action_min, max=self.action_max).detach()
        else:
            action = actionRaw
        return action.reshape(-1,self.action_dim).detach().cpu()
    
    def update(self, batch_size, Info):
        done = Info['done']

        # only update at the end of each episode
        if not done:
            return

        if Info['iUpdate'] == 0:
            if self.dynamic_noise:
                self.exploration_noise = self.exploration_noise*0.9995

        # Sample replay buffer
        x, y, u, r, d = self.replay_buffer.sample(batch_size)
        observation = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device).reshape(-1,self.action_dim)
        next_observation = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(1-np.array(d)).to(self.device).reshape(-1,1)
        reward = torch.FloatTensor(r).to(self.device).reshape(-1,1)

        # Compute the target Q value
        target_Q = self.critic_target(next_observation, self.actor_target(next_observation))
        target_Q = reward + (done * self.gamma * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(observation, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        #self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if torch.abs(list(self.critic.parameters())[0].grad).mean() < 1e-7:
            pass
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(observation, self.actor(observation)).mean()
        #self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if torch.abs(list(self.actor.parameters())[0].grad).mean() < 1e-7:
            pass
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

        # return critic_loss, actor_loss
        #return F.mse_loss(current_Q, target_Q), -self.critic(observation, self.actor(observation)).mean()
        self.LossDict = {'Critic': critic_loss.cpu().item(),
                    'Actor': actor_loss.cpu().item()}

        self.LogDict = {'Critic': {'model': self.critic, 'epoch': self.num_critic_update_iteration, 'loss': critic_loss.cpu().item()},
                        'Actor': {'model': self.actor, 'epoch': self.num_actor_update_iteration, 'loss': actor_loss.cpu().item()}}
        
    # def save(self):
    #     torch.save(self.actor.state_dict(), directory + 'actor.pth')
    #     torch.save(self.critic.state_dict(), directory + 'critic.pth')
    #     # print("====================================")
    #     # print("Model has been saved...")
    #     # print("====================================")

    # def load(self):
    #     self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
    #     self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
    #     print("====================================")
    #     print("model has been loaded...")
    #     print("====================================")

    def replayEpisodeValue(self, batch):

        observationBatch = torch.FloatTensor(batch[0])
        actionBatch = torch.FloatTensor(batch[2]).reshape(-1,self.action_dim)

        ValueDict = {'reward': batch[3],
                     'Critic': self.critic(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                     'Critic_t': self.critic_target(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten()
                     }

        return ValueDict