
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from torch.optim import Adam
from operator import itemgetter

class Replay_buffer():
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        # batch = random.sample(self.storage, batch_size)
        # x, y, u, r, d = map(np.stack, zip(*batch))
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d, xn ,rn = map(np.stack, zip(*itemgetter(*ind)(self.storage)))

        return x,y,u,r,d,xn,rn
    
    def getEpisodeBatch(self, steps):

        if len(self.storage) == 0:
            return
        
        if len(self.storage) == self.max_size:
            idx = self.ptr % self.max_size
            if idx-steps-1 < 0:
                idxList = np.concatenate((np.arange(self.max_size-(steps+1-idx), self.max_size),np.arange(0,idx)))
            else:
                idxList = np.arange(idx-steps-1,idx)
        else:
            idxList = np.arange(len(self.storage)-steps-1,len(self.storage))
        batch = list(zip(*itemgetter(*idxList)(self.storage)))
        # if observation is numpy
        if isinstance(batch[0][0], np.ndarray):
            observationBatch = np.array(batch[0])
            observationNextBatch = np.array(batch[1])
            actionBatch = np.array(batch[2])
            rewardBatch = np.array(batch[3])
        else:
            observationBatch = torch.stack(batch[0]).cpu().data.numpy()
            observationNextBatch = torch.stack(batch[1]).cpu().data.numpy()
            actionBatch = torch.stack(batch[2]).cpu().data.numpy().flatten()
            rewardBatch = torch.stack(batch[3]).cpu().data.numpy().flatten()
        doneBatch = np.array(batch[4])
        xNBatch = np.array(batch[5])
        reward_NBatch = np.array(batch[6])
        return (observationBatch, observationNextBatch, actionBatch, rewardBatch, doneBatch, xNBatch, reward_NBatch)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6
torch.autograd.set_detect_anomaly(False)
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256):
        super(Actor, self).__init__()
        num_feature = 32
        # self.fc1 = nn.Linear(state_dim + num_feature, hidden_dim)
        self.fc1 = nn.Linear(state_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        #self.max_action = max_action
        self.xmean = xMean
        self.xstd = xStd

        self.apply(weights_init_)

    def forward(self, x, feature_vector=None):
        if not x.is_cuda:    
            x = x.cuda()
        x = (x-self.xmean)/self.xstd        
        # x = torch.cat((x, feature_vector), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = self.log_std_head(x)
        log_std_head = torch.clamp(log_std_head, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std_head

class Q(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256):
        super(Q, self).__init__()
        self.xN_dim = 2
        self.fc1 = nn.Linear(state_dim + action_dim + self.xN_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.xmean = xMean
        self.xstd = xStd
        self.apply(weights_init_)

    def forward(self, s, a, xN):
        if not s.is_cuda:    
            s = s.cuda()
        if not a.is_cuda:    
            a = a.cuda()
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        
        x = torch.cat((s, a), -1)
        x = (x-self.xmean)/self.xstd
        x = torch.cat((x, xN), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x[:,0], x[:,1]
    

class SAC1:
    def __init__(self, state_dim, action_space, ScalingDict, device, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = device
        self.alpha = torch.FloatTensor([args.alpha]).to(device)
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]
        self.is_discrete = args.is_discrete
        self.automatic_entropy_tuning = args.automatic_entropy_tuning   
        self.replay_buffer = Replay_buffer()


        xumean = torch.cat([ScalingDict.get('xMean', torch.zeros(state_dim)).to(device),
                            ScalingDict.get('uMean', torch.zeros(self.action_dim)).to(device)])
        xustd = torch.cat([ScalingDict.get('xStd', torch.ones(state_dim)).to(device),
                           ScalingDict.get('uStd', torch.ones(self.action_dim)).to(device)])

        self.Q_net1 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer).to(device)
        self.Q_net2 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer).to(device)
        self.Q_target_net1 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer).to(device)
        self.Q_target_net2 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer).to(device)

        self.Q1_optimizer = Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        for target_param, param in zip(self.Q_target_net1.parameters(), self.Q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.Q_target_net2.parameters(), self.Q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.policy_type = args.policy_type
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.learning_rate)

            self.policy_net = GaussianPolicy(state_dim, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        else:
            self.policy_net = DeterministicPolicy(state_dim, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=args.learning_rate)

    def select_action(self, state, evaluate=False,ref=None):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy_net.sample(state)
        else:
            _, _, action = self.policy_net.sample(state)
        return action.detach().cpu()[0]

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
            y_t = torch.tanh(x_t)
            action = y_t
            log_det = torch.log(1 - y_t.pow(2) + EPSILON)
            log_det = log_det.sum(dim=-1, keepdim=True)
            log_prob = dist.log_prob(x_t) - log_det
            return action, log_prob, mu, log_std, torch.tanh(mu)

    def update(self, batch_size, Info=None):
        x, y, u, r, d, xN, rN = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(x).to(self.device)
        action_batch = torch.LongTensor(u).to(self.device) if self.is_discrete else torch.FloatTensor(u).to(self.device).reshape(-1, self.action_dim)
        next_state_batch = torch.FloatTensor(y).to(self.device)
        reward_batch = torch.FloatTensor(r).reshape(-1, 1).to(self.device)
        undone_batch = torch.FloatTensor(d).reshape(-1, 1).to(self.device)
        done_batch = torch.FloatTensor(1 - np.array(d)).reshape(-1, 1).to(self.device)
        
        terminal_state_batch = torch.FloatTensor(xN).reshape(-1, 2).to(self.device)
        rN_batch = torch.FloatTensor(rN).reshape(-1, 1).to(self.device)
        with torch.no_grad():
            next_action, next_log_pi, _= self.policy_net.sample(next_state_batch)
            q1_next, RN1 = self.Q_target_net1(next_state_batch, next_action, terminal_state_batch)
            q2_next, RN2 = self.Q_target_net2(next_state_batch, next_action, terminal_state_batch)
            # next_log_pi = next_log_pi.sum(dim=1, keepdim=True)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.reshape(-1, 1)
            next_q_value  = reward_batch + done_batch * self.gamma * min_q_next
            

        qf1, RN1  = self.Q_net1(state_batch, action_batch, terminal_state_batch)
        qf2, RN2 = self.Q_net2(state_batch, action_batch, terminal_state_batch)
        
        q1_loss = F.mse_loss(qf1, next_q_value)
        q2_loss = F.mse_loss(qf2, next_q_value)
        
        RN1_loss = F.mse_loss(RN1, rN_batch)
        RN2_loss = F.mse_loss(RN2, rN_batch)
        
        
        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        (q1_loss+q2_loss+RN1_loss+RN2_loss).backward()   
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()
        
        pi, log_prob, _ = self.policy_net.sample(state_batch)
        q1_pi, RN1   = self.Q_net1(state_batch, pi, terminal_state_batch)
        q2_pi, RN2   = self.Q_net2(state_batch, pi, terminal_state_batch)

        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_prob - min_q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

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
        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha.item()
    
    
    def save(self, modelPath):
        import os
        torch.save(self.policy_net.state_dict(), os.path.join(modelPath, 'policy_net.pth'))
        torch.save(self.Q_net1.state_dict(), os.path.join(modelPath, 'Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), os.path.join(modelPath, 'Q_net2.pth'))
        torch.save(self.Q_target_net1.state_dict(), os.path.join(modelPath, 'Q_target_net1.pth'))
        torch.save(self.Q_target_net2.state_dict(), os.path.join(modelPath, 'Q_target_net2.pth'))
        #torch.save(self.critic_target.state_dict(), os.path.join(savePath, 'critic_target.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")
    def load(self, modelPath):
        import os
        self.policy_net.load_state_dict(torch.load(os.path.join(modelPath, 'policy_net.pth')))
        self.Q_net1.load_state_dict(torch.load(os.path.join(modelPath, 'Q_net1.pth')))
        self.Q_net2.load_state_dict(torch.load(os.path.join(modelPath, 'Q_net2.pth')))
        self.Q_target_net1.load_state_dict(torch.load(os.path.join(modelPath, 'Q_target_net1.pth')))
        self.Q_target_net2.load_state_dict(torch.load(os.path.join(modelPath, 'Q_target_net2.pth')))
