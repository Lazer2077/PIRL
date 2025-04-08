
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from torch.optim import Adam
from .lib.ReplayBuffer import Replay_buffer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

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

        self.Q_net1 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_net2 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_target_net1 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_target_net2 = Q(state_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)

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

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy_net.sample(state)
        else:
            _, _, action = self.policy_net.sample(state)
        return action.detach().cpu().numpy()[0]

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
        x, y, u, r, d = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(x).to(self.device)
        action_batch = torch.LongTensor(u).to(self.device) if self.is_discrete else torch.FloatTensor(u).to(self.device).reshape(-1, self.action_dim)
        next_state_batch = torch.FloatTensor(y).to(self.device)
        reward_batch = torch.FloatTensor(r).reshape(-1, 1).to(self.device)
        done_batch = torch.FloatTensor(1 - np.array(d)).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi, _= self.policy_net.sample(next_state_batch)
            q1_next = self.Q_target_net1(next_state_batch, next_action)
            q2_next = self.Q_target_net2(next_state_batch, next_action)
            # next_log_pi = next_log_pi.sum(dim=1, keepdim=True)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.reshape(-1, 1)
            next_q_value = reward_batch + done_batch * self.gamma * min_q_next

        q1_pred = self.Q_net1(state_batch, action_batch)
        q2_pred = self.Q_net2(state_batch, action_batch)
        q1_loss = F.mse_loss(q1_pred, next_q_value.detach())
        q2_loss = F.mse_loss(q2_pred, next_q_value.detach())

        sample_action, log_prob, _ = self.policy_net.sample(state_batch)
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