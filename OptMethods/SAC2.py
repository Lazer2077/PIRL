
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from torch.optim import Adam
from operator import itemgetter
import torch
# import replay buffer
from .lib.ReplayBuffer import Replay_buffer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6
# torch.autograd.set_detect_anomaly(True)
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class StateEncoder(nn.Module):
    def __init__(self, state_dim, output_dim, xmean, xstd):
        super(StateEncoder, self).__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.xmean = xmean
        self.xstd = xstd
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, state):
        s = (state - self.xmean[:self.state_dim]) / self.xstd[:self.state_dim]
        return self.encoder(s)
    
    
class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=150, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)  # 简单注意力权重
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, memory):
        # memory: [B, T_enc, input_dim]
        if memory.ndim <3:
            memory = memory.reshape(-1, 1, memory.shape[-1])
        
        lstm_out, _ = self.lstm(memory)  # [B, T, H]
        # attention over time
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # [B, T, 1]
        context = (weights * lstm_out).sum(dim=1)  # [B, H]
        return self.fc(context)  # [B, 150]

class RefDecoder(nn.Module):
    def __init__(self, state_feature_dim, hidden_dim, ref_dim):
        super(RefDecoder, self).__init__()
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.ref_dim = ref_dim

        self.decoder = nn.Sequential(   
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ref_dim),
            nn.ReLU()
        )
    
    def forward(self, state_feature):
        return self.decoder(state_feature)

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

    def forward(self, state_feature):
        x = F.relu(self.linear1(state_feature))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state_feature):
        mean, log_std = self.forward(state_feature)
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

    def forward(self, state_feature):
        x = F.relu(self.linear1(state_feature))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state_feature):
        mean = self.forward(state_feature)
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
    def __init__(self, feature_dim, action_dim, xMean, xStd, hidden_dim=256, is_discrete=False):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.xmean = xMean
        self.xstd = xStd
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_std

class Q(nn.Module):
    def __init__(self, feature_dim, action_dim, xMean, xStd, hidden_dim=512):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(feature_dim+action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # self.fc_model = nn.Linear(hidden_dim, state_dim)
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.xmean = xMean
        self.xstd = xStd
        # self.is_discrete = is_discrete
        self.apply(weights_init_)

    def forward(self, feature, action):
        x = torch.cat([feature, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class SAC2:
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
        self.replay_buffer_list = [self.replay_buffer]

        xumean = torch.cat([ScalingDict.get('xMean', torch.zeros(state_dim)).to(device),
                            ScalingDict.get('uMean', torch.zeros(self.action_dim)).to(device)])
        xustd = torch.cat([ScalingDict.get('xStd', torch.ones(state_dim)).to(device),
                           ScalingDict.get('uStd', torch.ones(self.action_dim)).to(device)])

        
        ''' START 5/27/2025'''
        self.state_feature_dim = 128
        self.ref_dim = 150
        
        self.state_encoder = StateEncoder(state_dim, self.state_feature_dim, xumean, xustd).to(device) 
        self.ref_decoder = LSTMDecoder(self.state_feature_dim, self.state_feature_dim, self.ref_dim).to(device)
        decoder_lr = args.learning_rate 
        self.ref_decoder_optimizer = Adam(self.ref_decoder.parameters(), lr=decoder_lr)
        
        ''' END 5/27/2025'''
        
        self.Q_net1 = Q(self.state_feature_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_net2 = Q(self.state_feature_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        
        self.Q_net_list = [self.Q_net1, self.Q_net2]
        
        self.Q_target_net1 = Q(self.state_feature_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)
        self.Q_target_net2 = Q(self.state_feature_dim, self.action_dim, xumean, xustd, args.num_hidden_units_per_layer, self.is_discrete).to(device)

        self.Q_target_net_list = [self.Q_target_net1, self.Q_target_net2]

        self.Q1_optimizer = Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = Adam(self.Q_net2.parameters(), lr=args.learning_rate)
        
        self.Q_optimizer = [self.Q1_optimizer, self.Q2_optimizer]
        
        for target_Q_net, Q_net in zip(self.Q_target_net_list, self.Q_net_list):
            for target_param, param in zip(target_Q_net.parameters(), Q_net.parameters()):
                target_param.data.copy_(param.data)
        
        self.policy_type = args.policy_type
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.learning_rate)
            self.policy_net = GaussianPolicy(self.state_feature_dim, self.action_dim, args.hidden_size, action_space).to(self.device)
        else:
            self.policy_net = DeterministicPolicy(self.state_feature_dim, self.action_dim, args.hidden_size, action_space).to(self.device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=args.learning_rate)

    def select_action(self, state, evaluate=False,ref=None):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state_feature = self.state_encoder(state)
        if evaluate is False:
            action, _, _ = self.policy_net.sample(state_feature)
        else:
            _, _, action = self.policy_net.sample(state_feature)
        return action.detach().cpu()[0]

    def evaluate(self, state):
        state_feature = self.state_encoder(state)
        if self.is_discrete:
            probs, logits = self.policy_net(state_feature)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, probs, logits, action
        else:
            mu, log_std = self.policy_net(state_feature)
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
        x, y, u, r, d, ref = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(x).to(self.device)
        action_batch = torch.LongTensor(u).to(self.device) if self.is_discrete else torch.FloatTensor(u).to(self.device).reshape(-1, self.action_dim)
        next_state_batch = torch.FloatTensor(y).to(self.device)
        done_batch = torch.FloatTensor(d).reshape(-1, 1).to(self.device)    
        undone_batch = torch.FloatTensor(1 - np.array(d)).reshape(-1, 1).to(self.device)
        reward_batch = torch.FloatTensor(r).reshape(-1, 1).to(self.device)  
        with torch.no_grad():
            next_state_feature = self.state_encoder(next_state_batch)
            next_action, next_log_pi, _ = self.policy_net.sample(next_state_feature)
            q1_next = self.Q_net1(next_state_feature, next_action)
            q2_next = self.Q_net2(next_state_feature, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.reshape(-1, 1)
            next_q_value  = reward_batch + done_batch * self.gamma * min_q_next

        
        state_feature = self.state_encoder(state_batch)            
        qf1 = self.Q_net1(state_feature, action_batch)
        qf2 = self.Q_net2(state_feature, action_batch)
        q1_loss = F.mse_loss(qf1, next_q_value)
        q2_loss = F.mse_loss(qf2, next_q_value)
        decoded_ref = self.ref_decoder(next_state_feature)
        Ref = torch.FloatTensor(ref[:,:self.ref_dim]).to(self.device)
        recon_loss = F.mse_loss(decoded_ref, Ref)
        # back reconstruction loss
        self.ref_decoder_optimizer.zero_grad()
        recon_loss.backward()
        self.ref_decoder_optimizer.step()
        
        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        (q1_loss+q2_loss).backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()
        
        state_feature = self.state_encoder(state_batch) 
        pi, log_prob, _ = self.policy_net.sample(state_feature)
        q1_pi = self.Q_net1(state_feature, pi)
        q2_pi = self.Q_net2(state_feature, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_prob - min_q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning is True:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        # update target network
        for target_Q_net, Q_net in zip(self.Q_target_net_list, self.Q_net_list):    
            for target_param, param in zip(target_Q_net.parameters(), Q_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
            
        return q1_loss.item(), q2_loss.item(),policy_loss.item(),alpha_loss.item(),self.alpha.item()
            
    def get_ref(self, x):
        
        x = torch.FloatTensor(x).to(self.device)
        state_feature = self.state_encoder(x)
        return self.ref_decoder(state_feature)  
    
    def save(self, modelPath):
        import os
        torch.save(self.policy_net.state_dict(), os.path.join(modelPath, 'policy_net.pth'))
        torch.save(self.Q_net1.state_dict(), os.path.join(modelPath, 'Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), os.path.join(modelPath, 'Q_net2.pth'))
        
        torch.save(self.Q_target_net1.state_dict(), os.path.join(modelPath, 'Q_target_net1.pth'))
        torch.save(self.Q_target_net2.state_dict(), os.path.join(modelPath, 'Q_target_net2.pth'))

        print("====================================")
        print("Model has been saved...")
        print("====================================")