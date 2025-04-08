import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import numpy as np
import os
import json, inspect, copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

from .lib.ReplayBuffer import Replay_buffer
from .lib.NeuroModel import *

ENABLE_PINN = True

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, xMean, xStd, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.xmean = xMean
        self.xstd = xStd

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        xu = (xu-self.xmean)/self.xstd

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2    
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        #self.max_action = max_action

        self.xmean = xMean
        self.xstd = xStd

        self.apply(weights_init_)

    def forward(self, x):
        if not x.is_cuda:    
            x = x.cuda()

        x = (x-self.xmean)/self.xstd

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        #log_std_head = F.relu(self.log_std_head(x))
        log_std_head = self.log_std_head(x)
        log_std_head = torch.clamp(log_std_head, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std_head

# class Critic(nn.Module):
#     def __init__(self, state_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim, xMean, xStd, hidden_dim=256):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.xmean = xMean
        self.xstd = xStd

        self.apply(weights_init_)

    def forward(self, s, a):
        if not s.is_cuda:    
            s = s.cuda()
   
        if not a.is_cuda:    
            a = a.cuda()
    
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1) # combination s and a

        x = (x-self.xmean)/self.xstd

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class PINNSAC():
    def __init__(self, state_dim, action_dim, ScalingDict, device, args):
        super(PINNSAC, self).__init__()

        # hyperparameters
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = torch.FloatTensor([args.alpha]).to(device)

        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = ScalingDict['actionMax']

        self.replay_buffer = Replay_buffer()

        xumean = torch.cat([ScalingDict['xMean'].to(device), ScalingDict['uMean'].to(device)])
        xustd = torch.cat([ScalingDict['xStd'].to(device), ScalingDict['uStd'].to(device)])
        xMean = ScalingDict['xMean'].to(device)
        xStd = ScalingDict['xStd'].to(device)

        self.policy_net = Actor(state_dim, action_dim, ScalingDict['xMean'].to(device), ScalingDict['xStd'].to(device), hidden_dim=args.num_hidden_units_per_layer).to(device)
        self.Q_net1 = Q(state_dim, action_dim, xumean, xustd, hidden_dim=args.num_hidden_units_per_layer).to(device)
        self.Q_net2 = Q(state_dim, action_dim, xumean, xustd, hidden_dim=args.num_hidden_units_per_layer).to(device)
        self.Q_target_net1 = Q(state_dim, action_dim, xumean, xustd, hidden_dim=args.num_hidden_units_per_layer).to(device)
        self.Q_target_net2 = Q(state_dim, action_dim, xumean, xustd, hidden_dim=args.num_hidden_units_per_layer).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=args.learning_rate)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=args.learning_rate)

        self.value_net = ThreeLayerMLP(state_dim, xMean, xStd, 1, [args.num_hidden_units_per_layer,args.num_hidden_units_per_layer]).to(device)
        self.value_net_target = ThreeLayerMLP(state_dim, xMean, xStd, 1, [args.num_hidden_units_per_layer,args.num_hidden_units_per_layer]).to(device)
        self.value_net_target.load_state_dict(self.value_net.state_dict())
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)

        self.critic = QNetwork(state_dim, action_dim, xumean, xustd, args.num_hidden_units_per_layer).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.learning_rate)

        self.critic_target = QNetwork(state_dim, action_dim, xumean, xustd, args.num_hidden_units_per_layer).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor((action_dim,)).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.learning_rate)

        self.num_training = 0
        #self.writer = SummaryWriter('./test_agent')

        for target_param, param in zip(self.Q_target_net1.parameters(), self.Q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.Q_target_net2.parameters(), self.Q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.Env = args.Env

        self.device = device

        self.valuePhysicalWeight = args.valuePhysicalWeight
        self.policyPhysicalWeight = args.policyPhysicalWeight

        self.LossDict = {'Q1': 0,
                    'Q2': 0,
                    'Policy': 0,
                    'Alpha': 0, 
                    'Value': 0,
                    'Physics1': 0,
                    'Physics2': 0,
                    'Physics': 0}
        
        self.AuxDict = {'Alpha': 0}

        self.LogDict = {'Q1': {'model': self.Q_net1, 'epoch': self.num_training, 'loss': 0},
                        'Q2': {'model': self.Q_net2, 'epoch': self.num_training, 'loss': 0},
                        'Policy': {'model': self.policy_net, 'epoch': self.num_training, 'loss': 0},
                        'Value': {'model': self.value_net, 'epoch': self.num_training, 'loss': 0}}
        
        #os.makedirs('./SAC_model/', exist_ok=True)
    def getValue(self,state,action):
        min_qf_pi = torch.min(self.Q_net1(state, action), self.Q_net2(state, action))
        return min_qf_pi.detach().cpu().item()
    
    def genDiffFunc(self, x, xnext, NNfunc):
   
        def __reluDiff(x):
            dReLu1dx = x
            dReLu1dx[dReLu1dx<=0]=0
            dReLu1dx[dReLu1dx>0]=1
            dReLu1dx=torch.diag_embed(dReLu1dx)

            return dReLu1dx

        A1=NNfunc.fc1.weight
        B1=NNfunc.fc1.bias
        A2=NNfunc.fc2.weight
        B2=NNfunc.fc2.bias
        A3=NNfunc.fc3.weight
        B3=NNfunc.fc3.bias

        dReLu1dx = __reluDiff(torch.matmul(x, A1.T)+B1)
        dReLu2dx = __reluDiff(torch.matmul(torch.max(torch.matmul(x, A1.T)+B1, torch.Tensor([0]).cuda()), A2.T)+B2)
        dNNdx = torch.matmul(A3, torch.matmul(dReLu2dx, torch.matmul(A2, torch.matmul(dReLu1dx, A1))))

        dReLu1dx = __reluDiff(torch.matmul(xnext, A1.T)+B1)
        dReLu2dx = __reluDiff(torch.matmul(torch.max(torch.matmul(xnext, A1.T)+B1, torch.Tensor([0]).cuda()), A2.T)+B2)
        dNNdxnext = torch.matmul(A3, torch.matmul(dReLu2dx, torch.matmul(A2, torch.matmul(dReLu1dx, A1))))

        dAgent_dict = {'dVdx': dNNdx[:,:,:self.state_dim],
                   'dVdxnext': dNNdxnext[:,:,:self.state_dim]}

        return dAgent_dict
    
    def select_action(self, state, IS_EVALUATION=False):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state)
        state = state.to(self.device)
        # mu, log_sigma = self.policy_net(state)
        # sigma = torch.exp(log_sigma)
        # dist = Normal(mu, sigma)
        # z = dist.sample()
        # action = self.action_range * torch.tanh(z).detach().cpu().numpy()
        if not IS_EVALUATION:
            action, _, _, _, _ = self.evaluate(state)
        else:
            _, _, _, _, action = self.evaluate(state)
        return action.reshape(-1,self.action_dim).detach().cpu()

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        
        # noise = Normal(0, 1)
        # z = noise.sample()
        # action0 = torch.tanh(batch_mu + batch_sigma*z.to(device))

        x_t = dist.rsample() # for reparameterization trick (mean + std * N(0,1))
        action0 = torch.tanh(x_t)

        action = self.action_range * action0
        log_prob = dist.log_prob(x_t) - torch.log(self.action_range * (1 - action0.pow(2) + EPSILON))

        actionMean = torch.tanh(batch_mu) * self.action_range
        #return action, log_prob, batch_mu, batch_log_sigma
        return action, log_prob, batch_mu, batch_log_sigma, actionMean

    def update(self, batch_size, Info):

        done = Info['done']
        
        # if self.num_training % 500 == 0:
        #     print("Training ... \t{} times ".format(self.num_training))

        # Sample replay buffer
        x, y, u, r, d = self.replay_buffer.sample(batch_size)
        observation = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device).reshape(-1,self.action_dim)
        next_observation = torch.FloatTensor(y).to(self.device)
        done_batch = torch.FloatTensor(1-np.array(d)).to(self.device).reshape(-1,1)
        reward_batch = torch.FloatTensor(r).to(self.device).reshape(-1,1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _ = self.evaluate(next_observation)
            qf1_next_target = self.Q_target_net1(next_observation, next_state_action)
            qf2_next_target = self.Q_target_net2(next_observation, next_state_action)
            # qf1_next_target, qf2_next_target = self.critic_target(next_observation, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + done_batch * self.gamma * (min_qf_next_target)

        # Dual Q net to mitigate positive bias in the policy improvement step
        excepted_Q1 = self.Q_net1(observation, action)
        excepted_Q2 = self.Q_net2(observation, action)
        qf1_loss = F.mse_loss(excepted_Q1, next_q_value.detach()).mean() # J_Q
        qf2_loss = F.mse_loss(excepted_Q2, next_q_value.detach()).mean()


        #################################################################
        # calculate PMP condition: need to make sure to use torch to calculate jacobian so that it can get gradient for NN iteration
        #################################################################
       
        from copy import deepcopy
        obs = deepcopy(observation).requires_grad_(True)
        act = deepcopy(action).requires_grad_(True)
        obsnext = deepcopy(next_observation).requires_grad_(True)
        actnext = self.select_action(obsnext, IS_EVALUATION=True).cuda()

        #import time
        #tBeg = time.time()

        excepted_value = self.value_net(observation)
        sample_action, log_prob, _, _, _ = self.evaluate(observation)
        excepted_new_Q = torch.min(self.Q_net1(observation, sample_action), self.Q_net2(observation, sample_action))
        _, index_exp_new_Q = torch.min(torch.hstack((self.Q_net1(observation, sample_action), self.Q_net2(observation, sample_action))),dim=1)
        next_value = excepted_new_Q - self.alpha * log_prob

        policy_loss = (self.alpha * log_prob - excepted_new_Q).mean() # according to original paper

        if 1 and ENABLE_PINN:
            #dAgent_dict=self.genDiffFunc(self.critic, torch.cat([obs,act],1), torch.cat([obsnext, self.actor(obsnext)],1))
            # dAgent_dict=self.genDiffFunc(torch.cat([obs,act],1), torch.cat([obsnext, actnext],1), self.Q_net1)
            # # dAgent_dict = {}
            # # dAgent_dict['dVdx']=torch.autograd.functional.jacobian(lambda obs,act: self.Q_net1(obs, act).cpu()-self.alpha.cpu()*self.evaluate(obs)[1].cpu(),(obs.cpu(), act.cpu()), create_graph=True, vectorize=True)[0][range(batch_size),:,range(batch_size),:].reshape(batch_size,-1).cuda()
            # # dAgent_dict['dVdxnext']=torch.autograd.functional.jacobian(lambda obs,act: self.Q_net1(obs, act).cpu()-self.alpha.cpu()*self.evaluate(obs)[1].cpu(),(obsnext.cpu(), actnext.cpu()), create_graph=True, vectorize=True)[0][range(batch_size),:,range(batch_size),:].reshape(batch_size,-1).cuda()
            dAgent_dict1 = self.genDiffFunc(torch.cat([obs,sample_action],1), torch.cat([obsnext, next_state_action],1), self.Q_net1)
            pErr1, uLoss, info = self.Env.calcDiff(obs, sample_action, obsnext, dAgent_dict1)
            physical_loss1 = F.mse_loss(pErr1.to(self.device), torch.zeros(pErr1.shape).to(self.device))
            
            dAgent_dict2 = self.genDiffFunc(torch.cat([obs,sample_action],1), torch.cat([obsnext, next_state_action],1), self.Q_net2)
            pErr2, uLoss, info = self.Env.calcDiff(obs, sample_action, obsnext, dAgent_dict2)
            physical_loss2 = F.mse_loss(pErr2.to(self.device), torch.zeros(pErr2.shape).to(self.device))
            physical_loss2 = torch.FloatTensor([0])[0].cuda()
            wp = self.valuePhysicalWeight 
            wp1 = wp
            wp2 = wp
        # def __MultipleSampleDiff(self,observation):
        #     # take many sample to approximate the value 
        #     NUM_SAMPLE = 10
        #     for k in range(NUM_SAMPLE):
        #         sample_action,  _, _, _ = self.evaluate(observation)
        #         dAgent_dict1 = self.genDiffFunc(torch.cat([observation,sample_action],1), torch.cat([next_observation, next_state_action],1), self.Q_net1)
        #         # take average of the dAgent_dict1
        #         if k==0:
        #             dAgent_dict1_sum = dAgent_dict1
        #         else:
        #             for key in dAgent_dict1.keys():
        #                 dAgent_dict1_sum[key] += dAgent_dict1[key]
        #     for key in dAgent_dict1_sum.keys():
        #         dAgent_dict1_sum[key] /= NUM_SAMPLE
        #     return dAgent_dict1_sum
                
        dAgent_dict1 = self.genDiffFunc(torch.cat([observation,sample_action],1), torch.cat([next_observation, next_state_action],1), self.Q_net1)                
        pErr1, uLoss, info = self.Env.calcDiff(observation, sample_action, next_observation, dAgent_dict1)
        physical_loss1 = F.mse_loss(pErr1.to(self.device), torch.zeros(pErr1.shape).to(self.device))
        # physical_loss1 = torch.FloatTensor([0])[0].cuda() # only physical loss 2 
        
        
        dAgent_dict2 = self.genDiffFunc(torch.cat([observation,sample_action],1), torch.cat([next_observation, next_state_action],1), self.Q_net2)
        pErr2, uLoss, info = self.Env.calcDiff(observation, sample_action, next_observation, dAgent_dict2)
        physical_loss2 = F.mse_loss(pErr2.to(self.device), torch.zeros(pErr2.shape).to(self.device))

        wp = self.valuePhysicalWeight # 0.03 # weight of physical loss, physical loss will have nObs times more elements than critic_loss, so at least should use wp=1/nObs 

        wp1 = wp
        wp2 = wp

            # wp1 = qf1_loss/physical_loss1/5
            # wp2 = qf2_loss/physical_loss2/5
        actorPhysicalLoss = F.mse_loss(uLoss.to(self.device), torch.zeros(uLoss.shape).to(self.device))
        wup = self.policyPhysicalWeight
        idx = index_exp_new_Q.repeat(1,self.Env.nObservation).flatten().reshape(self.Env.nObservation,-1).T.flatten()
        pErr = torch.zeros(pErr1.shape).to(self.device)
        pErr[idx==0]=pErr1[idx==0]
        pErr[idx==1]=pErr2[idx==1]
        physical_loss_atU = F.mse_loss(pErr.to(self.device), torch.zeros(pErr.shape).to(self.device))
        wpu = wp
        wpu = policy_loss/physical_loss_atU/5
        physical_loss = torch.FloatTensor([0])[0].cuda() 

        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        # print(qf1_loss/physical_loss1)
        total_loss =  (qf1_loss+wp1*physical_loss1)+(qf2_loss)+policy_loss # 362.5357
        total_loss.backward(retain_graph=False)
        
        self.Q1_optimizer.step()
        self.Q2_optimizer.step()
        self.policy_optimizer.step()
        

        value_loss = F.mse_loss(excepted_value, next_value.detach()).mean()  # J_V

        self.value_optimizer.zero_grad()
        (value_loss+wp*physical_loss).backward(retain_graph=True)
        #nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        # # policy net update
        # # sample_action, log_prob, _, _, _ = self.evaluate(observation)
        # # min_qf_pi = torch.min(self.Q_net1(observation, sample_action), self.Q_net2(observation, sample_action))
        # # qf1_pi, qf2_pi = self.critic(state_batch, sample_action)
        # # min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # # if ENABLE_PINN:
        # #     aCalc, logCalc, _, _, _ = self.evaluate(next_observation)
        # #     policy_loss = (self.alpha * logCalc - ( self.Env.getReward(observation, aCalc, IS_OBS=True) + done_batch * self.gamma * self.value_net(self.Env.calcDyn(observation, aCalc, IS_OBS=True)) ) ).mean()
        # # else:
        #     # sample_action, log_prob, _, _, _ = self.evaluate(observation)
        #     # excepted_new_Q = torch.min(self.Q_net1(observation, sample_action), self.Q_net2(observation, sample_action))

        # self.policy_optimizer.zero_grad()
        # policy_loss.backward(retain_graph=True)
        # # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        # self.policy_optimizer.step()
    
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        # update target v net update
        for target_param, param in zip(self.Q_target_net1.parameters(), self.Q_net1.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
        for target_param, param in zip(self.Q_target_net2.parameters(), self.Q_net2.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)
        for param, target_param in zip(self.value_net.parameters(), self.value_net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
        # self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
        # self.writer.add_scalar('Loss/policy_loss', pi_loss, global_step=self.num_training)

        self.num_training += 1

        self.LossDict = {'Q1': qf1_loss.cpu().item(),
                    'Q2': qf2_loss.cpu().item(),
                    'Policy': policy_loss.cpu().item(),
                    'Alpha': alpha_loss.cpu().item(),
                    'Value': value_loss.cpu().item(),
                    'Physics1': physical_loss1.cpu().item(),
                    'Physics2': physical_loss2.cpu().item(),
                    'Physics': physical_loss.cpu().item()}
        
        self.AuxDict = {'Alpha': self.alpha.cpu().item()}

        self.LogDict = {'Q1': {'model': self.Q_net1, 'epoch': self.num_training, 'loss': qf1_loss.cpu().item()},
                        'Q2': {'model': self.Q_net2, 'epoch': self.num_training, 'loss': qf2_loss.cpu().item()},
                        'Policy': {'model': self.policy_net, 'epoch': self.num_training, 'loss': policy_loss.cpu().item()},
                        'Value': {'model': self.value_net, 'epoch': self.num_training, 'loss': value_loss.cpu().item()}}

    def save(self, modelPath):
        torch.save(self.policy_net.state_dict(), os.path.join(modelPath, 'policy_net.pth'))
        torch.save(self.Q_net1.state_dict(), os.path.join(modelPath, 'Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), os.path.join(modelPath, 'Q_net2.pth'))
        torch.save(self.Q_target_net1.state_dict(), os.path.join(modelPath, 'Q_target_net1.pth'))
        torch.save(self.Q_target_net2.state_dict(), os.path.join(modelPath, 'Q_target_net2.pth'))
        #torch.save(self.critic_target.state_dict(), os.path.join(savePath, 'critic_target.pth'))
        
        argsToSave = {}
        for key, val in vars(self.args).items():
            if key == 'Env':
                continue
            if key == 'EnvOptions':
                tmp = {}
                for k, v in val.items():
                    if k == 'dataFilter':
                        tmp[k] = inspect.getsource(v)
                    else:
                        tmp[k] = v
                argsToSave[key] = tmp
            else:
                argsToSave[key] = val
        txt_path = os.path.join(modelPath, 'commandline_args.txt')                
        with open(txt_path, 'w') as f:
            json.dump(argsToSave, f, indent=2)

        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, modelPath):
        self.policy_net.load_state_dict(torch.load(os.path.join(modelPath, 'policy_net.pth')))
        self.Q_net1.load_state_dict(torch.load(os.path.join(modelPath, 'Q_net1.pth')))
        self.Q_net2.load_state_dict(torch.load(os.path.join(modelPath, 'Q_net2.pth')))
        self.Q_target_net1.load_state_dict(torch.load(os.path.join(modelPath, 'Q_target_net1.pth')))
        self.Q_target_net2.load_state_dict(torch.load(os.path.join(modelPath, 'Q_target_net2.pth')))
        print("model has been load")

        
    def replayEpisodeValue(self, batch):
        observationBatch = torch.FloatTensor(batch[0])
        actionBatch = torch.FloatTensor(batch[2]).reshape(-1,self.action_dim)
        if batch.__len__() > 5:
            lqr_value =  batch[5]
            lqr_reward = batch[6]
            lqr_reward_rev = lqr_reward[::-1]
            cummulative_LQR_Reward = np.cumsum(lqr_reward_rev)
            cummulative_LQR_Reward = -cummulative_LQR_Reward[::-1]
            reward_rev = batch[3][::-1]
            
            cummulativeReward = np.cumsum(reward_rev)
            cummulativeReward = -cummulativeReward[::-1]
        else:
            lqr_value = 0
            lqr_reward= 0 
            cummulative_LQR_Reward = 0
            cummulativeReward = 0

        
        
        ValueDict = {'reward': batch[3],
                    #  'Q1': self.Q_net1(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                    #  'Q2': self.Q_net2(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                    #  'Q1_t': self.Q_target_net1(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                    #  'Q2_t': self.Q_target_net2(observationBatch.to(self.device), actionBatch.to(self.device)).detach().cpu().data.numpy().flatten(),
                     'lqr_value': lqr_value,
                    'RL_cumSumReward': cummulativeReward,
                    'LQR_cumSumReward': cummulative_LQR_Reward,
                     'lqr_reward': lqr_reward}

        return ValueDict

