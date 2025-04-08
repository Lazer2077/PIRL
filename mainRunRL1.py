import argparse
from itertools import count
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import os, sys, random
from copy import deepcopy
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--update_iteration', default=1, type=int)
parser.add_argument('--mode', default='train', type=str) # test or train
parser.add_argument('--learning_rate', default=3e-4, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discount gamma
parser.add_argument('--capacity', default=2000, type=int) # replay buffer size
parser.add_argument('--max_episode', default=12000, type=int) #  num of  games
parser.add_argument('--batch_size', default=32, type=int) # mini batch size
parser.add_argument('--seed', default=True, type=bool)
parser.add_argument('--random_seed', default=526963494564900, type=int) # 108271139271800
parser.add_argument('--dynamic_noise', default=False, type=bool)
#parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
#parser.add_argument('--sample_frequency', default=256, type=int)
#parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
#parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
#parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument("--buffer_warm_size", type=int, default=256)
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--eval_interval', type=int, default=10,
                    help='Evaluates a policy a policy every X episode (default: 10)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')

parser.add_argument('--is_discrete', type=bool, default=True, metavar='G',
                    help='Is the action space discrete? (default: True) ')
args = parser.parse_args()

# create a folder to save model and training log

    
if args.seed:
    selectRandomSeed = args.random_seed
else:
    selectRandomSeed = torch.seed()

# env.seed(args.random_seed)
random.seed(selectRandomSeed)
torch.manual_seed(selectRandomSeed)
np.random.seed(selectRandomSeed & 0xFFFFFFFF)

# add system path

args.OPT_METHODS = 'SAC' #'ddpg' 'SAC' 'pinn' 'pinnsac' 'pinntry' 'sacwithv','pinnsac_3'
args.ENV_NAME = 'CartPole-v1' # 'cartpole' # #'SpeedTracking' # pendulumFH # SimpleSpeed # 'mountaincar','PointMassLQR'
args.ENABLE_VALIDATION = True
args.EnvOptions = {}

MODEL_NAME = f'model_{args.ENV_NAME}_diff_weights'
Env = gym.make(args.ENV_NAME)



Last_50_reward = 0
state_dim = Env.observation_space.shape[0]
action_dim = Env.action_space.n
ScalingDict = {}

from apscheduler.schedulers.background  import BackgroundScheduler
from RL_dashboard.DashHome import *

savePath = os.path.join(os.getcwd(), 'LogTmp', '{}_{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),MODEL_NAME))

DashInstance_c=DashInstance(Env, savePath, selectRandomSeed, port=8800)
from RL_dashboard.LogWriter import *
LogWriter_c = LogWriter(savePath, port=6001)

import OptMethods

def plotResults(agent, i_episode, steps):
    batch = agent.replay_buffer.getEpisodeBatch(steps)
    # batch->(observation, action)->value through calling NNs
    ValueDict = agent.replayEpisodeValue(batch) 
    # batch->(state, action)
    xaxis, TrajDict = Env.replayEpisode(batch)
    #AuxDict = deepcopy(agent.AuxDict)
    #AuxDict.update(tmp) # combine two dict
    if 0 and i_episode > 100:
        observationBatch = torch.FloatTensor(batch[0]).requires_grad_(True)
        actionBatch = torch.FloatTensor(batch[2]).reshape(-1,Env.nAction).requires_grad_(True)
        observationNextBatch = torch.FloatTensor(batch[1]).requires_grad_(True)
        #pErr,uLoss,Info = Env.calcDiff(observationBatch, actionBatch, observationNextBatch, agent.critic, agent.actor, USE_CUDA=False)
        pErr,uLoss,Info = Env.calcDiff(observationBatch, actionBatch, observationNextBatch, agent.Q_net1, agent.select_action, USE_CUDA=False)
    DashInstance_c.updatePlot(xaxis, TrajDict, ValueDict, i_episode, steps, agent.LossDict, agent.AuxDict)
    

def main():
    Last_50_reward = 0
    args.Env = Env
    Last_50_reward = 0
    if 'ddpg' in args.OPT_METHODS.lower():
        args.exploration_noise = 0.5
        args.dynamic_noise = False
        args.batch_size = 100
        args.gamma = 1
        args.update_iteration = 200
        args.buffer_warm_size = 1000
        args.actor_learning_rate = 1e-4
        args.critic_learning_rate = 1e-3
        pass
    elif 'sac' in args.OPT_METHODS.lower():
        pass
    elif 'try' in args.OPT_METHODS.lower():
        args.exploration_noise = 0.5
        args.dynamic_noise = False
        args.batch_size = 100
        args.gamma = 1
        args.update_iteration = 200
        args.buffer_warm_size = 1000
        args.actor_learning_rate = 1e-4
        args.critic_learning_rate = 1e-3
    else:
        pass
    if 'pinn' in args.OPT_METHODS.lower():
        args.valuePhysicalWeight = 0.1# 0.03
        args.policyPhysicalWeight = 0
    print(f"========= Exp Name: {MODEL_NAME}   Env: {args.ENV_NAME.lower()}   Agent: {args.OPT_METHODS.upper()} ===========")
    agent = getattr(OptMethods, '{}'.format(args.OPT_METHODS.upper()))(state_dim, action_dim, ScalingDict, device, args)
    episode_reward = 0
    iStepEvaluation = 0 # number of evaluation steps
    EvalReplayBuffer = OptMethods.lib.ReplayBuffer.Replay_buffer()
    total_numsteps = 0
    for i in range(1, args.max_episode):
            episode_steps = 0
            state, _ = Env.reset()
            episode_reward = 0
            for t in count():
                action = agent.select_action(state)
                # next_state shape [nObs], reward,terminated,trunctated shape [], scalar
                next_state, reward, terminated, truncated, _ = Env.step(action)
                episode_reward += reward
                #if args.render and i >= args.render_interval : Env.render()
                done=terminated or truncated
                agent.replay_buffer.push((state, next_state, action, reward, float(done))) # when done, there will be an artificial next_state be stored, but it will not be used for value estimation
                state = next_state
                #total_numsteps += 1
                episode_steps += 1
                # only start training when buffer size is larger than specified warm size
                if len(agent.replay_buffer.storage) >= args.buffer_warm_size:
                #if len(agent.replay_buffer.storage) >= args.batch_size:
                    # repeat update for update_iteration times
                    #print('update iteration: ')
                    Info = {'done': done}
                    for iUp in range(args.update_iteration):
                        #print(iUp,end=',')
                        # pass done to the update, so can either update only after done, or every step, based on the specific algorithm
                        Info['iUpdate'] = iUp
                        agent.update(args.batch_size, Info)
                        # if done:
                        #     print('update {} sec'.format(time.time()-tBeg))
                if done:
                    break
            # if i % args.log_interval == 0:
            #     agent.save()
            total_numsteps += episode_steps+1
            if args.max_episode-i < 50:
                Last_50_reward += episode_reward
                
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i, total_numsteps, episode_steps, episode_reward, 2))

            # plotResults(agent, i, t)
            try:
                LogWriter_c.writeAll(agent)
            except:
                print('LogWriter_c.writeAll failed')
                pass

            # validation steps
            if (args.ENABLE_VALIDATION) & (i % args.eval_interval == 0):
                avg_reward = 0.
                episodes = 10
                for _  in range(episodes):
                    state, _ = Env.reset()
                    episode_reward = 0
                    done = False
                    for t in count():
                        action = agent.select_action(state, IS_EVALUATION=True)

                        next_state, reward, terminated, truncated, _ = Env.step(action)
                        episode_reward += reward
                        done=terminated or truncated

                        EvalReplayBuffer.push((state, next_state, action, reward, float(done)))
                        state = next_state
                        if done:
                            break
                    avg_reward += episode_reward
                    #plotResults(agent, i, t)
                avg_reward /= episodes
                #agent.writer.add_scalar('avg_reward/test', avg_reward, i)
                iStepEvaluation += 1
                try:
                    LogWriter_c.writeEvaluation(avg_reward, iStepEvaluation)
                except:
                    print('LogWriter_c.writeEvaluation failed')
                    pass
                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {} ".format(episodes, avg_reward, 2))
                print("----------------------------------------")

                pass
    print("====== Train Finish, Avg Last 50 episode reward: {} ======".format(Last_50_reward/50))
if __name__ == '__main__':
    main()
