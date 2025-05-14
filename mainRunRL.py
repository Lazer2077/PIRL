import argparse
from itertools import count
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import os, sys, random
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--update_iteration', default=1, type=int)
parser.add_argument('--mode', default='train', type=str) # test or train
parser.add_argument('--learning_rate', default=1e-4, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discount gamma
parser.add_argument('--capacity', default=1e6, type=int) # replay buffer size
parser.add_argument('--max_episode', default=3000, type=int) #  num of  games
parser.add_argument('--batch_size', default=128, type=int) # mini batch size
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
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument("--buffer_warm_size", type=int, default=256)
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--eval_interval', type=int, default=20,
                    help='Evaluates a policy a policy every X episode (default: 10)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
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
args.OPT_METHODS = 'SAC3' #'ddpg' 'SAC' 'PINNSAC1' 'pinntry' 'sacwithv','pinnsac_3'
args.ENV_NAME = 'SimpleSpeed' # 'cartpole-v1', 'Acrobot-v1', 'Pendulum-v1','HalfCheetah-v4', Ant-v4
args.SELECT_OBSERVATION = 'poly'
args.ENABLE_VALIDATION = False
args.EnvOptions = {}
SEP_BUFFER = True if 'SAC3' in args.OPT_METHODS else False  

if 'ddpg' in args.OPT_METHODS.lower():
    args.exploration_noise = 0.5
    args.dynamic_noise = False
    args.batch_size = 100
    args.gamma = 1
    args.update_iteration = 200
    args.buffer_warm_size = 1000
    args.actor_learning_rate = 1e-4
    args.critic_learning_rate = 1e-3
if 'sac' in args.OPT_METHODS.lower():
    args.policy_type = 'Gaussian'
    
if 'ref' in args.OPT_METHODS.lower():
    obs_type = 'none'

if args.ENV_NAME == 'SimpleSpeed':
    from Env import SimpleSpeed
    
    if os.name == 'nt':
        dataPath = r'D:/RL/trainData.mat'
    else:
        dataPath = r'/mnt/d/RL/traindata.mat'
    Env = SimpleSpeed(dataPath, SELECT_OBSERVATION=args.SELECT_OBSERVATION, options=args.EnvOptions)
    args.is_discrete = False
    action_dim = Env.action_dim
    state_dim = Env.obs_dim  

    ScalingDict = {
            'actionMax': Env.umax,
            'actionMin': Env.umin,
            'xMean': Env.xmean, 
            'xStd': Env.xstd,
               }
    # construct continuous action space on gym 
    action_space = gym.spaces.Box(low=Env.umin, high=Env.umax, shape=(action_dim,))
    
else:
    Env = gym.make(args.ENV_NAME)
    if isinstance(Env.action_space, gym.spaces.Discrete):
        action_space = Env.action_space.n
        state_dim = Env.observation_space.shape[0]
        args.is_discrete = True
    else:  # Box
        action_space = Env.action_space.shape[0]
        state_dim = Env.observation_space.shape[0]
        args.is_discrete = False
        ScalingDict = {}


MODEL_NAME = f'{args.OPT_METHODS}_{args.SELECT_OBSERVATION}_{args.ENV_NAME}'

delete_command = 'python delete.py'
os.system(delete_command)

savePath = os.path.join(os.getcwd(), 'LogTmp', '{}_{}'.format(datetime.now().strftime("%m_%d_%H_%M"),MODEL_NAME))
writer = SummaryWriter(savePath)
port = 6007

import subprocess
import platform

def free_port(port: int):

    def get_pid_on_port(port):
        system = platform.system()
        try:
            if system == "Windows":
                cmd = f'netstat -aon | findstr :{port}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                for line in result.stdout.strip().split('\n'):
                    if 'LISTENING' in line:
                        parts = line.strip().split()
                        return parts[-1]  # PID
            else:
                cmd = f'lsof -i :{port} | grep LISTEN'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]  # PID
        except Exception as e:
            print(f"[!] Error getting PID on port {port}: {e}")
        return None

    def kill_pid(pid):
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
            else:
                subprocess.run(f'kill -9 {pid}', shell=True)
            print(f"[✔] Killed process {pid} on port {port}")
        except Exception as e:
            print(f"[!] Error killing process {pid}: {e}")

    pid = get_pid_on_port(port)
    if pid:
        print(f"[!] Port {port} is occupied by PID {pid}. Attempting to kill it...")
        kill_pid(pid)
    else:
        print(f"[✓] Port {port} is free.")

free_port(port)
if os.name == 'nt':
    cmd_line = '''start /b cmd.exe /k "tensorboard --logdir {} --port {} --reload_interval {} --reload_multifile True"'''.format(
        savePath, port, 10
    )
else:
    cmd_line = "tensorboard --logdir {} --port {} --reload_interval {} &".format(
        savePath, port, 10
    )
os.system(cmd_line)

import OptMethods
def main():
    print(f"========= Exp Name: {MODEL_NAME}   Env: {args.ENV_NAME.lower()}   Agent: {args.OPT_METHODS.upper()} ===========")
    agent = getattr(OptMethods, '{}'.format(args.OPT_METHODS.upper()))(state_dim, action_space, ScalingDict, device, args)
    episode_reward = 0
    iStepEvaluation = 0 # number of evaluation steps
    total_numsteps = 0
    for i in range(1, args.max_episode):
            episode_steps = 0
            state, _ = Env.reset()
            dp = Env.dp
            vp = Env.vp
            episode_reward = 0
            for t in count():
                # concat dp and env.k
                dp[-1] = Env.k
                # ref = np.concatenate((dp, vp), axis=-1) 
                action = agent.select_action(state, ref=dp)
                next_state, reward, terminated, truncated, _ = Env.step(action)
                episode_reward += reward
                done=terminated or truncated
                if SEP_BUFFER:
                    
                    if Env.k<51:
                        done1 = (Env.k==50)
                        agent.replay_buffer.push((state, next_state, action, reward, float(done1),dp))
                    elif 51 <=Env.k < 101 :
                        done2 = (Env.k==100)
                        agent.replay_buffer2.push((state, next_state, action, reward, float(done2),dp))
                    else:
                        agent.replay_buffer3.push((state, next_state, action, reward, float(done),dp))
                else:   
                    agent.replay_buffer.push((state, next_state, action, reward, float(done),dp))    
                    
                state = next_state
                episode_steps += 1
                if i % 10 == 0:  
                    for j in range(min(state_dim, 2)):
                        writer.add_scalar(f'Trajectory/Episode_{i}/State{j}', state[j], t)
                    for j in range(min(action_dim, 1)):
                        writer.add_scalar(f'Trajectory/Episode_{i}/Action{j}', action[j], t)
                    dfk = dp[int(dp[-1])] -state[j]
                    writer.add_scalar(f'Trajectory/Episode_{i}/CarFollowing', dfk, t)
                

                if len(agent.replay_buffer.storage) >= args.buffer_warm_size:
                    Info = {'done': done}
                    for iUp in range(args.update_iteration):
                        Info['iUpdate'] = iUp
                        agent.update(args.batch_size, Info)
                if done:
                    break
            if SEP_BUFFER:      
                Q_loss, policy_loss, alpha_loss, alpha = agent.update(args.batch_size)
                q1_loss, q2_loss, q3_loss = Q_loss
                writer.add_scalar(f'Loss/Q3', q3_loss, i)
            else:   
                q1_loss, q2_loss, policy_loss, alpha_loss, alpha = agent.update(args.batch_size)
            
            writer.add_scalar(f'Loss/Q1', q1_loss, i)
            writer.add_scalar(f'Loss/Q2', q2_loss, i)
            writer.add_scalar(f'Loss/Policy', policy_loss, i)
            writer.add_scalar(f'Loss/Alpha_loss', alpha_loss, i)
            writer.add_scalar(f'Loss/Alpha', alpha, i)
 
            total_numsteps += episode_steps+1
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i, total_numsteps, episode_steps, episode_reward, 2))
            writer.add_scalar('Episode/Reward', episode_reward, i)
           
            if (args.ENABLE_VALIDATION) & (i % args.eval_interval == 0):
                EvalReplayBuffer = OptMethods.lib.ReplayBuffer.Replay_buffer()
                avg_reward = 0.
                episodes = 10
                for _  in range(episodes):
                    state, _ = Env.reset()
                    episode_reward = 0
                    done = False
                    dp = Env.dp
                    vp = Env.vp
                    for t in count():
                        dp[-1] = Env.k
                        # ref = np.concatenate((dp, vp), axis=-1)
                        action = agent.select_action(state, ref=dp)
                        next_state, reward, terminated, truncated, _ = Env.step(action)
                        episode_reward += reward
                        done=terminated or truncated
                        EvalReplayBuffer.push((state, next_state, action, reward, float(done),dp))
                        state = next_state
                        if done:
                            break
                    avg_reward += episode_reward
                    #plotResults(agent, i, t)
                avg_reward /= episodes
                agent.save(savePath)
                iStepEvaluation += 1
                writer.add_scalar('Test/Reward', avg_reward, i)
                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {} ".format(episodes, avg_reward, 2))
                print("----------------------------------------")

if __name__ == '__main__':
    main()
