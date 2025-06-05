import numpy as np
import gym
from gym import spaces
import torch
class LQT():
    def __init__(self, N=50, dt=0.1,args=None):
        super(LQT, self).__init__()
        self.dt = dt
        self.N = N
        self.n = 4
        self.m = 2
        self.k = 0
        self.u_dim = 2
        self.umin = -3.0
        self.umax = 3.0
        self.obs_dim = 4
        if args is None:
            self.with_ref = True
        else:
            if args.SELECT_OBSERVATION == 'ref':
                self.with_ref = True
            elif args.SELECT_OBSERVATION == 'none':
                self.with_ref = False
        if self.with_ref:
            self.x_dim = 4 + 4 
        else:
            self.x_dim = 4
        
        self.A = torch.FloatTensor([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = torch.FloatTensor([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])

        # Cost weights
        self.Q = torch.FloatTensor([
            [100, 0, 0, 0],
            [0, 100, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.R = torch.FloatTensor([
            [0.1, 0],
            [0, 0.1]
        ])
        self.Qf = torch.FloatTensor([
            [200, 0, 0, 0],
            [0, 200, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2]
        ])

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def generate_reference_trajectory(self):
        t_series = np.arange(self.N + 1) * self.dt
        A = np.random.uniform(0.0, 5.0)     # Amplitude x
        B = np.random.uniform(0.0, 5.0)     # Amplitude y
        w = np.random.uniform(0.01, 0.3)    # Frequency

        x = A * np.sin(w * t_series)
        y = B * np.sin(w * t_series)
        vx = np.gradient(x, self.dt)
        vy = np.gradient(y, self.dt)

        self.dp = np.stack([x, y, vx, vy], axis=1)  # shape (N+1, 4)
        self.dpA = A
        self.dpB = B
        self.dpw = w
        return A, B, w
    
    def trajectory_derivative(self, k):
        derivative = torch.FloatTensor([self.dpA * self.dpw * np.cos(self.dpw * k * self.dt), 
                           self.dpB * self.dpw * np.cos(self.dpw * k * self.dt)])
        return derivative
    

    def get_dp(self, k):
        traj = torch.FloatTensor(self.dp[k,:])
        return traj



    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        r_k = self.get_dp(self.k)

        cost = (self.x - r_k).T @ self.Q @ (self.x - r_k) + action.T @ self.R @ action

        self.x = self.A @ self.x + self.B @ action
        self.k += 1

        done = (self.k >= self.N)
        if done:
            r_N = self.get_dp(self.k)
            cost += (self.x - r_N).T @ self.Qf @ (self.x - r_N)
        cost = cost*0.01
        if self.with_ref:
            return torch.cat([self.x, torch.FloatTensor([self.k, self.dpA, self.dpB, self.dpw])], dim=0), -cost, done, done , {}
        else:
            return self.x, -cost, done, done , {}

    def reset(self, random_state=False):
        # random initial state
        if random_state:
            self.x = torch.FloatTensor([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 0, 0])
        else:
            self.x = torch.FloatTensor([0, 0, 0, 0])
        self.k = 0
        self.generate_reference_trajectory()
        if self.with_ref:
            return torch.cat([self.x, torch.FloatTensor([self.k, self.dpA, self.dpB, self.dpw])], dim=0), None
        else:
            return self.x , None


