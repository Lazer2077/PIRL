import numpy as np
import gym
from gym import spaces

class LQT2DEnv(gym.Env):
    def __init__(self, N=100, dt=0.1):
        super(LQT2DEnv, self).__init__()
        self.dt = dt
        self.N = N
        self.n = 4
        self.m = 2
        self.t = 0

        # System dynamics
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])

        # Cost weights
        self.Q = np.diag([100, 100, 1, 1])
        self.R = np.diag([0.1, 0.1])
        self.Qf = np.diag([200, 200, 2, 2])

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.reset()

    def generate_reference_trajectory(self):
        t_series = np.arange(self.N + 1) * self.dt
        A = np.random.uniform(1.0, 2.0)     # Amplitude x
        B = np.random.uniform(1.0, 2.0)     # Amplitude y
        w = np.random.uniform(0.05, 0.2)    # Frequency

        x = A * np.sin(w * t_series)
        y = B * np.cos(w * t_series)
        vx = np.gradient(x, self.dt)
        vy = np.gradient(y, self.dt)

        self.traj = np.stack([x, y, vx, vy], axis=1)  # shape (N+1, 4)

    def reference(self, t):
        if t < len(self.traj):
            return self.traj[t]
        else:
            return self.traj[-1]  # hold last position if overflow

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        r_k = self.reference(self.t)

        cost = (self.x - r_k).T @ self.Q @ (self.x - r_k) + action.T @ self.R @ action

        self.x = self.A @ self.x + self.B @ action
        self.t += 1

        done = (self.t >= self.N)
        if done:
            r_N = self.reference(self.t)
            cost += (self.x - r_N).T @ self.Qf @ (self.x - r_N)

        return self.x.copy(), -cost, done, {}

    def reset(self):
        self.x = np.zeros(4, dtype=np.float32)
        self.t = 0
        self.generate_reference_trajectory()
        return self.x.copy()
