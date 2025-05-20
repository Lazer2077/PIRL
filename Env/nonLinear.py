import torch
import numpy as np

class NonLinear:
    def __init__(self, options={}):
        self.options = options
        self.Q = torch.tensor([[10.0,0.0],[0.0,10.0]])
        self.R = torch.tensor([[10.0,0.0],[0.0,10.0]])
        self.Qf = torch.tensor([[700.0,0.0],[0.0,700.0]])
        self.g = torch.tensor([[-0.2,0.0],[0.0,-0.2]])    
        self.x_dim = 2+1
        self.u_dim = 2
        self.umin = np.array([-4,-4])
        self.umax = np.array([4,4])
        self.N = 5
        self.k = 0
        self.IS_K = True
        self.dp = None
        self.vp = None
        
    def reset(self):
        import random
        self.k = 0
        self.x = torch.tensor([random.uniform(-1,1),random.uniform(-1,1),self.k])         
        return self.x, None
    
    def getReward(self, x, u):
        x = x[:2]
        if self.k == self.N:
            tc = 0.5* x.T @self.Qf@ x
            return - tc
        else:
            sc = 0.5* x.T @self.Q @ x + u.T @self.R@ u
            return -sc
    
    def step(self, u):  
        x_next = self.getNextState(self.x, u)
        self.x = x_next
        self.k += 1
        done = (self.k == self.N)
        if self.x[0] > 1e5:
            print('x[0] > 1e5'  )
        return x_next, self.getReward(x_next, u), done, False, None    
        
    def getNextState(self, x, u):
        x_next = torch.zeros_like(x)
        x_next[0] = 0.2*x[0] * torch.exp(x[1]**2) + self.g[0,0]*u[0]
        x_next[1] = 0.3*x[1]**3 + self.g[1,1]*u[1]
        x_next[2] = self.k
        return x_next
    
        