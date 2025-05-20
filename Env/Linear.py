import torch
import numpy as np

class Linear:
    def __init__(self, options={}):
        self.options = options
        self.IS_K = False
        if self.IS_K:
            self.x_dim = 2
        else:
            self.x_dim = 1
        self.u_dim = 1
        self.umin = np.array([-6])
        self.umax = np.array([6])
        self.N =  10
        self.k =  0
        self.dp = 0
        self.vp = 0
        
    def reset(self):
        import random
        self.k = 0
        if self.IS_K:
            self.x = torch.tensor([random.uniform(-2,2),self.k])         
        else:
            self.x = torch.tensor([random.uniform(-2,2)])
        return self.x, None
    
    def getReward(self, x, u):
        if self.k == self.N:
            tc = 0.5*x[0]**2 
            return -tc.unsqueeze(-1)
        else:
            sc = 0.5*u**2 
            return -sc
    
    def step(self, u):  
        x_next = self.getNextState(self.x, u)
        self.x = x_next
        self.k += 1
        done = (self.k == self.N)
        return x_next, self.getReward(x_next, u), done, False, None    
        
    def getNextState(self, x, u):
        if self.IS_K:
            x_next = x + u
            x_next[1] = self.k
        else:
            x_next = x + u
        return x_next
    
        