import torch
import numpy as np

def plot_value(self):
    import plotly.graph_objects as go
    xx = torch.tensor([0.0,0.0]).to(self.device).reshape(1,-1)
    uu = torch.tensor([0.0]).to(self.device).reshape(1,1)
    Num_Q = 2
    N = 150
    V = [[] for _ in range(Num_Q)]
    for i in range(N-1):
        xx[:,1] = i
        for j in range(Num_Q):
            vv = self.Q_target_net_list[j](xx, uu)
            V[j].append(-vv.item())
    x_next = xx + uu
    tt  = 0.5* x_next[:,0]**2 + 0.5* uu**2

    fig = go.Figure()
    for j in range(Num_Q):
        fig.add_trace(go.Scatter(y=V[j], mode='lines', name=f'Q{j+1}', line=dict(width=2)))

    fig.add_trace(go.Scatter(x=[N-2], y=[tt.item()], mode='markers', name='Terminal reward',
        marker=dict(size=10, color='black', symbol='circle')
    ))
    fig.update_layout(
        title=dict(text='Estimated Q-values vs. Time Step k', x=0.5, font=dict(size=20, family='Times New Roman')),
        xaxis_title='Time Step k',
        yaxis_title='Q-value',
        font=dict(family='Times New Roman', size=16),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig.show()
    fig.write_html(f"Linear_Q_{N}.html")
    fig.write_image(f"Linear_Q_{N}.svg", scale=3) 
    return 
    
          

class Linear:
    def __init__(self, options={}):
        self.options = options
        self.IS_K = True
        self.u_dim = 1
        self.umin = np.array([-5])
        self.umax = np.array([5])
        self.N =  150
        if self.IS_K:
            self.x_dim = 2
            self.xmean = np.array([0.0, 0.0])
            self.xstd = np.array([1.0, self.N])
        else:
            self.x_dim = 1
            self.xmean = np.array([0.0])
            self.xstd = np.array([1.0])
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
    
        