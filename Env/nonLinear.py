import torch
import numpy as np
def plot_value(self):
    import plotly.graph_objects as go
    N = 6
    Num_Q = 2
    Qf = torch.tensor([[700.0, 0.0], [0.0, 700.0]]).to(self.device)
    aa = torch.tensor([0.5, -0.5, 0.0]).to(self.device)
    bb = torch.tensor([0.2, 0.3]).to(self.device)
    V = [[] for _ in range(Num_Q)]
    for k in range(N+1):
        aa[2] = k
        for j in range(Num_Q):  
            vv = self.Q_net_list[j](aa, bb)
            V[j].append(-vv.item())
    tt = 0.5 * aa[:2].T @ Qf @ aa[:2]
    fig = go.Figure()
    for j in range(Num_Q):
        fig.add_trace(go.Scatter(y=V[j], mode='lines', name=f'Q{j+1}', line=dict(width=2)))

    fig.add_trace(go.Scatter(
        x=[N], y=[tt.item()], mode='markers', name='Terminal reward',
        marker=dict(size=10, color='black', symbol='circle')
    ))
    fig.update_layout(
        title=dict(
            text='Estimated Q-values vs. Time Step k',
            x=0.5,
            font=dict(size=20, family='Times New Roman')
        ),
        xaxis_title='Time Step k',
        yaxis_title='Q-value',
        font=dict(family='Times New Roman', size=16),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig.show()
    fig.write_html(f"NonLinear_Q{Num_Q}_{N}.html")
    fig.write_image(f"NonLinear_Q{Num_Q}_{N}.svg", scale=3)
    return 

class NonLinear:
    def __init__(self, options={}):
        self.options = options
        self.Q = torch.tensor([[10.0,0.0],[0.0,10.0]])
        self.R = torch.tensor([[10.0,0.0],[0.0,10.0]])
        self.Qf = torch.tensor([[700.0,0.0],[0.0,700.0]])
        self.g = torch.tensor([[-0.2,0.0],[0.0,-0.2]])    
        self.u_dim = 2
        self.umin = np.array([-3,-3])
        self.umax = np.array([3,3])
        self.N = 6
        self.k = 0
        self.IS_K = True
        if self.IS_K:
            self.x_dim = 2+1
            self.xmean = np.array([0.0, 0.0, 0.0])
            self.xstd = np.array([1.0, 1.0, self.N-1])
        else:
            self.x_dim = 2
            self.xmean = np.array([0.0, 0.0])
            self.xstd = np.array([1.0, 1.0])
            
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
        return x_next, self.getReward(x_next, u), done, False, None    
        
    def getNextState(self, x, u):
        x_next = torch.zeros_like(x)
        x_next[0] = 0.2*x[0] * torch.exp(x[1]**2) + self.g[0,0]*u[0]
        x_next[1] = 0.3*x[1]**3 + self.g[1,1]*u[1]
        if self.IS_K:
            x_next[2] = self.k
        return x_next
    
        