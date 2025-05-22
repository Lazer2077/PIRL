import torch
import numpy as np
def plot_value(self, state_batch, action_batch):
    import plotly.graph_objects as go
    import numpy as np  
    import torch
    # 初始化变量
    Qf = torch.tensor([[700.0, 0.0], [0.0, 700.0]]).to(self.device)
    aa = torch.tensor([0.5, -0.5, 0.0]).to(self.device)
    bb = torch.tensor([0.2, 0.3]).to(self.device)

    V1, V2, V3 = [], [], []

    # 扫描 k 并记录 Q 值
    for i in range(91):
        aa[2] = i
        vv1 = self.Q_target_net1(aa, bb)
        vv2 = self.Q_target_net2(aa, bb)
        vv3 = self.Q_target_net3(aa, bb)
        V1.append(-vv1.item())
        V2.append(-vv2.item())
        V3.append(-vv3.item())

    # 计算终端 reward 值
    tt = 0.5 * aa[:2].T @ Qf @ aa[:2]

    # 创建图形
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=V1, mode='lines', name='Q1', line=dict(width=2)))
    fig.add_trace(go.Scatter(y=V2, mode='lines', name='Q2', line=dict(width=2)))
    fig.add_trace(go.Scatter(y=V3, mode='lines', name='Q3', line=dict(width=2)))

    fig.add_trace(go.Scatter(
        x=[90], y=[tt.item()], mode='markers', name='Terminal reward',
        marker=dict(size=10, color='black', symbol='circle')
    ))
    # 更新布局
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

    # 保存文件
    fig.show()
    fig.write_html("Q_value_90.html")
    fig.write_image("Q_value_90.svg", scale=3)  # SVG 输出，适合论文矢量图
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
        self.N = 90
        self.k = 0
        self.IS_K = True
        if self.IS_K:
            self.x_dim = 2+1
            self.xmean = np.array([0.0, 0.0, 0.0])
            self.xstd = np.array([1.0, 1.0, self.N])
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
    
        