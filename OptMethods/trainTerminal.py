import numpy as np

def TerminalReward(state, dp_final, vp_final):
    w5 = 1
    w6 = 10
    ht = 1.5
    dmin = 1
    s = state[:, 0]
    v = state[:, 1]
    df = dp_final - s
    vf = vp_final - v
    dsafe = vf * ht + dmin
    reward = w5 * (df - dsafe)**2 + w6 * vf**2
    return reward

# 生成训练数据
N = 100000
state = np.random.uniform(low=[0, 0], high=[100, 30], size=(N, 2))
dp_final = np.random.uniform(0, 100, size=(N,))
vp_final = np.random.uniform(0, 30, size=(N,))
reward = TerminalReward(state, dp_final, vp_final)

# 拼接输入
X = np.hstack([state, dp_final.reshape(-1,1), vp_final.reshape(-1,1)])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

y = reward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 转换为 tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 构造数据集
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

# 定义网络
import torch.nn.functional as F

class TerminalQ(nn.Module):
    def __init__(self,  xMean, xStd, state_dim=4, hidden_dim=256):
        super(TerminalQ, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.xmean = torch.tensor(xMean, dtype=torch.float32).to('cuda')
        self.xstd = torch.tensor(xStd, dtype=torch.float32).to('cuda')
    def load_state_dict(self, path = 'terminal.pth'):
        self.load_state_dict(torch.load(path))
    def forward(self, s):
        s = (s - self.xmean) / self.xstd
        x = F.gelu(self.fc1(s))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)
    
    
model = TerminalQ(X_mean, X_std).to('cuda')
# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_list = []
import matplotlib.pyplot as plt
import tqdm
for epoch in tqdm.tqdm(range(5000), desc='Training'):
    for xb, yb in loader:
        # convert to float64
        xb = xb.to('cuda')
        yb = yb.to('cuda')
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        loss_list.append(loss.item())
        plt.figure(figsize=(10, 5)) 
        xx = np.arange(len(loss_list))*10
        plt.plot(xx, loss_list)
        plt.yscale('log') 
        plt.savefig(f'loss.png')
        plt.close()
        # save the model 
        torch.save(
            {
                'model': model.state_dict(),
                'X_mean': X_mean,
                'X_std': X_std
            },
            'terminal.pth'
        )
# laod the model 





