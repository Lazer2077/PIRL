import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class RefEncoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=256):
        super(RefEncoder, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)    
        self.fc4 = nn.Linear(hidden_dim, num_outputs)    

    def forward(self, para):
        x = F.relu(self.fc1(para))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) 
        return x
    def compute_gradient_wrt_k(self, para):
        para = para.clone().detach()
        k = para[:, 0].clone().detach().requires_grad_(True)
        para[:, 0] = k

        output = self.forward(para)  # shape: [batch_size, 2]
        grads = []

        for i in range(output.shape[1]):
            grad = torch.autograd.grad(
                outputs=output[:, i],
                inputs=k,
                grad_outputs=torch.ones_like(output[:, i]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grads.append(grad)  # Each is [batch_size]

        # Stack into [batch_size, 2]
        return torch.stack(grads, dim=1)
        
from Env.LQT import LQT

if __name__ == "__main__":
    N =50 
    env = LQT(N=N)
    batch_size=128 
    num_batches = 1000
    num_epochs = 400
    DP = [] 
    para = []
    for n in range(num_batches):
        env.generate_reference_trajectory()
        for i in range(N):
            para.append(np.array([i,env.dpA, env.dpB, env.dpw]))
            DP.append(env.get_dp(i))
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DP = torch.stack(DP, dim=0).to(device)
    para = np.stack(para, axis=0)
    X = torch.FloatTensor(para)
    criterion = nn.MSELoss()
    # convert to torch datasets 
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, DP), batch_size=batch_size, shuffle=True)
    ref_enc = RefEncoder(num_inputs=4, num_outputs=4, hidden_dim=256)
    optimizer = torch.optim.Adam(ref_enc.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_enc.to(device)
    avg_loss_list = []
    eval_loss_list = []
    for epoch in range(num_epochs):
        ref_enc.train()
        total_loss = 0
        # plot loss 
        for batch_para, batch_dp in dataloader:
            # to device
            batch_para = batch_para.to(device)
            batch_dp = batch_dp.to(device)
            y = ref_enc(batch_para)
            y_prime = ref_enc.compute_gradient_wrt_k(batch_para)
            loss = criterion(y, batch_dp) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss_list.append(total_loss/len(dataloader))    
        # save model 
        
        
        ref_enc.eval()
        env.generate_reference_trajectory()
    
        random_k = np.random.randint(0, N)  
        X_test=  torch.FloatTensor([[random_k, env.dpA, env.dpB, env.dpw]]).to(device)
        Y_test=  env.get_dp(random_k).to(device).reshape(1,-1)
        y = ref_enc(X_test)
        loss = criterion(y, Y_test)
        print(f"Evaluation MSELoss: {loss.item()}")
        
        eval_loss_list.append(loss.item())
        ref_enc.train()
        # plot loss with plotly
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(avg_loss_list))), y=avg_loss_list, mode='lines', name='train'))
        fig.add_trace(go.Scatter(x=list(range(len(eval_loss_list))), y=eval_loss_list, mode='lines', name='eval'))
        fig.update_layout(title='Loss', xaxis_title='Epoch', yaxis_title='Loss')
        fig.write_html(f"ref_loss.html")
        torch.save(ref_enc.state_dict(), f"ref_enc.pth")

        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader)}")
        
        
    
        
    