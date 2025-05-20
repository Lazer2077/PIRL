import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数 ----------
N = 100
Q = 1.0
Qf = 1.0

device = torch.device("cpu")

# ---------- 定义 MLP 网络 ----------
class LambdaMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_tau):  # 输入: [x, tau]
        return self.net(x_tau)

# ---------- λ 理论解 ----------
def lambda_theoretical(x, k):
    return 1 / (N - k + 2) * x

# ---------- 训练网络 ----------
def train(model, num_epochs=15000, batch_size=128, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_history = []

    for epoch in range(num_epochs):
        xk = torch.FloatTensor(batch_size, 1).uniform_(-1, 1).to(device)
        k = torch.randint(0, N - 1, (batch_size, 1)).to(torch.float32).to(device)
        tau = N - k

        # Forward simulate: x_{k+1} = x_k
        xk1 = xk.clone()
        tau_k1 = tau - 1

        with torch.no_grad():
            lambda_kplus2 = model(torch.cat([xk1, tau_k1], dim=1))

        Qx = Q * xk1
        lambda_kplus1 = Qx + lambda_kplus2  # A = 1

        # Terminal λ_N
        xN = torch.FloatTensor(batch_size, 1).uniform_(-1, 1).to(device)
        lambda_N = Qf * xN

        # 构造训练样本
        x_input = torch.cat([xk, tau], dim=1)
        xN_input = torch.cat([xN, torch.ones_like(xN)], dim=1)

        inputs = torch.cat([x_input, xN_input], dim=0)
        targets = torch.cat([lambda_kplus1, lambda_N], dim=0)

        preds = model(inputs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.6e}")

    return loss_history

# ---------- 主程序 ----------
if __name__ == "__main__":
    model = LambdaMLP().to(device)
    loss_history = train(model)

    # 画 loss 曲线
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP-SNAC Training Loss (Example 1)")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 对比预测 λ vs 理论 λ
    x_vals = np.linspace(-1, 1, 100)
    tau = N - 2
    x_tensor = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)
    tau_tensor = torch.full_like(x_tensor, tau)
    input_tensor = torch.cat([x_tensor, tau_tensor], dim=1)

    with torch.no_grad():
        lambda_pred = model(input_tensor).squeeze().numpy()

    lambda_true = np.array([lambda_theoretical(x, 2) for x in x_vals])

    plt.plot(x_vals, lambda_true, label="Theoretical λ")
    plt.plot(x_vals, lambda_pred, '--', label="MLP Predicted λ")
    plt.xlabel("x₀")
    plt.ylabel("λ₁")
    plt.title("MLP-SNAC Prediction vs Theory")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    error = np.abs(lambda_pred - lambda_true)
    print(f"最大相对误差: {np.max(error / (np.abs(lambda_true) + 1e-8)) * 100:.4f}%")
    print(f"平均相对误差: {np.mean(error / (np.abs(lambda_true) + 1e-8)) * 100:.4f}%")
