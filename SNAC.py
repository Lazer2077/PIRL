import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数设置 ----------
N = 30                  # 时间步长
n_state = 1             # 状态维度
n_control = 1           # 控制维度
n_basis = 4             # 基函数数量
R = np.eye(n_control)
Q = np.eye(n_state)
Qf = np.eye(n_state)

# ---------- 理论最优解 ----------
def lambda_theoretical(xk, k):
    return (1 / (N - k + 2)) * xk

# ---------- 系统模型 ----------
def f(x):
    return x  # x_{k+1} = x_k + u_k, u=0

def g(x):
    return np.array([[1.0]])

# ---------- 基函数 ----------
def phi(x, tau):
    x = float(x)
    return np.array([x / (tau + 2),  x * tau, x, tau])

# ---------- 网络预测 ----------
def predict_lambda(x, tau, W):
    return float(W.T @ phi(x, tau))

# ---------- SNAC 训练 ----------
def train(W_init, num_iters=1000, samples_per_iter=20):
    W = W_init.copy()
    error_history = []

    for iter in range(num_iters):
        Phi_all = []
        Lambda_all = []
        total_error = 0.0

        for _ in range(samples_per_iter):
            xk = np.random.uniform(-1, 1)
            k = np.random.randint(0, N - 1)
            tau = N - k

            # step 1: predict λ_{k+2}
            xk1 = f(xk)  # + u = 0
            lambda_kplus2 = predict_lambda(xk1, N - (k + 1), W)

            Qx = float(Q @ np.array([[xk1]]))
            A = 1.0  # df/dx
            lambda_kplus1 = Qx + A * lambda_kplus2

            # step 2: terminal λ_N
            xN = np.random.uniform(-2, 2)
            lambda_N = float(Qf @ np.array([[xN]]))

            # step 3: phi + target
            phi1 = phi(xk, tau)
            phi2 = phi(xN, 1)
            phi_bar = np.vstack([phi1, phi2]).T
            lambda_bar = np.array([[lambda_kplus1, lambda_N]])

            lambda_pred = W.T @ phi_bar
            error = np.linalg.norm(lambda_pred - lambda_bar)**2
            total_error += error

            Phi_all.append(phi_bar)
            Lambda_all.append(lambda_bar)

        # 最小二乘更新
        Phi_all = np.concatenate(Phi_all, axis=1)
        Lambda_all = np.concatenate(Lambda_all, axis=1)
        W = np.linalg.pinv(Phi_all @ Phi_all.T) @ Phi_all @ Lambda_all.T

        mean_error = total_error / samples_per_iter
        error_history.append(mean_error)
        print(f"Iter {iter + 1:02d}, Training Error: {mean_error:.6e}")
    print(W.shape)
    
    return W, error_history
if __name__ == "__main__":
    # 初始化权重并训练
    W_init = np.zeros((n_basis, n_state))
    W_trained, error_history = train(W_init)
    # W_trained[0,0] =1 
    # W_trained[1,0] =0 

    # 绘制训练误差
    plt.figure(figsize=(6, 3))
    plt.plot(error_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Training Error (Example 1)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 对比预测与理论 λ
    x_vals = np.linspace(-1, 1, 100)
    k = 3
    tau = N - k
    pred_vals = [predict_lambda(x, tau, W_trained) for x in x_vals]
    true_vals = [lambda_theoretical(x, k) for x in x_vals]
    x_test = x_vals[1]
    
    
    
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, true_vals, label="Theoretical λ", linewidth=2)
    plt.plot(x_vals, pred_vals, '--', label="Predicted λ", linewidth=2)
    plt.xlabel("x₀")
    plt.ylabel("λ₁")
    plt.title("Costate Function λ₁(x₀)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("SNAC.png")

    # 误差评估
    abs_error = np.abs(np.array(pred_vals) - np.array(true_vals))
    rel_error = abs_error / (np.abs(true_vals) + 1e-8)
    print(f"最大相对误差: {np.max(rel_error) * 100:.4f}%")
    print(f"平均相对误差: {np.mean(rel_error) * 100:.4f}%")
