import torch

def load_critic(path):
    critic = torch.load(path)
    return critic



def get_full_state(critic):
    N=10
    for i in range(100):
        UK_RL = []
        UK_RICCATI = []
        for k in range(N):
            x = torch.tensor([0.01*i-0.5,k])
            uk_rl = critic.select_action(x)
            uk_riccati = -(1/(N-k+2))*x[0]
            print(f"RL: {uk_rl.item()}, RICCATI: {uk_riccati}, k: {k}")
            UK_RL.append(uk_rl.item())
            UK_RICCATI.append(uk_riccati)
       

