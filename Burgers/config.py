import torch
import numpy as np
from scipy.io import loadmat

def config(train=True):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    if train:
        e = 0.1
    else:
        e = 0

    N_u = 2000

    data = loadmat("../Burgers/burgers_shock.mat")
    x = data["x"]
    t = data["t"]
    u = data["usol"].T

    ub = np.array([x.max()-e, t.max()-e])
    lb = np.array([x.min(), t.min()])

    # Clean Data Preparation
    x_, t_ = np.meshgrid(x, t)
    x_ = x_.reshape(-1, 1)
    t_ = t_.reshape(-1, 1)
    u_ = u.reshape(-1, 1)

    rand_idx = np.random.choice(len(u_), N_u, replace=False)

    x = torch.tensor(x_[rand_idx], dtype=torch.float32).to(device)
    t = torch.tensor(t_[rand_idx], dtype=torch.float32).to(device)
    xt = torch.cat((x, t), dim=1)
    u = torch.tensor(u_[rand_idx], dtype=torch.float32).to(device)

    # 1% Noisy Data Preparation
    noise = 0.1
    noisy_u = u_ + noise * np.std(u_) * np.random.randn(*u_.shape)
    noisy_u = torch.tensor(noisy_u[rand_idx], dtype=torch.float32).to(device)
    return device, x, t, u, ub, lb, xt, noisy_u, u_
