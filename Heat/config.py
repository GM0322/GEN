import numpy as np
import torch


def config():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    N_0 = 200
    N_bc = 400
    N_f = 10000

    x_min = 0.0
    x_max = 2.0
    t_min = 0.0
    t_max = 2.0

    ub = np.array([x_max, t_max])
    lb = np.array([x_min, t_min])
    xt_0 = np.random.uniform([x_min, 0], [x_max, 0], size=(N_0, 2))
    u_0 = np.sin(np.pi/2*xt_0[:, 0:1])
    # u_0 = np.ones_like(xt_0[:, 0:1]) * 5

    # Boundary Condition
    # u(0, t) = 0 & u(L, t) = 0 for t > 0

    xt_bc_0 = np.random.uniform([x_min, t_min], [x_min, t_max], size=(N_bc // 2, 2))
    u_bc_0 = np.ones((len(xt_bc_0), 1)) * 0

    xt_bc_1 = np.random.uniform([x_max, t_min], [x_max, t_max], size=(N_bc // 2, 2))
    u_bc_1 = np.zeros((len(xt_bc_1), 1))

    xt_bc = np.vstack([xt_bc_0, xt_bc_1])
    u_bc = np.vstack([u_bc_0, u_bc_1])

    xt_f = np.random.uniform(lb, ub, (N_f, 2))
    xt_f = np.vstack([xt_0, xt_bc, xt_f])
    u_f = np.zeros((N_f,1))
    u_f = np.vstack([u_0, u_bc, u_f])


    # Convert to Tensor
    xt_0 = torch.tensor(xt_0, dtype=torch.float32).to(device)
    u_0 = torch.tensor(u_0, dtype=torch.float32).to(device)

    xt_bc = torch.tensor(xt_bc, dtype=torch.float32).to(device)
    u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)

    xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)
    u_f = torch.tensor(u_f, dtype=torch.float32).to(device)
    return device, x_min,x_max,t_min,t_max,xt_0,u_0,xt_bc,u_bc,xt_f,u_f