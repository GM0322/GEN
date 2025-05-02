import numpy as np
import torch

def config():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Parameters
    x_min = 0
    x_max = 2
    t_min = 0
    t_max = 5

    ub = np.array([x_max, t_max])
    lb = np.array([x_min, t_min])

    N_0 = 100
    N_bc = 300
    N_f = 10000

    ## IC , u(x,0) = 0
    x_0 = np.random.uniform(x_min, x_max, (N_0, 1))
    t_0 = np.zeros((N_0, 1))
    xt_0 = np.hstack([x_0, t_0])
    u_0 = np.where(x_0 < 1, x_0 / 2, 1 - x_0 / 2)
    u_t_0 = np.zeros((N_0, 1))

    # BC : Fixed end points
    x_bc = np.random.choice([x_min, x_max], size=(N_bc, 1))
    t_bc = np.random.uniform(t_min, t_max, (N_bc, 1))
    xt_bc = np.hstack([x_bc, t_bc])
    u_bc = np.zeros((N_bc, 1))

    # Collocation points
    xt_f = np.random.uniform(lb, ub, (N_f, 2))
    u_f = np.zeros((N_f, 1))  # +0.1*np.random.randn(N_f,1)

    # Convert to tensor
    xt_0 = torch.tensor(xt_0, dtype=torch.float32).to(device)
    u_0 = torch.tensor(u_0, dtype=torch.float32).to(device)
    u_t_0 = torch.tensor(u_t_0, dtype=torch.float32).to(device)

    xt_bc = torch.tensor(xt_bc, dtype=torch.float32).to(device)
    u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)

    xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)
    u_f = torch.tensor(u_f, dtype=torch.float32).to(device)

    # N_0 = 100
    # N_bc = 300
    # N_f = 10000
    #
    # ## IC , u(x,0) = 0
    # x_0 = np.linspace(x_min, x_max,N_0).reshape((N_0, 1))
    # t_0 = np.zeros((N_0, 1))
    # xt_0 = np.hstack([x_0, t_0])
    # u_0 = np.where(x_0 < 1, x_0 / 2, 1 - x_0 / 2)
    # u_t_0 = np.zeros((N_0, 1))
    #
    # # BC : Fixed end points
    # x_bc = np.random.choice([x_min, x_max], size=(N_bc, 1))
    # # t_bc = np.random.uniform(t_min, t_max, (N_bc, 1))
    # t_bc = np.linspace(t_min, t_max, N_bc).reshape((N_bc, 1))
    # xt_bc = np.hstack([x_bc, t_bc])
    # u_bc = np.zeros((N_bc, 1))
    #
    # # Collocation points
    # # xt_f = np.random.uniform(lb, ub, (N_f, 2))
    # xt_f_x = np.random.choice(x_0[:,0],size=(N_f,1))
    # xt_f_t = np.random.choice(t_bc[:,0],size=(N_f,1))
    # xt_f = np.concatenate([xt_f_x,xt_f_t],axis=1)
    #
    # u_f = np.zeros((N_f, 1))  # +0.1*np.random.randn(N_f,1)
    #
    # # Convert to tensor
    # xt_0 = torch.tensor(xt_0, dtype=torch.float32).to(device)
    # u_0 = torch.tensor(u_0, dtype=torch.float32).to(device)
    # u_t_0 = torch.tensor(u_t_0, dtype=torch.float32).to(device)
    #
    # xt_bc = torch.tensor(xt_bc, dtype=torch.float32).to(device)
    # u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)
    #
    # xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)
    # u_f = torch.tensor(u_f, dtype=torch.float32).to(device)
    return device,x_min,x_max,t_min,t_max,xt_0,u_0,u_t_0,xt_bc,u_bc,xt_f,u_f