import sys
sys.path.append(".")
from utils.pinn_network import DNN
import numpy as np
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
from utils.function import *
from pde.Wave import config

# elementType = 'Sine'
# elementType = 'Gauss'

device, x_min, x_max, t_min, t_max, xt_0, u_0, u_t_0, xt_bc, u_bc, xt_f, u_f = config.config()

ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])


class PINN:
    c = 1.0
    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=1, n_layer=5, n_node=40, ub=ub, lb=lb,device=device).to(device)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters())
        self.ms = lambda x: torch.mean(torch.square(x))
        self.iter = 0

    def loss_ic(self, xt):
        xt = xt.clone()
        xt.requires_grad = True
        u = self.net(xt)

        u_t = grad(u.sum(), xt, create_graph=True)[0][:, 1:2]
        mse_0 = self.ms(u - u_0) + self.ms(u_t - u_t_0)
        return mse_0

    def loss_bc(self, xt):
        u = self.net(xt)
        mse_bc = self.ms(u - u_bc)
        return mse_bc

    def loss_pde(self, xt):
        xt = xt.clone()
        xt.requires_grad = True
        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True, retain_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        u_t = u_xt[:, 1:2]
        u_tt = grad(u_t.sum(), xt, create_graph=True)[0][:, 1:2]

        pde = u_tt - (self.c ** 2) * (u_xx)
        mse_pde = self.ms(pde - u_f)
        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()
        mse_0 = self.loss_ic(xt_0)
        mse_bc = self.loss_bc(xt_bc)
        mse_pde = self.loss_pde(xt_f)

        # collocation points
        loss = mse_0 + mse_bc + mse_pde

        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter}, Loss : {loss.item():.5e}, ic : {mse_0:.3e}, bc : {mse_bc:.3e}, f : {mse_pde:.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            torch.save(self.net.state_dict(), "checkpoint/pinn_weight.pt")
            print("")

        return loss.item()

if __name__ == "__main__":
    pinn = PINN()
    loss_update = []
    for i in range(200000):
        loss_update.append(pinn.closure())
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    ####################### Plot ########################
    x = np.arange(x_min, x_max, 0.01)
    t = np.arange(t_min, t_max+2, 0.01)

    X, T = np.meshgrid(x, t)

    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    xt_f = np.hstack([X, T])
    xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)

    u_pred = pinn.net(xt_f)
    u_pred = u_pred.detach().cpu().numpy()

    u_pred = u_pred.reshape(len(x), len(t), order="F")


    # torch.save(pinn.net.state_dict(), "../Wave/weight.pt")
    res = {'loss': loss_update, 'x': x, 't': t, 'u': u_pred}
    np.save("../Wave/res/pinn_res.npy", res)
    im = plt.imshow(u_pred, cmap='BrBG')
    plt.xticks([i * t.size // 14 for i in range(15)],
               ['{:.02f}'.format(i * 0.5) for i in range(15)])  # 自定义 x 轴刻度和标签
    plt.yticks([i * x.size // 4 for i in range(5)],
               ['{:.02f}'.format(i * 0.5) for i in range(5)])  # 自定义 y 轴刻度和标签
    add_colorbar(im)
    plt.grid(color='white', linestyle='--', linewidth=0.5)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('PINN')
    plt.savefig("../Wave/res/pinn.pdf")
    plt.show(block=True)
