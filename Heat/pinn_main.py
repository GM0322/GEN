import sys

sys.path.append("")

from utils.pinn_network import DNN
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.function import *

from matplotlib import offsetbox
from torch.autograd import grad
from matplotlib.animation import FuncAnimation
from pde.Heat import config

device,x_min,x_max,t_min,t_max,xt_0,u_0,xt_bc,u_bc,xt_f,u_f = config.config()
ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

class PINN:
    c = 1.0
    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=1, n_layer=7, n_node=40, ub=ub, lb=lb,device=device).to(
            device
        )
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            history_size=50,
            max_eval=10000,
            max_iter=10000,
        )
        self.adam = torch.optim.Adam(self.net.parameters(),lr=1e-3)
        self.iter = 0

    def f(self, xt):
        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)
        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        f = u_t - self.c ** 2 * u_xx
        return f

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        u0_pred = self.net(xt_0)
        mse_0 = torch.mean(torch.square(u0_pred - u_0))
        u_bc_pred = self.net(xt_bc)
        mse_bc = torch.mean(torch.square(u_bc_pred - u_bc))

        f_pred = self.f(xt_f)
        mse_f = torch.mean(torch.square(f_pred - u_f))
        loss = mse_0 + mse_bc + mse_f
        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter}, Loss : {loss.item():.5e}, ic : {mse_0:.3e}, bc : {mse_bc:.3e}, f : {mse_f:.3e}",
            end="",
        )

        if self.iter % 500 == 0:
            print("")
        return loss.item()


pinn = PINN()
loss_update = []
for i in range(100000):
    loss_update.append(pinn.closure())
    pinn.adam.step()
# pinn.lbfgs.step(pinn.closure)

####################### Plot ########################
x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max+0.5, 0.01)

X, T = np.meshgrid(x, t)

X = X.reshape(-1, 1)
T = T.reshape(-1, 1)
xt_f = np.hstack([X, T])
xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)

u_pred = pinn.net(xt_f)
u_pred = u_pred.detach().cpu().numpy()

u_pred = u_pred.reshape(len(x), len(t), order="F")

res = {'loss':loss_update,'x':x,'t':t,'u':u_pred}
np.save("../Heat/res/pinn_res.npy", res)

im= plt.imshow((u_pred),cmap='seismic')
plt.xticks([i*t.size//5 for i in range(6)], ['{:.02f}'.format(i*(t_max+0.5)*0.2) for i in range(6)])  # 自定义 x 轴刻度和标签
plt.yticks([i*x.size//10 for i in range(11)], ['{:.02f}'.format(i*x_max*0.1) for i in range(11)])  # 自定义 y 轴刻度和标签
add_colorbar(im)
plt.grid(color='white', linestyle='--', linewidth=0.5)
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('PINN')
plt.savefig("../Heat/res/pinnsolution.pdf")
plt.show(block=True)

