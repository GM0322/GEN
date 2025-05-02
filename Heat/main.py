import sys

sys.path.append(".")
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.function import *
from matplotlib import offsetbox
from torch.autograd import grad
from matplotlib.animation import FuncAnimation
from pde.Heat import config


# elementType = 'Sine'
elementType = 'Gauss'

device,x_min,x_max,t_min,t_max,xt_0,u_0,xt_bc,u_bc,xt_f,u_f = config.config()


class Sine1DElement(torch.nn.Module):
    def __init__(self,elements=25,period=torch.pi):
        super().__init__()
        self.period = period
        self.A1 = torch.nn.Parameter(torch.rand((1,1,elements),dtype=torch.float32),requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand((1,1,elements),dtype=torch.float32),requires_grad=True)
        w1 = torch.rand((1,1,elements),dtype=torch.float32)
        w2 = torch.tensor([i for i in range(elements)],dtype=torch.float32).view(1,1,-1)*period
        self.w1 = torch.nn.Parameter(w1*w2,requires_grad=True)
        w1 = torch.rand((1,1,elements),dtype=torch.float32)
        self.w2 = torch.nn.Parameter(w1*w2,requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.rand((1,1,1),dtype=torch.float32),requires_grad=True)
        self.b = torch.nn.Parameter(torch.rand((1,1,1),dtype=torch.float32),requires_grad=True)
        self.fc1 = torch.nn.Linear(elements,20)
        self.act1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(20,1)
        self.act2 = torch.nn.Identity()

    def forward(self,xt):
        x = xt[:,0:1][...,None]
        t = xt[:,1:2][...,None]
        out = torch.exp(-(self.w1)**2*t)*(self.A1*torch.sin(self.w1*x))+self.b
        return self.act2(self.fc2(self.act1(self.fc1(out.view(out.shape[0], -1))))),out

class GaussGridElement(torch.nn.Module):
    def __init__(self,element_x=5,element_t=5,range_x=2,range_t=1):
        super().__init__()
        self.A1 = torch.nn.Parameter(torch.rand((1, 1,element_t,element_x), dtype=torch.float32), requires_grad=True)
        self.A2 = torch.nn.Parameter(torch.rand((1, 1,element_t,element_x), dtype=torch.float32), requires_grad=True)
        mu_x = torch.rand((1,1,1,element_x),dtype=torch.float32)
        mu_t = torch.rand((1,1,element_t,1),dtype=torch.float32)
        self.sigma_x_1 = torch.nn.Parameter(torch.rand((1,1,1,element_x),dtype=torch.float32), requires_grad=True)
        self.sigma_t_1 = torch.nn.Parameter(torch.rand((1,1,element_t,1),dtype=torch.float32), requires_grad=True)
        self.sigma_x_2 = torch.nn.Parameter(torch.rand((1,1,1,element_x),dtype=torch.float32), requires_grad=True)
        self.sigma_t_2 = torch.nn.Parameter(torch.rand((1,1,element_t,1),dtype=torch.float32), requires_grad=True)
        wt = torch.tensor([i for i in range(element_t)],dtype=torch.float32).view(1,1,-1,1)*range_t/element_t
        wx = torch.tensor([i for i in range(element_x)],dtype=torch.float32).view(1,1,1,-1)*range_x/element_x
        self.mu_x = torch.nn.Parameter(mu_x*wx, requires_grad=True)
        self.mu_t = torch.nn.Parameter(mu_t*wt, requires_grad=True)
        self.b = torch.nn.Parameter(torch.rand((1,1,1,1),dtype=torch.float32),requires_grad=True)
        self.fc1 = torch.nn.Linear(element_t*element_x,20)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(20,1)
        self.act2 = torch.nn.Identity()

    def forward(self,xt):
        x = xt[:,0:1][...,None,None]
        t = xt[:,1:2][...,None,None]
        out = torch.exp(-(1.0 * self.mu_t) ** 2 * t) * self.A1 * torch.exp(-((x-self.mu_x) / self.sigma_x_1) ** 2) + self.b
        return self.act2(self.fc2(self.act1(self.fc1(out.view(out.shape[0],-1))))),out


class GEN:
    c = 1.0
    def __init__(self,elementType=elementType) -> None:
        if elementType == 'Sine':
            self.net = Sine1DElement(elements=25).to(device)
        elif elementType == 'Gauss':
            self.net = GaussGridElement().to(device)
        else:
            self.net=None
            assert 'element type not defined'
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

        u,_ = self.net(xt)
        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        f = u_t - self.c ** 2 * u_xx
        return f

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        # initial condition, u(x, 0) = f(x)
        u0_pred,_ = self.net(xt_0)
        mse_0 = torch.mean(torch.square(u0_pred - u_0))
        u_bc_pred,_ = self.net(xt_bc)
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
            torch.save(self.net.state_dict(), "checkpoint/" + elementType + "_weight.pt")
            print("")
        return loss.item()

def train():
    gen = GEN()
    loss_update = []
    for i in range(100000):
        loss_update.append(gen.closure())
        gen.adam.step()
    # gen.lbfgs.step(gen.closure)

    ####################### Plot ########################
    x = np.arange(x_min, x_max, 0.01)
    t = np.arange(t_min, t_max+0.5, 0.01)

    X, T = np.meshgrid(x, t)

    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    xt_f = np.hstack([X, T])
    xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)

    u_pred,_ = gen.net(xt_f)
    u_pred = u_pred.detach().cpu().numpy()

    u_pred = u_pred.reshape(len(x), len(t), order="F")

    # fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    # bar_x = np.linspace(x_min, x_max, u_pred.shape[0])
    # bar_y = np.zeros_like(bar_x)
    # axes[0].set_title("1D Heat Equation")
    # axes[0].set_yticks([])
    #
    # im0 = axes[0].scatter(
    #     x,
    #     np.zeros_like(x),
    #     c=u_pred[:, 0],
    #     marker="s",
    #     cmap="rainbow",
    #     lw=0,
    #     vmin=u_pred.min() * 1.1,
    #     vmax=u_pred.max() * 1.1,
    # )
    # fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, extend="neither")
    # textbox = offsetbox.AnchoredText("", loc=1)
    # axes[0].add_artist(textbox)
    #
    # (im1,) = axes[1].plot([], [], color="k")
    # axes[1].set_ylim(u_pred.min() * 1.3, u_pred.max() * 1.3)
    # axes[1].set_xlabel("$x$")
    # axes[1].set_ylabel("Temperature $u(x,t)$")
    #
    # def update(frame):
    #     temp = u_pred[:, frame]
    #     im0.set_array(temp)
    #     im1.set_data(bar_x, temp)
    #     textbox = offsetbox.AnchoredText(f"{t[frame]:.2f} sec", loc=1)
    #     axes[0].add_artist(textbox)

    # ani = FuncAnimation(fig, update, frames=len(t), interval=50)
    # ani.save("../Heat/solution.gif", dpi=100)
    # ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu','RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])
    # plt.imshow(u_pred,cmap='PiYG')

    # res = {'loss':loss_update,'x':x,'t':t,'u':u_pred}
    # np.save("../Heat/res/"+elementType+"_res.npy",res)
    # im = plt.imshow((u_pred),cmap='seismic')
    # plt.xticks([i*t.size//5 for i in range(6)], ['{:.02f}'.format(i*(t_max+0.5)*0.2) for i in range(6)])  # 自定义 x 轴刻度和标签
    # plt.yticks([i*x.size//10 for i in range(11)], ['{:.02f}'.format(i*x_max*0.1) for i in range(11)])  # 自定义 y 轴刻度和标签
    # add_colorbar(im)
    # plt.grid(color='white', linestyle='--', linewidth=0.5)
    # plt.xlabel('$t$')
    # plt.ylabel('$x$')
    # plt.title(elementType+'GEN')
    # plt.savefig("../Heat/res/"+elementType+"solution.pdf")
    # plt.show(block=True)

if __name__ == "__main__":
    train()