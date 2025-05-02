import sys


sys.path.append(".")
from utils.pinn_network import DNN
import numpy as np
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
from utils.function import *
from pde.Wave import config


elementType = 'Sine'
# elementType = 'Gauss'


device, x_min, x_max, t_min, t_max, xt_0, u_0, u_t_0, xt_bc, u_bc, xt_f, u_f = config.config()


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
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(20,1)
        self.act2 = torch.nn.Identity()

    def forward(self,xt):
        x = xt[:,0:1][...,None]
        t = xt[:,1:2][...,None]
        out = (self.A1*torch.sin(self.w1*(x-t))+self.A2*torch.sin(self.w1*(x+t)))+self.b
        return self.act2(self.fc2(self.act1(self.fc1(out.view(out.shape[0], -1)))))

class GaussGridElement(torch.nn.Module):
    def __init__(self,element_x=5,element_t=5,range_x=2,range_t=1):
        super().__init__()
        # self.A = torch.nn.Parameter(torch.rand((1,1,1,element_x),dtype=torch.float32),requires_grad=True)
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
        out = (self.A1*torch.exp(-((x-self.mu_x-(t-self.mu_t))/(self.sigma_x_1*self.sigma_t_1))**2)+
               self.A2*torch.exp(-((x-self.mu_x+(t-self.mu_t))/(self.sigma_x_1)*self.sigma_t_1)**2)+self.b)
        return self.act2(self.fc2(self.act1(self.fc1(out.view(out.shape[0],-1)))))

def equation(xt):
    x = xt[:, 0:1].cpu().numpy()
    t = xt[:, 1:2].cpu().numpy()
    return np.where(x-t<1,(x-t)/2,1-x-t/2)

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
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(),lr=1e-3)
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
        mse_pde = self.ms(pde-u_f)
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
        print(f"\r{self.iter}, Loss : {loss.item():.5e}, ic : {mse_0:.3e}, bc : {mse_bc:.3e}, f : {mse_pde:.3e}",end="")
        if self.iter % 500 == 0:
            torch.save(self.net.state_dict(), "checkpoint/" + elementType + "_weight.pt")
            print("")
        return loss.item()

if __name__ == "__main__":
    gen = GEN()
    loss_update = []
    for i in range(200000):
        loss_update.append(gen.closure())
        gen.adam.step()
    # pinn.lbfgs.step(pinn.closure)
    ####################### Plot ########################
    x = np.arange(x_min, x_max, 0.01)
    t = np.arange(t_min, t_max+2, 0.01)

    X, T = np.meshgrid(x, t)

    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    xt_f = np.hstack([X, T])
    xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)

    u_pred = gen.net(xt_f)
    u_pred = u_pred.detach().cpu().numpy()
    u_pred = u_pred.reshape(len(x), len(t), order="F")

    # torch.save(gen.net.state_dict(), "../Wave/weight.pt")
    res = {'loss': loss_update, 'x': x, 't': t, 'u': u_pred}
    np.save("../Wave/res/" + elementType + "_res.npy", res)

    im = plt.imshow(u_pred, cmap='BrBG')

    plt.xticks([i * t.size // 14 for i in range(14)],
               ['{:.02f}'.format(i * 0.5) for i in range(14)])  # 自定义 x 轴刻度和标签
    plt.yticks([i * x.size // 4 for i in range(5)],
               ['{:.02f}'.format(i * 0.5) for i in range(5)])  # 自定义 y 轴刻度和标签
    add_colorbar(im)
    plt.grid(color='white', linestyle='--', linewidth=0.5)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(elementType+'GEN')
    plt.savefig("../Wave/res/"+elementType+"GEN.pdf")
    plt.show(block=True)
