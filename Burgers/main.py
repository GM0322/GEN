import sys
sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from utils.pinn_network import DNN
from pde.Burgers import config


"""
Burgers Eqn.
f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx = 0, x ~ [-1, 1], t ~ [0, 1]
lambda_1 = 1
lambda_2 = 0.01/pi = 0.0031831
"""

elementType = 'Sine'
device, x, t, u, ub, lb, xt, noisy_u, u_ = config.config()


class Sine1DElement(torch.nn.Module):
    def __init__(self,elements=25,period=torch.pi):
        super().__init__()
        self.lambda_1 = torch.tensor([1.0]).to(device)
        self.lambda_2 = torch.tensor([0.01/torch.pi]).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1, requires_grad=False)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2, requires_grad=False)
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
        self.lambda_1 = torch.tensor([1.0]).to(device)
        self.lambda_2 = torch.tensor([0.01/torch.pi]).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1, requires_grad=False)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2, requires_grad=False)
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
               self.A2*torch.exp(-((x-self.mu_x+(t-self.mu_t))/(self.sigma_x_1*self.sigma_t_1))**2)+self.b)
        return self.act2(self.fc2(self.act1(self.fc1(out.view(out.shape[0],-1)))))


class GEN:
    def __init__(self, u, elementType='Sine',elements=100):
        self.u = u
        if elementType == 'Sine':
            self.net = Sine1DElement(elements=elements).to(device)
        elif elementType == 'Gauss':
            self.net = GaussGridElement().to(device)
        else:
            self.net=None
            assert 'element type not defined'

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),lr=5e-3)
        # self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.LBFGS(
        #     self.net.parameters(),
        #     lr=1.0,
        #     max_iter=50000,
        #     max_eval=50000,
        #     history_size=50,
        #     tolerance_grad=1e-5,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe",
        # )
        self.iter = 0

    def f(self, xt):
        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]

        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        f = u_t + self.net.lambda_1 * u * u_x - self.net.lambda_2 * u_xx
        return f

    def closure(self):
        self.optimizer.zero_grad()

        u_pred = self.net(xt)
        f_pred = self.f(xt)

        mse_u = torch.mean(torch.square(u_pred - self.u))
        mse_f = torch.mean(torch.square(f_pred))

        loss = mse_u + mse_f
        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.net.lambda_1.item():.7f}, l2 : {self.net.lambda_2.item():.7f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss.item()


def calcError(pinn):
    u_pred = pinn.net(torch.hstack((x, t)))
    u_pred = u_pred.detach().cpu().numpy()
    u_ = u.detach().cpu().numpy()
    error_u = np.linalg.norm(u_ - u_pred, 2) / np.linalg.norm(u_, 2)
    lambda1 = pinn.lambda_1.detach().cpu().item()
    lambda2 = np.exp(pinn.lambda_2.detach().cpu().item())
    error_lambda1 = np.abs(lambda1 - 1.0) * 100
    error_lambda2 = np.abs(lambda2 - 0.01 / np.pi) * 100
    print(
        f"\nError u  : {error_u:.5e}",
        f"\nError l1 : {error_lambda1:.5f}%",
        f"\nError l2 : {error_lambda2:.5f}%",
    )
    return (error_u, error_lambda1, error_lambda2)


if __name__ == "__main__":
    gen = GEN(u,elementType=elementType,elements=25)
    loss_update = []
    for epoch in range(100000):
    #     loss_update.append(gen.closure())
    #     gen.optimizer.step()
        gen.optimizer.step(gen.closure)
    torch.save(gen.net.state_dict(), "../Burgers/checkpoint/"+elementType+"_weight_clean25.pt")
    gen.net.load_state_dict(torch.load("../Burgers/checkpoint/"+elementType+"_weight_clean25.pt",weights_only=False))
    # calcError(gen)

    gen = GEN(noisy_u,elementType=elementType)
    gen.optimizer.step(gen.closure)
    loss_update = []
    for epoch in range(100000):
    #     loss_update.append(gen.closure())
    #     gen.optimizer.step()
        gen.optimizer.step(gen.closure)
    torch.save(gen.net.state_dict(), "../Burgers/checkpoint/"+elementType+"_weight_noisy25.pt")
    # calcError(gen)
