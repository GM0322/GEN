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

device, x, t, u, ub, lb, xt, noisy_u, u_ = config.config()


class PINN:
    def __init__(self, u,device=device):
        self.u = u
        self.lambda_1 = torch.tensor([1.0], requires_grad=False).to(device)
        self.lambda_2 = torch.tensor([0.01/torch.pi], requires_grad=False).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.net = DNN(dim_in=2, dim_out=1, n_layer=9, n_node=20, ub=ub, lb=lb,device=device).to(
            device
        )
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)
        self.net.lambda_1.requires_grad = False
        self.net.lambda_2.requires_grad = False
        self.adam = torch.optim.Adam(self.net.parameters(),lr=1e-3)
        # self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1e-3)
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
        self.iter = 0

    def f(self, xt):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        xt = xt.clone()
        xt.requires_grad = True

        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]

        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        u_pred = self.net(xt)
        f_pred = self.f(xt)

        mse_u = torch.mean(torch.square(u_pred - self.u))
        mse_f = torch.mean(torch.square(f_pred))

        loss = mse_u + mse_f
        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.lambda_1.item():.5f}, l2 : {self.lambda_2.item():.5f}",
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
        f"\nError l1 : {error_lambda1:.7f}%",
        f"\nError l2 : {error_lambda2:.7f}%",
    )
    return (error_u, error_lambda1, error_lambda2)


if __name__ == "__main__":
    pinn = PINN(u)
    loss_update = []
    for epoch in range(100000):
        loss_update.append(pinn.closure())
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "checkpoint/pinn_weight_clean.pt")
    pinn.net.load_state_dict(torch.load("checkpoint/pinn_weight_clean.pt", weights_only=False))
    calcError(pinn)

    pinn = PINN(noisy_u)
    loss_update = []
    for epoch in range(100000):
        loss_update.append(pinn.closure())
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "checkpoint/pinn_weight_noisy1.pt")
    calcError(pinn)
