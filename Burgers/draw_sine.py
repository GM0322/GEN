import torch
from pde.Burgers.main import GEN
from pde.Burgers.pinn_main import PINN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat


pgf_with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(pgf_with_latex)
np.random.seed(1234)
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

elementType = 'Sine'

data = loadmat("../Burgers/burgers_shock.mat")
x = data["x"]
t = data["t"]
u = data["usol"]

x_, t_ = np.meshgrid(x, t)
x_ = x_.reshape(-1, 1)
t_ = t_.reshape(-1, 1)
u_ = u.reshape(-1, 1)

rand_idx = np.random.choice(len(u_), 2000, replace=False)
x_train = x_[rand_idx]
t_train = t_[rand_idx]

# if elementType == 'pinn':
#     net = PINN(None,device=device)
# elif elementType == 'Sine':
#     net = GEN(u,elementType=elementType,elements=25)
# elif elementType == 'Gauss':
#     net = GEN(u,elementType=elementType)
# else:
#     net = None
#     assert 'element type not defined'


net1 = GEN(u, elementType=elementType, elements=25)
net2 = GEN(u, elementType=elementType, elements=100)
with torch.no_grad():
    x_ts = torch.tensor(x_, dtype=torch.float32).to(device)
    t_ts = torch.tensor(t_, dtype=torch.float32).to(device)
    net1.net.load_state_dict(torch.load("../Burgers/checkpoint/"+elementType+"_weight_clean25.pt",weights_only=False,map_location=device))
    net2.net.load_state_dict(torch.load("../Burgers/checkpoint/"+elementType+"_weight_clean.pt",weights_only=False,map_location=device))
    net1.net = net1.net.to(device)
    net2.net = net2.net.to(device)
    u_pred1 = net1.net(torch.hstack((x_ts, t_ts))).cpu().numpy().reshape((100, 256)).T
    u_pred2 = net2.net(torch.hstack((x_ts, t_ts))).cpu().numpy().reshape((100, 256)).T

################ Plot ###################
fig = plt.figure(figsize=(10, 6))
plt.rcParams.update({
    'font.size': 15,                  # 基础字体大小
    'axes.titlesize': 17,             # 子图标题
    'axes.labelsize': 15,             # 坐标轴标签
    'xtick.labelsize': 15,            # x轴刻度
    'ytick.labelsize': 15,            # y轴刻度
    'legend.fontsize': 13             # 图例
})


gs0 = GridSpec(1, 3, figure=fig)
gs0.update(top=0.94, bottom=0.70, left=0.05, right=0.95, wspace=0.15)

ax = fig.add_subplot(gs0[0, 0])
# ax = plt.subplots(3,2,1)
im1 = ax.imshow(
    u,
    extent=[t_.min(), t_.max(), x_.min(), x_.max()],
    cmap="rainbow",
    aspect="auto",
    interpolation="nearest",
    origin="lower",
)
ax.set_title("Exact")
ax = fig.add_subplot(gs0[0, 1])
im1 = ax.imshow(
    u_pred1,
    extent=[t_.min(), t_.max(), x_.min(), x_.max()],
    cmap="rainbow",
    aspect="auto",
    interpolation="nearest",
    origin="lower",
)
ax.set_title("GEN 25")
ax = fig.add_subplot(gs0[0, 2])
im2 = ax.imshow(
    u_pred2,
    extent=[t_.min(), t_.max(), x_.min(), x_.max()],
    cmap="rainbow",
    aspect="auto",
    interpolation="nearest",
    origin="lower",
)
ax.set_title("GEN 100")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)
cbar = fig.colorbar(im1, cax=cax)
fig.colorbar(im2, cax=cax)

gs1 = GridSpec(1, 3, figure=fig)
gs1.update(top=0.60, bottom=0.28, left=0.1, right=0.9, wspace=0.5)

t_slice = [0.25, 0.5, 0.75]
axes = []

for i in range(3):
    ax = fig.add_subplot(gs1[0, i])
    ax.plot(x, u[:, int(t_slice[i] * 100)], "b-", linewidth=3, label="Exact")
    ax.plot(x, u_pred1[:, int(t_slice[i] * 100)], "r--", linewidth=3, label="GEN 25")
    ax.plot(x, u_pred2[:, int(t_slice[i] * 100)], "g--", linewidth=3, label="GEN 100")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x,t)$")
    ax.set_title(f"$t = {t_slice[i]}$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    axes.append(ax)

    # 添加局部放大图（核心修改部分）
    # 1. 确定峰值位置
    peak_idx = np.argmax(u[:, int(t_slice[i] * 100)])
    peak_x = x[peak_idx]

    # 2. 创建放大子图坐标系
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])  # 位置参数：x,y,width,height
    axins.set_facecolor((0.7, 0.95, 1.0, 0.4))
    # 3. 绘制放大区域曲线
    axins.plot(x, u[:, int(t_slice[i] * 100)], "b-", linewidth=3)
    axins.plot(x, u_pred1[:, int(t_slice[i] * 100)], "r--", linewidth=3)
    axins.plot(x, u_pred2[:, int(t_slice[i] * 100)], "g--", linewidth=3)

    # 4. 设置放大范围
    axins.set_xlim(peak_x - 0.2, peak_x + 0.2)  # x轴放大范围
    axins.set_ylim(0.75, 1.05)  # y轴放大范围（根据数据调整）
    axins.set_xticks([])
    axins.set_yticks([])
    lines = axins.get_lines()
    lines[0].set_alpha(0.9)  # 精确解
    lines[1].set_alpha(0.7)  # GEN25
    lines[2].set_alpha(0.7)  # GEN100
    ax.indicate_inset_zoom(axins, edgecolor="#FF000F", alpha=0.6)

    # 1. 确定峰值位置
    peak_idx = np.argmin(u[:, int(t_slice[i] * 100)])
    peak_x = x[peak_idx]

    # 2. 创建放大子图坐标系
    axins = ax.inset_axes([0.05, 0.05, 0.4, 0.4])  # 位置参数：x,y,width,height
    axins.set_facecolor((0.9, 0.95, 0.7, 0.4))
    # 3. 绘制放大区域曲线
    axins.plot(x, u[:, int(t_slice[i] * 100)], "b-", linewidth=3)
    axins.plot(x, u_pred1[:, int(t_slice[i] * 100)], "r--", linewidth=3)
    axins.plot(x, u_pred2[:, int(t_slice[i] * 100)], "g--", linewidth=3)

    # 4. 设置放大范围
    axins.set_xlim(peak_x - 0.2, peak_x + 0.2)  # x轴放大范围
    axins.set_ylim(-1.05, -0.75)  # y轴放大范围（根据数据调整）
    axins.set_xticks([])
    axins.set_yticks([])
    lines = axins.get_lines()
    lines[0].set_alpha(0.9)  # 精确解
    lines[1].set_alpha(0.7)  # GEN25
    lines[2].set_alpha(0.7)  # GEN100



    # 5. 添加缩放指示框
    ax.indicate_inset_zoom(axins, edgecolor="#000FFF", alpha=0.6)

    # 6. 设置放大图刻度
    # axins.tick_params(axis='both', which='both',
    #                   labelsize=8,  # 缩小刻度文字
    #                   length=2)  # 缩短刻度线



axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=14)





##############################################
gs2 = GridSpec(1, 1, figure=fig)
gs2.update(top=0.24, bottom=0, left=0.0, right=1.0, wspace=0.0)

ax = plt.subplot(gs2[:, :])
ax.axis("off")

fig.savefig(
    "../Burgers/res/"+elementType+".pdf", bbox_inches="tight", pad_inches=0, dpi=500,
)
plt.show(block=True)