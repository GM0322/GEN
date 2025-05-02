import numpy as np
import matplotlib.pyplot as plt
import sys

from pde.Burgers.main import elementType

sys.path.append(".")
from utils.pinn_network import DNN
from pde.Wave.main import GEN
from pde.Wave.pinn_main import PINN
from utils.function import *
from pde.Wave import config
import torch

device, x_min, x_max, t_min, t_max, xt_0, u_0, u_t_0, xt_bc, u_bc, xt_f, u_f = config.config()

# 解析解（取前20项）
def analytical_solution(x, t, terms=20):
    u = 0.0
    for k in range(terms):
        n = 2*k + 1
        coef = 4*(-1)**k / (n**2 * np.pi**2)
        u += coef * np.sin(n * np.pi * x / 2) * np.cos(n * np.pi * t / 2)
    return u

# 参数设置
L = 2.0       # 空间范围 [0, 2]
T = 7.0       # 时间范围 [0, 7]
Nx = 201      # 空间网格数（高分辨率）
Nt = 701      # 时间网格数（高分辨率）
x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)
X, T_grid = np.meshgrid(x, t)  # 生成网格

# 初始化存储解的二维数组
u_data = np.zeros((Nt, Nx))

# 初始条件
u0 = np.piecewise(x, [x < 1, x >= 1], [lambda x: x/2, lambda x: 1 - x/2])
u_prev = u0.copy()
u_current = u0.copy()

# 初始化第二个时间层（假设初始速度 u_t(x,0)=0）
dx = x[1] - x[0]
dt = t[1] - t[0]
for i in range(1, Nx-1):
    u_current[i] = u0[i] + 0.5 * (dt**2 / dx**2) * (u0[i+1] - 2*u0[i] + u0[i-1])
u_data[0, :] = u0
u_data[1, :] = u_current

# 迭代求解（存储所有时间步）
for n in range(1, Nt-1):
    u_next = np.zeros(Nx)
    for i in range(1, Nx-1):
        u_next[i] = 2*u_current[i] - u_prev[i] + (dt**2 / dx**2) * (u_current[i+1] - 2*u_current[i] + u_current[i-1])
    u_next[0], u_next[-1] = 0, 0  # 边界条件
    u_prev, u_current = u_current.copy(), u_next.copy()
    u_data[n+1, :] = u_current

u = analytical_solution(X,T_grid)
pinn = PINN()
sine = GEN(elementType='Sine')
gauss = GEN(elementType='Gauss')
pinn.net.load_state_dict(torch.load("../Wave/checkpoint/pinn_weight.pt",
                                   weights_only=False,
                                   map_location=device))
sine.net.load_state_dict(torch.load("../Wave/checkpoint/Sine_weight.pt",
                                   weights_only=False,
                                   map_location=device))
gauss.net.load_state_dict(torch.load("../Wave/checkpoint/Gauss_weight.pt",
                                   weights_only=False,
                                   map_location=device))
X = X.reshape(-1, 1)
T_grid = T_grid.reshape(-1, 1)
xt_f = torch.from_numpy(np.hstack([X, T_grid])).float().to(device)
u_pinn = pinn.net(xt_f).detach().cpu().numpy().reshape(Nt,Nx)
u_sine = sine.net(xt_f).detach().cpu().numpy().reshape(Nt,Nx)
u_gauss = gauss.net(xt_f).detach().cpu().numpy().reshape(Nt,Nx)



fig1 = plt.figure(figsize=(8, 6))
plt.style.use('seaborn-v0_8-bright')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.constrained_layout.use': True,
    'axes.facecolor': '#f8f8f8',
    'grid.color': 'white',
    'grid.linewidth': 0.9,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#333333'
})
plt.subplot(4,1,1)
plt.imshow(u_data.T,  # 转置矩阵使 x 为纵轴，t 为横轴
           extent=[0, T, 0, L],
           aspect='auto',
           cmap='BrBG',
           vmin=-0.75, vmax=0.5)
plt.colorbar()
# plt.xlabel('$t$')
plt.ylabel('Numerical')
# plt.title('Numerical Solution')
plt.grid(linestyle='--', alpha=0.5)
plt.subplot(4,1,2)
plt.imshow(u_pinn.T,  # 转置矩阵使 x 为纵轴，t 为横轴
           extent=[0, T, 0, L],
           aspect='auto',
           cmap='BrBG',
           vmin=-0.75, vmax=0.5)
plt.colorbar()
# plt.xlabel('$t$')
plt.ylabel('PINN')
# plt.title('PINN')
plt.grid(linestyle='--', alpha=0.5)
plt.subplot(4,1,3)
plt.imshow(u_sine.T,  # 转置矩阵使 x 为纵轴，t 为横轴
           extent=[0, T, 0, L],
           aspect='auto',
           cmap='BrBG',
           vmin=-0.75, vmax=0.5)
plt.colorbar()
# plt.xlabel('$t$')
plt.ylabel('Sine')
# plt.title('Sine')
plt.grid(linestyle='--', alpha=0.5)
plt.subplot(4,1,4)
plt.imshow(u_gauss.T,  # 转置矩阵使 x 为纵轴，t 为横轴
           extent=[0, T, 0, L],
           aspect='auto',
           cmap='BrBG',
           vmin=-0.75, vmax=0.5)
plt.colorbar()
# plt.xlabel('$t$')
plt.ylabel('Gauss')
# plt.title('Gauss')
plt.grid(linestyle='--', alpha=0.5)

# 可视化参数配置
METHOD_STYLES = {
    'Analytical': {'color': '#000000', 'ls': '-', 'lw': 2.5, 'alpha': 0.9},
    'Numerical': {'color': '#d62728', 'ls': '-', 'lw': 2, 'alpha': 0.7, 'marker': 's', 'markersize': 5},
    'PINN': {'color': '#ff7f0e', 'ls': '-.', 'lw': 2, 'alpha': 0.7, 'marker': '^', 'markersize': 5},
    'Sine': {'color': '#9467bd', 'ls': '--', 'lw': 2, 'alpha': 0.7, 'marker': '*', 'markersize': 5},
    'Gauss': {'color': '#1f77b4', 'ls': ':', 'lw': 2, 'alpha': 0.7, 'marker': 'v', 'markersize': 5}
}
# ==================== 3. 曲线绘制函数 ====================
def plot_method(ax, x_loc, data_dict, method, style_config):
    """通用化曲线绘制函数"""
    # 动态计算数据索引
    idx = int(x_loc * 100)
    ax.plot(
        data_dict['tt'][::8],
        data_dict[method][::8,idx],
        ** style_config,
        label = method if method != 'analytical' else 'Analytical'
    )

# ==================== 2. 标注管理系统 ====================
class AnnotationManager:
    """智能标注处理器"""

    def __init__(self, ax):
        self.ax = ax
        self.base_y = 5e-3  # 动态标注基准位置

    def add_region(self, start, end, color, label=None):
        """添加区域背景与智能标注"""
        # 背景色带
        self.ax.axvspan(start, end, alpha=0.12, color=color, zorder=0)


# 创建画布与子图
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
positions = [0.5, 1]  # 三个监测点位置
for ax, x_pos in zip(axs, positions):
    # 绘制各方法曲线
    for method in METHOD_STYLES:
        plot_method(ax, x_pos,
                    {'tt': t, 'Analytical':  u,
                     'Numerical': u_data,
                     'PINN': u_pinn, 'Sine': u_sine, 'Gauss': u_gauss},
                    method, METHOD_STYLES[method])
        # 添加标注系统
        ann_mgr = AnnotationManager(ax)
        ann_mgr.add_region(0, 5.0, '#2ca02c')
        ann_mgr.add_region(5.0, 7, '#d62728')
        ax.text(2, -0.6, 'Fitting Region',
                rotation=0, ha='center', va='center',
                fontsize=10, color='#2ca02c', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='#2ca02c', boxstyle='round,pad=0.3'))

        ax.text(6, 0.2, 'Extrapolation Region',
                rotation=0, ha='center', va='center',
                fontsize=10, color='#d62728', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))
        plt.legend(loc='upper left', ncol=1,
                   title_fontsize='12',
                   borderaxespad=0.5,
                   frameon=True,
                   title='Methods',
                   bbox_to_anchor=(0.98, 0.98))
        # 坐标轴设置
        # ax.set(xlim=(-0.75,0.5), ylim=(1e-3, 3e-2))
        ax.set_xlabel("$t$", fontweight='bold')
        ax.set_title(f"$u(x = {x_pos},t)$", pad=12, fontsize=12)
        ax.grid(True, which='both', alpha=0.4)
        ax.axvline(2.0, color='gray', ls='--', lw=1, alpha=0.6)
fig.savefig('./res/wave_curve.pdf')

plt.show(block=True)