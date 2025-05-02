import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pde.Heat.main import GEN
import torch
from pde.Heat import config



device,x_min,x_max,t_min,t_max,xt_0,u_0,xt_bc,u_bc,xt_f,u_f = config.config()
# 自定义美学参数
PLOT_STYLE = {
    "figure1": {
        # "title": "Neural Basis Functions Visualization",
        "cmap": LinearSegmentedColormap.from_list('custom_reds', ['#fee0d2', '#de2d26']),
        "subplot_pad": 0.8,
        "label_params": {
            # "xlabel": "Input Domain",
            # "ylabel": "Function Output",
            # "title_fmt": "Basis Function ${A}_{%d} \sin(\omega_{%d} x) + b$"
        }
    },
    "figure2": {
        "title": "Feature Space Activation Patterns",
        "cmap": "viridis",
        "cbar_title": "Activation Strength",
        "subplot_pad": 0.4,
        "annotation_color": "#ffffff"
    }
}


def configure_style():
    """全局可视化样式配置"""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'figure.constrained_layout.use': True,
        'figure.dpi': 300,
        'savefig.transparent': True
    })


def plot_basis_functions(gen):
    """绘制基函数图形"""
    fig = plt.figure(1, figsize=(12, 9))
    gs = GridSpec(5, 5, figure=fig, wspace=0.15, hspace=0.4)

    x = np.linspace(-10, 10, 2000)
    cmap = plt.get_cmap(PLOT_STYLE["figure1"]["cmap"])

    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            index = i * 5 + j

            # 计算函数曲线
            A = gen.net.A1[0, 0, index].item()
            w = gen.net.w1[0, 0, index].item()
            b = gen.net.b.item()
            y = A * np.sin(w * x) + b

            # 绘制动态颜色映射
            color = cmap(index / 25)
            ax.plot(x, y, lw=1.2, color=color, alpha=0.9)

            # 美学优化
            ax.set_xlim(-10, 10)
            # ax.set_ylim(-1.2 * (abs(A) + abs(b)), 1.2 * (abs(A) + abs(b)))
            ax.text(0.05, 0.9,
                    f"A={A:.2f}\nω={w:.2f}",
                    transform=ax.transAxes,
                    fontsize=15,
                    va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                    )

            # if j == 0:
            #     ax.set_ylabel(PLOT_STYLE["figure1"]["label_params"]["ylabel"],
            #                   fontsize=7)
            # if i == 4:
            #     ax.set_xlabel(PLOT_STYLE["figure1"]["label_params"]["xlabel"],
            #                   fontsize=7)

    # fig.suptitle(PLOT_STYLE["figure1"]["title"],
    #              y=0.95, fontsize=12, fontweight='semibold')


def plot_activation_patterns(gen, device):
    """绘制激活模式热图"""
    fig = plt.figure(2, figsize=(12, 9))
    gs = GridSpec(5, 5, figure=fig, wspace=0.05, hspace=0.15)

    # 生成网格数据
    x = np.linspace(-0.0, 2.5, 250)
    t = np.linspace(0, 1, 300)
    xx, tt = np.meshgrid(x, t)

    # 神经网络前传
    with torch.no_grad():
        xx_tensor = torch.from_numpy(xx).float().view(-1, 1).to(device)
        tt_tensor = torch.from_numpy(tt).float().view(-1, 1).to(device)
        _, out = gen.net(torch.cat((xx_tensor, tt_tensor), dim=1))

    # 创建共享颜色规范
    vmin = out.min().item()
    vmax = out.max().item()

    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            index = i * 5 + j

            temp = out[:, 0, i, j].data.cpu().numpy().reshape(300, 250)

            # 绘制热图
            im = ax.imshow(temp.T, cmap=PLOT_STYLE["figure2"]["cmap"])
                           # extent=[-0.5, 2.5, 0, 1])
            cbar = fig.colorbar(im)
            cbar.set_label('',fontsize=15)
            ax.grid('off')
            ax.axis('off')


# 主程序
configure_style()

# 绘制第一个图形
elementType = "Sine"
gen = GEN(elementType=elementType)
gen.net.load_state_dict(torch.load("../Heat/checkpoint/" + elementType + "_weight.pt",
                                   weights_only=False,
                                   map_location=device))
plot_basis_functions(gen)
plt.savefig('./res/heat_sine.pdf')
# 绘制第二个图形
elementType = "Gauss"
gen = GEN(elementType=elementType)
gen.net.load_state_dict(torch.load("../Heat/checkpoint/" + elementType + "_weight.pt",
                                   weights_only=False,
                                   map_location=device))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plot_activation_patterns(gen, device)
plt.savefig('./res/heat_gauss.pdf')
plt.show(block=True)