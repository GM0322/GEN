import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-bright')  # 使用现代风格
# plt.style.use('seaborn')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 300,
    'axes.facecolor': '#f5f5f5',
    'grid.color': 'white',
    'grid.linewidth': 1.2,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#444444'
})
sine = np.load('./res/Sine_res.npy', allow_pickle=True).item()
pinn = np.load('./res/pinn_res.npy', allow_pickle=True).item()
gauss = np.load('./res/Gauss_res.npy', allow_pickle=True).item()

x_min = 0
x_max = sine['x'][1]*sine['x'].size
t_min = 0
t_max = sine['t'][1]*sine['t'].size
dx = sine['x'][1]
dt = 0.5*sine['x'][1]*sine['t'][1]
c = 1

x = np.arange(x_min, x_max, dx)
t = np.arange(t_min, t_max, dt)
T,X = np.meshgrid(t, x)

# X = X.reshape(-1, 1)
# T = T.reshape(-1, 1)
U = np.exp(-(np.pi/2)**2*T)*np.sin(np.pi/2*X)
u_num = np.zeros_like(U)
u_num[:,0] = np.sin(np.pi/2*x)
for i,t0 in enumerate(t):
    if i == t.size - 1:
        break
    for j, x0 in enumerate(x):
        if j == 0:
            u_num[j,i+1] = 0
        elif j == x.size-1:
            u_num[j,i+1] = 0
        else:
            u_num[j,i+1] = u_num[j,i]+c**2*(u_num[j+1,i]+u_num[j-1,i]-2*u_num[j,i])*dt/(dx**2)
tt = t[::200]
u = U[:,::200]
u_num0 = u_num[:,::200]
# 创建图形
# plt.figure(figsize=(10, 6))
#
# # 自定义调色板
# colors = {
#     # 'analytical': '#2ca02c',
#     'analytical': '#000000',
#     'numerical': '#d62728',
#     'pinn': '#ff7f0e',
#     'sine': '#9467bd',
#     'gauss': '#1f77b4'
# }
#
# # 绘制曲线
# line_config = {
#     'linewidth': 1,
#     'markersize': 3,
#     'alpha': 0.55,
#     'markeredgewidth': 1.2
# }
#
# line_config_a = {
#     'linewidth': 1,
#     'markersize': 3,
#     'alpha': 1.0,
#     'markeredgewidth': 1.2
# }
#
# for i in range(3):
#     x = 0.25*(i+1)
#     plt.subplot(1,3,i+1)
#     plt.semilogy(tt[150:], u[int(x*200),150:],
#                 color=colors['analytical'],
#                 linestyle='-',#'(0, (5, 2)),  # 自定义虚线样式
#                 # marker='',
#                 markerfacecolor='black',
#                 **line_config_a,
#                 label='Analytical')
#
#     plt.semilogy(tt[150:], u_num0[int(x*200),150:],
#                 color=colors['numerical'],
#                 linestyle='-',
#                 marker='s',
#                 markerfacecolor=colors['numerical'],
#                 **line_config,
#                 label='Numerical')
#
#     plt.semilogy(tt[150:], pinn['u'][int(x*200),150:],
#                 color=colors['pinn'],
#                 linestyle=(0, (3, 1.5)),
#                 marker='^',
#                 markerfacecolor='white',
#                 **line_config,
#                 label='PINN')
#
#     plt.semilogy(tt[150:], sine['u'][int(x*200),150:],
#                 color=colors['sine'],
#                 linestyle='--',
#                 marker='o',
#                 markerfacecolor=colors['sine'],
#                 **line_config,
#                 label='SineGen')
#
#     plt.semilogy(tt[150:], gauss['u'][int(x*200),150:],
#                 color=colors['gauss'],
#                 linestyle='-.',
#                 marker='v',
#                 markerfacecolor='white',
#                 **line_config,
#                 label='GaussGen')
#
#     # 添加区域标注
#     ax = plt.gca()
#
#     # 绘制区域背景
#     ax.axvspan(1.5, 2.0, alpha=0.15, color='#2ca02c', label='_nolegend_')
#     ax.axvspan(2.0, 2.5, alpha=0.15, color='#d62728', label='_nolegend_')
#
#     # 添加区域文字标注
#     ax.text(1.72, 5e-3, 'Fitting Region\n(1.5-2.0)',
#             rotation=0, ha='center', va='center',
#             fontsize=10, color='#2ca02c', fontweight='bold',
#             bbox=dict(facecolor='white', edgecolor='#2ca02c', boxstyle='round,pad=0.3'))
#
#     ax.text(2.22, 5e-3, 'Extrapolation Region\n(2.0-2.5)',
#             rotation=0, ha='center', va='center',
#             fontsize=10, color='#d62728', fontweight='bold',
#             bbox=dict(facecolor='white', edgecolor='#d62728', boxstyle='round,pad=0.3'))
#
#     # 添加分隔线
#     ax.axvline(2.0, color='gray', linestyle='--', lw=1.2, alpha=0.7)
#
#     # 添加区域箭头标注
#     # ax.annotate('', xy=(1.75, 5e-3), xytext=(1.75, 1e-3),
#     #             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))
#     # ax.annotate('', xy=(2.25, 5e-3), xytext=(2.25, 1e-7),
#     #             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
#     # 装饰图形
#     plt.grid(True, which="both", ls=":", alpha=0.7)
#     plt.xlabel("$t$", fontweight='bold', labelpad=10)
#     plt.ylabel("$u(x={},t)$ (log scale)".format(x), fontweight='bold', labelpad=10)
#     # 优化图例
#     legend = plt.legend(loc='upper right', ncol=2,
#                         title_fontsize='13',
#                         borderaxespad=0.5,
#                         title='Methods',
#                         bbox_to_anchor=(0.98, 0.98))
#     legend.get_title().set_fontweight('bold')
#
#     # 调整坐标轴
#     plt.gca().set_axisbelow(True)
#     plt.gca().tick_params(axis='both', which='major', labelsize=11)
# plt.title("Wave Propagation: Numerical vs Machine Learning Methods",
#          fontsize=14, pad=15, fontweight='bold')


# ==================== 1. 全局配置系统 ====================
def set_global_style():
    """统一可视化样式配置"""
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

        # 动态计算标注位置
        # x_center = (start + end) / 2
        # y_pos = self.base_y * (1.2 if 'Extrapolation' in label else 1)

        # 带箭头的文本标注
        # self.ax.annotate(
        #     label,
        #     xy=(x_center, y_pos),
        #     xytext=(x_center, y_pos * 0.1),
        #     arrowprops=dict(
        #         arrowstyle="->",
        #         color=color,
        #         lw=1.2,
        #         connectionstyle="arc3,rad=-0.2"
        #     ),
        #     ha='center',
        #     va='center',
        #     fontsize=10,
        #     color=color,
        #     fontweight='bold',
        #     bbox=dict(
        #         boxstyle='round,pad=0.3',
        #         facecolor='white',
        #         edgecolor=color
        #     )
        # )


# ==================== 3. 曲线绘制函数 ====================
def plot_method(ax, x_loc, data_dict, method, style_config):
    """通用化曲线绘制函数"""
    # 动态计算数据索引
    idx = int(x_loc * 100)
    ax.semilogy(
        data_dict['tt'][150::4],
        data_dict[method]['u'][idx, 150::4],
        ** style_config,
        label = method if method != 'analytical' else 'Analytical'
    )

# ==================== 主程序 ====================
def main():
    set_global_style()

    # 数据预处理 (保持原有代码结构)
    # ... [数据加载与预处理代码保持不变] ...

    # 可视化参数配置
    METHOD_STYLES = {
        'Analytical': {'color': '#000000', 'ls': '-', 'lw': 1.5, 'alpha': 0.9},
        'Numerical': {'color': '#d62728', 'ls': '-', 'lw': 1, 'alpha': 0.7, 'marker': 's', 'markersize': 1.5},
        'PINN': {'color': '#ff7f0e', 'ls': '-.', 'lw': 1, 'alpha': 0.7, 'marker': '^', 'markersize': 1.5},
        'Sine': {'color': '#9467bd', 'ls': '--', 'lw': 1, 'alpha': 0.7, 'marker': 'o', 'markersize': 1.5},
        'Gauss': {'color': '#1f77b4', 'ls': ':', 'lw': 1, 'alpha': 0.7, 'marker': 'v', 'markersize': 1.5}
    }

    # 创建画布与子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    positions = [0.5, 1.0]  # 三个监测点位置

    # 循环生成子图
    for ax, x_pos in zip(axs, positions):
        # 绘制各方法曲线
        for method in METHOD_STYLES:
            plot_method(ax, x_pos,
                        {'tt': tt, 'Analytical': {'u': u},
                         'Numerical': {'u': u_num0},
                         'PINN': pinn, 'Sine': sine, 'Gauss': gauss},
                        method, METHOD_STYLES[method])

        # 添加标注系统
        ann_mgr = AnnotationManager(ax)
        ann_mgr.add_region(1.5, 2.0, '#2ca02c')
        ann_mgr.add_region(2.0, 2.5, '#d62728')
        ax.text(1.72, 0.0015, 'Fitting Region',
                rotation=0, ha='center', va='center',
                fontsize=10, color='#2ca02c', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='#2ca02c', boxstyle='round,pad=0.3'))

        ax.text(2.2, 0.0015, 'Extrapolation Region',
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
        ax.set(xlim=(1.5, 2.475), ylim=(1e-3, 3e-2))
        ax.set_xlabel("$t$", fontweight='bold')
        ax.set_title(f"$u(x = {x_pos},t)$", pad=12, fontsize=12)
        ax.grid(True, which='both', alpha=0.4)
        ax.axvline(2.0, color='gray', ls='--', lw=1, alpha=0.6)

        ax_inset = ax.inset_axes(
            [0.55, 0.55, 0.4, 0.4],  # 位置参数：x, y, width, height
            xlim=(2.3, 2.475),
            ylim=(0.00138,0.004)  # 根据实际数据范围调整
        )

        # 在放大图中绘制相同数据（简化样式）
        methods = ['Analytical', 'Numerical', 'PINN', 'Sine', 'Gauss']
        data_dict = {'tt': tt, 'Analytical': {'u': u},
         'Numerical': {'u': u_num0},
         'PINN': pinn, 'Sine': sine, 'Gauss': gauss},
        idx = int(x_pos * 100)
        for method in methods:
            ax_inset.semilogy(
                data_dict[0]['tt'][150::4],
                data_dict[0][method]['u'][idx, 150::4],
                ** METHOD_STYLES[method]
            )

        # 配置放大图样式
        ax_inset.grid(True, alpha=0.3)
        ax_inset.tick_params(axis='both', labelsize=6)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")

        # 添加放大区域指示框
        ax.indicate_inset_zoom(ax_inset,
                               edgecolor="#444444",
                               lw=0.8,
                               linestyle="--",
                               alpha=0.7)

    # 全局标签设置
    axs[0].set_ylabel("$u(x,t)$(log scale)", fontweight='bold')
    # fig.suptitle("Wave Propagation Analysis: Comparative Methods Performance",
    #              y=0.95, fontsize=11, fontweight='bold')

    # 智能图例管理 (仅显示一次)
    handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels,
    #            loc='lower right',
    #            nrow=5,
    #            bbox_to_anchor=(0.5, 1.08),
    #            frameon=True,
    #            title='Methodology Comparison',
    #            title_fontsize='8')
    # plt.legend(loc='upper right', ncol=1,
    #             title_fontsize='8',
    #             borderaxespad=0.5,
    #             frameon=True,
    #             title='Methods',
    #             bbox_to_anchor=(0.98, 0.98))
    # plt.subplots_adjust(top=0.99,left=0.01)
    # plt.tight_layout(rect=(0.01,0,1,0.99))

    plt.savefig('./res/heat_curve.pdf')
    plt.show(block=True)

if __name__ == "__main__":
    main()
