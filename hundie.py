import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ====== 中文支持（全系统通用） ======
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.sans-serif'] = [
    'SimHei',
    'Microsoft YaHei',
    'PingFang SC',
    'Hiragino Sans GB',
    'WenQuanYi Micro Hei',
    'sans-serif'
]

# ====== 初始参数 ======
f_init = 5
fs_init = 20
# 固定 t 的定义，全局生效，彻底解决 unresolved reference
t = np.linspace(0, 1, 1000)

# ====== 创建图 ======
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.3)

# 原始信号
line_signal, = ax.plot(t, np.sin(2*np.pi*f_init*t), color='green', label="原始信号")

# 自定义 stem（绿色）
sample_lines = ax.vlines([], [], [], colors='grey')

# 混叠信号（蓝色虚线）
line_alias, = ax.plot([], [], '--', color='red', label="混叠信号")

sample_points = ax.scatter([], [], color='blue', label="采样点")

# 状态文字
text_status = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12)

ax.set_xlabel("时间 (s)")
ax.set_ylabel("幅值")
ax.grid()

# ====== 滑块（修复警告，无报错）======
ax_f = fig.add_axes([0.2, 0.15, 0.65, 0.03])
ax_fs = fig.add_axes([0.2, 0.1, 0.65, 0.03])

slider_f = Slider(ax_f, '信号频率 f (Hz)', 1, 20, valinit=f_init)
slider_fs = Slider(ax_fs, '采样频率 fs (Hz)', 1, 40, valinit=fs_init)

# 去掉红线
slider_f.vline.set_visible(False)
slider_fs.vline.set_visible(False)

# ====== 更新函数 ======
def update(val):
    global sample_lines

    f = slider_f.val
    fs = slider_fs.val

    # 原始信号
    x = np.sin(2 * np.pi * f * t)
    line_signal.set_ydata(x)

    # 采样点
    ts = np.arange(0, 1, 1/fs)
    xs = np.sin(2 * np.pi * f * ts)

    # 更新采样点
    sample_points.set_offsets(np.c_[ts, xs])
    # 修复警告：不删除，直接更新数据
    sample_lines.set_segments([[[t_samp, 0], [t_samp, x_samp]] for t_samp, x_samp in zip(ts, xs)])

    # ====== 混叠判断（不取绝对值，不对称拟合）======
    if fs < 2 * f:
        f_alias = f - fs
        x_alias = np.sin(2 * np.pi * f_alias * t)

        line_alias.set_data(t, x_alias)
        line_alias.set_visible(True)

        text_status.set_text("发生混叠")
    else:
        line_alias.set_visible(False)

        text_status.set_text("满足采样定理")

    ax.legend(loc="upper right")
    fig.canvas.draw_idle()

# 初始化
update(None)

# 绑定事件
slider_f.on_changed(update)
slider_fs.on_changed(update)

plt.show()