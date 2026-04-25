import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("采样混叠演示 (Nyquist Theorem)")

# ================= 参数设置 =================
col1, col2 = st.columns(2)
with col1:
    f = st.slider(r"信号频率 $f\ (\rm Hz)$", 1, 50, 15)
with col2:
    fs = st.slider(r"采样频率 $f_{\rm s}\ (\rm Hz)$", 1, 50, 45)

# ================= 时域信号 =================
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * f * t)

ts = np.arange(0, 1, 1 / fs)
xs = np.sin(2 * np.pi * f * ts)

# ================= 图1：时域 =================
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(t, x, label="Original Signal", color="red")
ax1.vlines(ts, 0, xs, colors='green')
ax1.scatter(ts, xs, color='green', label="Sampling Points")

if fs < 2 * f:
    f_mod = f % fs
    f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
    x_alias = np.sin(2 * np.pi * f_alias * t)
    x_alias = -np.sin(2 * np.pi * f_alias * t) if f_mod > fs / 2 else x_alias
    ax1.plot(t, x_alias, '--', color='blue', label="Aliased Signal")
    st.warning("⚠ Aliasing Occurred")
else:
    st.success("✅ Satisfy Nyquist Theorem")

ax1.set_title("Time Domain Signal")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# ================= 频谱计算 =================
def fft_signal(sig, t):
    N = len(sig)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(N, dt)
    spectrum = np.abs(np.fft.fft(sig)) / N
    return np.fft.fftshift(freq), np.fft.fftshift(spectrum)

# 原信号频谱
freq_x, X = fft_signal(x, t)

# 抽样信号（用0填充模拟冲激采样）
x_sampled = np.zeros_like(t)
indices = (ts / (t[1] - t[0])).astype(int)
indices = indices[indices < len(t)]
x_sampled[indices] = xs[:len(indices)]

freq_s, Xs = fft_signal(x_sampled, t)

# ================= 图2：频域对比 =================
fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(freq_x, X, label="Original Spectrum", color='red')
ax2.set_xlim(-60, 60)
ax2.set_title("Original Signal Spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)

# ================= 图3：抽样后频谱 =================
fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.plot(freq_s, Xs, label="Sampled Spectrum", color='blue')
ax3.set_xlim(-60, 60)

# ================= 获取幅值最大值 =================
ymax = np.max(Xs)
ax3.set_ylim(0, ymax * 1.2)

# ================= 混叠区域（统一蓝色） =================
if fs < 2 * f:
    f_overlap_start = fs - f
    f_overlap_end = f

    if f_overlap_start < f_overlap_end:
        ax3.fill_betweenx(
            [0, ymax],
            f_overlap_start,
            f_overlap_end,
            color='blue',
            alpha=0.2,
            label="Aliasing Region"
        )
        ax3.fill_betweenx(
            [0, ymax],
            -f_overlap_end,
            -f_overlap_start,
            color='blue',
            alpha=0.2
        )

# ================= 统一虚线函数 =================
def vline(x, label=None):
    ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')
    if label:
        ax3.text(x, ymax * 1.05, label, ha='center')  # 标签在虚线上方

# ±f
vline(f, r"$+f$")
vline(-f, r"$-f$")

# ±fs
vline(fs, r"$+f_s$")
vline(-fs, r"$-f_s$")

# ================= 基本设置 =================
ax3.set_title("Sampled Signal Spectrum (Periodic Extension)")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Magnitude")
ax3.grid(True)

# 图例固定右上角
ax3.legend(loc="upper right")

st.pyplot(fig3)