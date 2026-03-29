import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== Streamlit 页面配置 ======
st.set_page_config(page_title="采样混叠演示", layout="wide")
st.title("信号采样与混叠演示 (Nyquist Theorem)")

# ====== 中文字体修复（Streamlit云通用）=====
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 云环境默认字体

# ====== Streamlit 滑块（网页交互）======
col1, col2 = st.columns(2)
with col1:
    f = st.slider("信号频率 f (Hz)", min_value=1, max_value=20, value=5)
with col2:
    fs = st.slider("采样频率 fs (Hz)", min_value=1, max_value=40, value=20)

# ====== 计算信号 ======
t = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * f * t)

# 采样点
ts = np.arange(0, 1, 1/fs)
xs = np.sin(2 * np.pi * f * ts)

# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(10, 6))

# 原始信号
ax.plot(t, x, label="Original Signal", color="red")

# 采样点
ax.vlines(ts, 0, xs, colors='green')
ax.scatter(ts, xs, color='green', label="Sampling Points")

# 混叠判断
if fs < 2 * f:
    f_mod = f % fs  # 取模
    f_alias = fs - f_mod if f_mod > fs / 2 else f_mod  # 折叠为最小正频率
    x_alias = np.sin(2 * np.pi * f_alias * t)
    ax.plot(t, x_alias, '--', color='blue', label="Aliased Signal")
    st.warning("⚠发生混叠 (Aliasing Occurred)")
else:
    st.success("✅满足采样定理 (Satisfy Nyquist Theorem)")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend(loc="upper right")

# ====== 在Streamlit中显示图像 ======
st.pyplot(fig)