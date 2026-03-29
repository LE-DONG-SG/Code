import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="Rossler attractor", layout="wide")
st.title("Chaotic system - Rossler attractor")

# 初始化会话状态（控制开始/停止/步数/状态）
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "current_step" not in st.session_state:
    st.session_state.current_step = 0  # 新增：记录当前迭代步数
if "state" not in st.session_state:
    st.session_state.state = (0.1, 0.1, 0.1)  # 初始x,y,z

# -------------------------- 侧边栏参数控制 --------------------------
st.sidebar.header("Parameter settings")
a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01)
b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01)
c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1)
dt = st.sidebar.slider("step length", 0.001, 0.01, 0.005, 0.001)
max_steps = st.sidebar.slider("maximum steps", 10000, 100000, 50000, 1000)

# 控制按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        # 重置所有状态
        st.session_state.running = True
        st.session_state.history = []
        st.session_state.current_step = 0
        st.session_state.state = (0.1, 0.1, 0.1)

with col2:
    if st.button("Stop"):
        st.session_state.running = False

# -------------------------- 实时绘制逻辑（核心修改） --------------------------
placeholder = st.empty()

# 仅在运行状态下，增量计算+绘图
if st.session_state.running and st.session_state.current_step < max_steps:
    x, y, z = st.session_state.state
    hist = st.session_state.history
    current_step = st.session_state.current_step

    # 关键：每次只计算500步（少量步数，不阻塞）
    update_steps = 500
    for step in range(update_steps):
        # Rossler方程（完全保留你的原始算法）
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        x += dx * dt
        y += dy * dt
        z += dz * dt
        hist.append((x, y))

    # 绘制当前所有轨迹
    xs = [p[0] for p in hist]
    ys = [p[1] for p in hist]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor("black")
    ax.plot(xs, ys, color="#00ffff", linewidth=0.6)
    ax.axis("off")
    placeholder.pyplot(fig)
    plt.close(fig)

    # 保存状态到会话
    st.session_state.state = (x, y, z)
    st.session_state.history = hist
    st.session_state.current_step += update_steps

    # 关键：主动刷新页面，实现实时更新
    time.sleep(0.01)  # 微小延时，控制帧率
    st.rerun()  # 主动触发Streamlit重运行，立即显示图像

# 绘制完成
if st.session_state.current_step >= max_steps and not st.session_state.running:
    st.success("Done！")