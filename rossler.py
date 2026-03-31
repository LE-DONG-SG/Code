import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="Rossler attractor", layout="wide")
st.title("Chaotic system - Rossler attractor")

# 初始化会话状态（控制开始/停止）
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------- 侧边栏参数控制 --------------------------
st.sidebar.header("Parameter settings")
a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01)
b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01)
c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1)
dt_display = st.sidebar.slider("step length (milli)", 1, 10, 1, 1)
max_steps_display = st.sidebar.slider("minimum steps (million)", 1, 10, 1, 1)

#重定义
dt = dt_display / 1000
max_steps = max_steps_display * 1000000

# 控制按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        st.session_state.running = True
        st.session_state.history = []
        # 初始值
        x, y, z = 0.1, 0.1, 0.1
        st.session_state.state = (x, y, z)

with col2:
    if st.button("Stop"):
        st.session_state.running = False

# -------------------------- 实时绘制逻辑 --------------------------
placeholder = st.empty()

if st.session_state.running:
    x, y, z = st.session_state.state
    hist = st.session_state.history

    with st.spinner("Drawing...you may click 'stop'"):
        for step in range(max_steps):
            # 【完全保留你的原始算法】
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)

            x += dx * dt
            y += dy * dt
            z += dz * dt

            hist.append((x, y))

            # 每隔一段时间更新图像，避免卡死
            if step % 500 == 0:
                # 检查是否被停止
                if not st.session_state.running:
                    break

        # 绘图
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_facecolor("black")
        ax.plot(xs, ys, color="#00ffff", linewidth=0.6)
        ax.axis("off")
        placeholder.pyplot(fig)
        plt.close(fig)

        # 保存状态
        st.session_state.state = (x, y, z)
        st.session_state.history = hist

        st.session_state.running = False
        st.success("Done！")