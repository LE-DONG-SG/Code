import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="Rossler attractor", layout="wide")
st.title("Chaotic system - Rossler attractor")

# 初始化会话状态
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------- 选项卡 --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])

# ==================== Tab 1: Animation（原功能） ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    # 侧边栏参数
    st.sidebar.header("Parameter settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1, key="c")
    dt_display = st.sidebar.slider("step length (milli)", 0.1, 1.0, 0.5, 0.1, key="dt")
    max_steps_display = st.sidebar.slider("max steps (thousand)", 100, 5000, 1000, 100, key="steps")

    dt = dt_display / 1000
    max_steps = max_steps_display * 1000

    # 控制按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start"):
            st.session_state.running = True
            st.session_state.history = []
            x, y, z = 0.1, 0.1, 0.1
            st.session_state.state = (x, y, z)

    with col2:
        if st.button("Stop", key="stop"):
            st.session_state.running = False

    placeholder = st.empty()

    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history

        with st.spinner("Drawing..."):
            for step in range(max_steps):
                # Rossler 方程
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)

                x += dx * dt
                y += dy * dt
                z += dz * dt

                hist.append((x, y))

                if step % 500 == 0:
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

            st.session_state.state = (x, y, z)
            st.session_state.history = hist
            st.session_state.running = False
            st.success("Drawing done!")

# ==================== Tab 2: Complexity Analysis（复杂度分析） ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")

    st.markdown("""
    Here you could check：
    - Max Lyapunov Exponent
    - 分形维数 / 关联维数
    - 功率谱
    - 相空间重构
    - 混沌判定
    """)

    if st.button("Compute Complexity", key="compute"):
        if len(st.session_state.history) < 100:
            st.warning("请先去 Animation 页面运行一次吸引子！")
        else:
            with st.spinner("Calculating..."):
                # 简单示例：用 x 序列计算
                xs = np.array([p[0] for p in st.session_state.history])

                # 估算最大李雅普诺夫指数（简易版）
                lyap = np.mean(np.abs(np.log(np.abs(np.diff(xs) + 1e-9))))
                lyap = round(lyap, 4)

                # 非混沌 / NaN → 显示 -
                if np.isnan(lyap) or lyap <= 0.01:
                    lyap_display = "-"
                    chaos_state = "Not Chaotic"
                    chaos_color = "#ff4444"
                else:
                    lyap_display = f"{lyap}"
                    chaos_state = "Chaotic"
                    chaos_color = "#00C000"

                # ==================== 正常字号显示（修复完成） ====================
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Max Lyapunov Exponent:** {lyap_display}")
                with col2:
                    st.markdown(f"**Chaos State:** <span style='color:{chaos_color}'>{chaos_state}</span>",
                                unsafe_allow_html=True)

            st.success("Complexity analysis completed!")