import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.spatial import KDTree

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Rossler Attractor", layout="wide")
st.title("Chaotic System - Rossler Attractor")

# Session State Initialization
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "state" not in st.session_state:
    st.session_state.state = (0.1, 0.1, 0.1)

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])


# Rossler方程定义（全局复用，避免重复定义）
def rossler(x, y, z, a, b, c):
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return dx, dy, dz


# RK4积分单步计算
def rk4_step(x, y, z, a, b, c, dt):
    k1x, k1y, k1z = rossler(x, y, z, a, b, c)
    k2x, k2y, k2z = rossler(x + dt * k1x / 2, y + dt * k1y / 2, z + dt * k1z / 2, a, b, c)
    k3x, k3y, k3z = rossler(x + dt * k2x / 2, y + dt * k2y / 2, z + dt * k2z / 2, a, b, c)
    k4x, k4y, k4z = rossler(x + dt * k3x, y + dt * k3y, z + dt * k3z, a, b, c)

    x_new = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    z_new = z + dt / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
    return x_new, y_new, z_new


# ==================== Tab 1: Animation ====================
with tab1:
    st.subheader("Rossler Attractor Real-time Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1, key="c")
    dt_display = st.sidebar.slider("Step Length (ms)", 0.1, 1.0, 0.5, 0.1, key="dt")
    max_steps_display = st.sidebar.slider("Max Steps", 1000, 20000, 5000, 1000, key="steps")

    dt = dt_display / 1000
    max_steps = max_steps_display

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start", key="start"):
            st.session_state.running = True
            st.session_state.history = []
            st.session_state.state = (0.1, 0.1, 0.1)

    with col2:
        if st.button("Stop", key="stop"):
            st.session_state.running = False

    with col3:
        if st.button("Reset", key="reset"):
            st.session_state.history = []
            st.session_state.state = (0.1, 0.1, 0.1)
            st.session_state.running = False

    # 实时动画占位符
    plot_placeholder = st.empty()
    status_placeholder = st.empty()

    # 实时绘制逻辑
    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history

        for step in range(max_steps):
            if not st.session_state.running:
                break

            # RK4单步积分
            x, y, z = rk4_step(x, y, z, a, b, c, dt)
            # 存储完整x,y,z轨迹
            hist.append((x, y, z))

            # 每20步更新一次绘图（性能优化）
            if step % 20 == 0:
                xs = [p[0] for p in hist]
                ys = [p[1] for p in hist]

                fig, ax = plt.subplots(figsize=(8, 7))
                ax.set_facecolor("black")
                ax.plot(xs, ys, color="#00ffff", linewidth=0.6)
                ax.axis("off")
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                status_placeholder.text(f"Running: Step {step}/{max_steps}")

        # 保存最终状态
        st.session_state.state = (x, y, z)
        st.session_state.history = hist
        status_placeholder.success(f"Animation completed! Total points: {len(hist)}")

# ==================== Tab 2: Complexity Analysis ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")


    # 关联维数计算
    def correlation_dimension(series, k=5):
        series = series.reshape(-1, 1)
        tree = KDTree(series)
        dists, _ = tree.query(series, k=k + 1)
        dists = dists[:, 1:]
        dists = dists[dists > 1e-9]
        if len(dists) == 0:
            return 0.0
        log_eps = np.log(np.sort(dists))
        log_C = np.log(np.arange(1, len(log_eps) + 1) / len(log_eps))
        fit = np.polyfit(log_eps[:len(log_eps) // 2], log_C[:len(log_eps) // 2], 1)
        return fit[0]


    # 相空间重构
    def phase_space_reconstruction(series, tau=1, m=2):
        N = len(series)
        max_i = N - m * tau
        if max_i <= 0:
            return np.array([[0, 0]])
        return np.array([series[i:i + m * tau:tau] for i in range(max_i)])


    # 0-1混沌测试
    def chaos_01_test(series):
        N = len(series)
        if N < 100:
            return 0.0
        n = np.arange(N)
        p, q = np.zeros(N), np.zeros(N)
        c = np.pi / 4
        for i in range(1, N):
            p[i] = p[i - 1] + series[i - 1] * np.cos(c * i)
            q[i] = q[i - 1] + series[i - 1] * np.sin(c * i)
        M = np.sqrt(p ** 2 + q ** 2)
        K = np.polyfit(np.log(n[1:]), np.log(M[1:]), 1)[0]
        return K


    # 正确的最大李雅普诺夫指数计算（RK4+三维轨迹）
    def max_lyapunov(x_traj, y_traj, z_traj, dt, a, b, c, eps=1e-6):
        n = len(x_traj)
        if n < 100:
            return 0.0
        log_div = []
        # 初始扰动
        x1, y1, z1 = x_traj[0], y_traj[0], z_traj[0]
        x2, y2, z2 = x1 + eps, y1, z1

        for i in range(n - 1):
            # 两条轨道都用RK4积分
            x1, y1, z1 = rk4_step(x1, y1, z1, a, b, c, dt)
            x2, y2, z2 = rk4_step(x2, y2, z2, a, b, c, dt)

            # 计算距离
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            d = max(d, 1e-12)
            # 归一化扰动
            x2 = x1 + eps * (x2 - x1) / d
            y2 = y1 + eps * (y2 - y1) / d
            z2 = z1 + eps * (z2 - z1) / d

            log_div.append(np.log(d / eps))

        return np.mean(log_div) / dt


    # 可调参数
    st.sidebar.header("Analysis Settings")
    tau = st.sidebar.slider("Delay tau (PSR)", 1, 10, 1)
    m = st.sidebar.slider("Embedding Dim m (PSR)", 2, 5, 2)

    if st.button("Compute Complexity Indicators", key="compute"):
        hist = st.session_state.history
        if len(hist) < 200:
            st.warning("Please run the animation first to generate trajectory data!")
        else:
            with st.spinner("Calculating..."):
                # 提取完整轨迹
                traj = np.array(hist)
                xs = traj[:, 0]
                ys = traj[:, 1]
                zs = traj[:, 2]
                current_dt = dt  # 使用动画的真实步长
                current_a, current_b, current_c = a, b, c

                # 1. 最大李雅普诺夫指数
                lyap = max_lyapunov(xs, ys, zs, current_dt, current_a, current_b, current_c)
                lyap = round(lyap, 4)
                is_chaotic = lyap > 0.01

                # 2. 关联维数
                corr_dim = round(correlation_dimension(xs), 4)

                # 3. 0-1测试
                K = round(chaos_01_test(xs), 4)
                k_chaos = K > 0.5

                # 4. 功率谱
                f, Pxx = welch(xs, fs=1000, nperseg=1024)

                # 5. 相空间重构
                psr = phase_space_reconstruction(xs, tau=tau, m=m)

                # 结果展示
                st.markdown("### Analysis Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Max Lyapunov Exponent", lyap)
                col2.metric("Correlation Dimension", corr_dim)
                col3.metric("0-1 Test K", K)

                col4, col5 = st.columns(2)
                col4.metric("Chaos State", "Chaotic" if is_chaotic else "Non-chaotic")
                col5.metric("0-1 Test Result", "Chaotic" if k_chaos else "Non-chaotic")

                # 绘图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(f, Pxx, color='c')
                ax1.set_title("Power Spectrum")
                ax1.grid(alpha=0.3)

                ax2.plot(psr[:, 0], psr[:, 1], color='magenta', linewidth=0.5)
                ax2.set_title(f"Phase Space Reconstruction (tau={tau}, m={m})")
                ax2.axis('equal')
                st.pyplot(fig)
                plt.close(fig)

            st.success("All indicators computed successfully!")