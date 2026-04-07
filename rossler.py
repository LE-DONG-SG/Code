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

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])

# ==================== Tab 1: Animation ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1, key="c")
    dt_display = st.sidebar.slider("Step Length (ms)", 0.1, 1.0, 0.5, 0.1, key="dt")
    max_steps_display = st.sidebar.slider("Max Steps (k)", 100, 5000, 1000, 100, key="steps")

    dt = dt_display / 1000
    max_steps = max_steps_display * 1000

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start"):
            st.session_state.running = True
            st.session_state.history = []
            st.session_state.state = (0.1, 0.1, 0.1)

    with col2:
        if st.button("Stop", key="stop"):
            st.session_state.running = False

    placeholder = st.empty()

    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history

        with st.spinner("Drawing..."):
            for step in range(max_steps):
                # --------------- FIX: RK4 积分 ----------------
                def rossler(x,y,z):
                    dx = -y - z
                    dy = x + a*y
                    dz = b + z*(x-c)
                    return dx, dy, dz

                k1x,k1y,k1z = rossler(x,y,z)
                k2x,k2y,k2z = rossler(x+dt*k1x/2, y+dt*k1y/2, z+dt*k1z/2)
                k3x,k3y,k3z = rossler(x+dt*k2x/2, y+dt*k2y/2, z+dt*k2z/2)
                k4x,k4y,k4z = rossler(x+dt*k3x, y+dt*k3y, z+dt*k3z)

                x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
                y += dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
                z += dt/6 * (k1z + 2*k2z + 2*k3z + k4z)

                hist.append((x, y))

                if step % 500 == 0 and not st.session_state.running:
                    break

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
            st.success("Drawing completed!")

# ==================== Tab 2: Complexity Analysis ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")

    def correlation_dimension(series, k=5):
        series = series.reshape(-1, 1)
        tree = KDTree(series)
        dists, _ = tree.query(series, k=k+1)
        dists = dists[:, 1:]
        dists = dists[dists > 1e-9]
        log_eps = np.log(np.sort(dists))
        log_C = np.log(np.arange(1, len(log_eps)+1) / len(log_eps))
        fit = np.polyfit(log_eps[:len(log_eps)//2], log_C[:len(log_eps)//2], 1)
        return fit[0]

    def phase_space_reconstruction(series, tau=1, m=2):
        N = len(series)
        return np.array([series[i:i+m*tau:tau] for i in range(N - m*tau + 1)])

    def chaos_01_test(series):
        N = len(series)
        n = np.arange(N)
        p, q = np.zeros(N), np.zeros(N)
        c = np.pi / 4
        for i in range(1, N):
            p[i] = p[i-1] + series[i-1] * np.cos(c*i)
            q[i] = q[i-1] + series[i-1] * np.sin(c*i)
        M = np.sqrt(p**2 + q**2)
        K = np.polyfit(np.log(n[1:]), np.log(M[1:]), 1)[0]
        return K

    # --------------- FIX: 正确李雅普诺夫指数 ----------------
    def correct_lyapunov(x_traj, y_traj, z_traj, dt, a=0.2, b=0.2, c=5.7):
        n = len(x_traj)
        log_divergence = []
        dx0 = 1e-6
        # 初始扰动
        x1 = x_traj.copy()
        y1 = y_traj.copy()
        z1 = z_traj.copy()
        x2 = x1 + dx0
        y2 = y1
        z2 = z1

        for i in range(n - 1):
            # 演化轨道1
            k1x = -y1[i] - z1[i]
            k1y = x1[i] + a * y1[i]
            k1z = b + z1[i] * (x1[i] - c)
            x1[i + 1] = x1[i] + dt * k1x
            y1[i + 1] = y1[i] + dt * k1y
            z1[i + 1] = z1[i] + dt * k1z

            # 演化轨道2
            k2x = -y2[i] - z2[i]
            k2y = x2[i] + a * y2[i]
            k2z = b + z2[i] * (x2[i] - c)
            x2[i + 1] = x2[i] + dt * k2x
            y2[i + 1] = y2[i] + dt * k2y
            z2[i + 1] = z2[i] + dt * k2z

            # 距离
            d = np.sqrt((x2[i + 1] - x1[i + 1]) ** 2 + (y2[i + 1] - y1[i + 1]) ** 2 + (z2[i + 1] - z1[i + 1]) ** 2)
            if d < 1e-12:
                d = 1e-12
            # 归一化
            x2[i + 1] = x1[i + 1] + dx0 * (x2[i + 1] - x1[i + 1]) / d
            y2[i + 1] = y1[i + 1] + dx0 * (y2[i + 1] - y1[i + 1]) / d
            z2[i + 1] = z1[i + 1] + dx0 * (z2[i + 1] - z1[i + 1]) / d

            log_divergence.append(np.log(d / dx0))

        lyap = np.mean(log_divergence) / dt
        return lyap

    if st.button("Compute Complexity Indicators", key="compute"):
        hist = st.session_state.history
        if len(hist) < 200:
            st.warning("Please run the animation first to generate trajectory data!")
        else:
            with st.spinner("Calculating..."):
                xs = np.array([p[0] for p in hist])
                N = len(xs)
                dt = 0.0005

                # 1. CORRECT Lyapunov
                lyap = correct_lyapunov(xs, dt)
                lyap = round(lyap, 4)
                is_chaotic = lyap > 0.01

                # 2. Correlation Dimension
                corr_dim = correlation_dimension(xs)
                corr_dim = round(corr_dim, 4)

                # 3. 0-1 Test
                K = chaos_01_test(xs)
                K = round(K, 4)
                k_chaos = K > 0.5

                # 4. Power Spectrum
                f, Pxx = welch(xs, fs=1000, nperseg=1024)

                # 5. Phase Space Reconstruction
                psr = phase_space_reconstruction(xs, tau=1, m=2)

                # Display
                st.markdown("### Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Lyapunov Exponent", lyap if is_chaotic else "-")
                with col2:
                    st.metric("Correlation Dimension", corr_dim)
                with col3:
                    st.metric("0-1 Test K", K)

                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Chaos State", "Chaotic" if is_chaotic else "Non-chaotic")
                with col5:
                    st.metric("0-1 Test Result", "Chaotic" if k_chaos else "Non-chaotic")

                # Plot
                st.markdown("### Power Spectrum & Phase Space Reconstruction")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(f, Pxx, color='c')
                ax1.set_title("Power Spectrum")
                ax1.grid(alpha=0.3)

                ax2.plot(psr[:,0], psr[:,1], color='magenta', linewidth=0.5)
                ax2.set_title("Phase Space Reconstruction")
                ax2.axis('equal')
                st.pyplot(fig)
                plt.close(fig)

            st.success("All complexity indicators computed successfully!")