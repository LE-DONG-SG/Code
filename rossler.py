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
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)

                x += dx * dt
                y += dy * dt
                z += dz * dt
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

    st.markdown("""
    Computed indicators:
    - Maximum Lyapunov Exponent (chaos core indicator)
    - Correlation Dimension (fractal dimension)
    - Power Spectrum
    - Phase Space Reconstruction
    - 0-1 Test for Chaos
    """)

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

    if st.button("Compute Complexity Indicators", key="compute"):
        hist = st.session_state.history
        if len(hist) < 200:
            st.warning("Please run the animation first to generate trajectory data!")
        else:
            with st.spinner("Calculating..."):
                xs = np.array([p[0] for p in hist])
                N = len(xs)

                # 1. Maximum Lyapunov Exponent
                dxs = np.diff(xs)
                dxs = dxs[np.abs(dxs) > 1e-9]
                lyap = np.mean(np.log(np.abs(dxs)))
                lyap = round(lyap, 4)
                is_chaotic = lyap > 0.01

                # 2. Correlation Dimension
                corr_dim = correlation_dimension(xs)
                corr_dim = round(corr_dim, 4)

                # 3. 0-1 Chaos Test
                K = chaos_01_test(xs)
                K = round(K, 4)
                k_chaos = K > 0.5

                # 4. Power Spectrum
                f, Pxx = welch(xs, fs=1000, nperseg=1024)

                # 5. Phase Space Reconstruction
                psr = phase_space_reconstruction(xs, tau=10, m=2)

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
                ax1.set_xlabel("Frequency")
                ax1.set_ylabel("Power")
                ax1.grid(alpha=0.3)

                ax2.plot(psr[:,0], psr[:,1], color='magenta', linewidth=0.5)
                ax2.set_title("Phase Space Reconstruction")
                ax2.axis('equal')
                st.pyplot(fig)
                plt.close(fig)

            st.success("All complexity indicators computed successfully!")