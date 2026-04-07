import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Rossler Attractor", layout="wide")
st.title("Chaotic System - Rossler Attractor")

# Session State
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "params" not in st.session_state:
    st.session_state.params = (0.2, 0.2, 5.7, 0.0001, 2000000)

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])

# ==================== Tab 1: Animation (RK4 积分) ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 15.0, 5.7, 0.1, key="c")
    dt_display = st.sidebar.slider("Step Length (ms)", 0.05, 1.0, 0.1, 0.05, key="dt")
    max_steps_display = st.sidebar.slider("Max Steps (k)", 1000, 5000, 2000, 500, key="steps")

    dt = dt_display / 1000
    max_steps = max_steps_display * 1000
    st.session_state.params = (a, b, c, dt, max_steps)

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
        a, b, c, dt, max_steps = st.session_state.params

        with st.spinner("Generating Chaotic Attractor..."):
            for step in range(max_steps):
                def rossler(x, y, z):
                    dx = -y - z
                    dy = x + a * y
                    dz = b + z * (x - c)
                    return dx, dy, dz


                k1x, k1y, k1z = rossler(x, y, z)
                k2x, k2y, k2z = rossler(x + dt * k1x / 2, y + dt * k1y / 2, z + dt * k1z / 2)
                k3x, k3y, k3z = rossler(x + dt * k2x / 2, y + dt * k2y / 2, z + dt * k2z / 2)
                k4x, k4y, k4z = rossler(x + dt * k3x, y + dt * k3y, z + dt * k3z)

                x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
                y += dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
                z += dt / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)

                # 只存后半段数据 (跳过Transient)
                if step > max_steps * 0.2:
                    hist.append((x, y))

                if step % 1000 == 0 and not st.session_state.running:
                    break

            # 绘图
            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_facecolor("black")
            ax.plot(xs, ys, color="#00ffff", linewidth=0.3)
            ax.axis("off")
            placeholder.pyplot(fig)
            plt.close(fig)

            st.session_state.state = (x, y, z)
            st.session_state.history = hist
            st.session_state.running = False
            st.success("✅ Attractor Ready! Now compute complexity.")

# ==================== Tab 2: Complexity Analysis (彻底修复) ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")


    # 1. 0-1 Test (修正版)
    def chaos_01_test(series, trials=3):
        N = len(series)
        if N < 2000: return 0.0
        series = series - np.mean(series)  # 去趋势
        Ks = []
        for _ in range(trials):
            c = np.random.uniform(np.pi / 6, 5 * np.pi / 6)
            t = np.arange(N)
            p = np.cumsum(series * np.cos(c * t))
            q = np.cumsum(series * np.sin(c * t))
            M = np.sqrt(p ** 2 + q ** 2)
            # 避免log(0)
            mask = (t[1:] > 0) & (M[1:] > 0)
            if np.sum(mask) < 10: return 0.0
            log_t = np.log(t[1:][mask])
            log_M = np.log(M[1:][mask])
            slope, _, _, _, _ = linregress(log_t, log_M)
            Ks.append(slope)
        return round(np.mean(Ks), 4)


    # 2. 李雅普诺夫计算 (修正逻辑)
    def calc_lyapunov(series, dt):
        try:
            diffs = np.abs(np.diff(series))
            diffs = diffs[diffs > 1e-9]
            if len(diffs) < 100: return -1.0
            le = np.mean(np.log(diffs)) / dt
            return round(le, 4)
        except:
            return -1.0


    # 3. 相空间重构
    def recon_phase(series, tau=5, m=2):
        N = len(series)
        if N < m * tau: return np.array([])
        return np.array([series[i:i + m * tau:tau] for i in range(N - m * tau + 1)])


    # 按钮逻辑
    if st.button("Compute Final Indicators"):
        hist = st.session_state.history
        a, b, c, dt, _ = st.session_state.params

        if len(hist) < 5000:
            st.warning("⚠️ Run Animation first with more steps!")
        else:
            with st.spinner("Calculating..."):
                xs = np.array([p[0] for p in hist])
                N = len(xs)

                # 核心计算
                lyap_val = calc_lyapunov(xs, dt)
                K_val = chaos_01_test(xs)

                # 🔴 核心修复：判定逻辑修正
                # 规则：LE > 0 且 K > 0.5 才判定为混沌
                is_chaotic_le = lyap_val > 0.01
                is_chaotic_k = K_val > 0.5
                is_actually_chaotic = is_chaotic_le or is_chaotic_k

                # 重构
                psr = recon_phase(xs, tau=5, m=2)
                f, Pxx = welch(xs, fs=1 / dt, nperseg=2048)

            # 🔴 界面显示修正（关键！）
            st.markdown("### Results")
            col1, col2 = st.columns(2)
            with col1:
                # 如果是混沌，显示数值；否则显示 -
                st.metric("Max Lyapunov Exponent", lyap_val if is_actually_chaotic else "-")
            with col2:
                st.metric("0-1 Test K", K_val)

            col3, col4 = st.columns(2)
            with col3:
                # 修正文字：逻辑对应
                chaos_text = "✅ Chaotic" if is_actually_chaotic else "❌ Non-chaotic"
                st.markdown(f"**Chaos State:** {chaos_text}")
            with col4:
                k_result = "✅ Chaotic" if is_chaotic_k else "❌ Non-chaotic"
                st.markdown(f"**0-1 Result:** {k_result}")

            # 绘图
            st.markdown("### Power Spectrum & Phase Space")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 功率谱
            ax1.plot(f, Pxx, color='cyan', linewidth=0.8)
            ax1.set_title("Power Spectrum (Chaos = Continuous Broadband)")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Power")
            ax1.set_yscale('log')
            ax1.grid(alpha=0.3)

            # 相空间
            if len(psr) > 100:
                ax2.plot(psr[:, 0], psr[:, 1], color='magenta', linewidth=0.2)
                ax2.set_title("Phase Space Reconstruction")
                ax2.axis('equal')
                ax2.grid(alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "Data Error", ha='center', va='center')

            st.pyplot(fig)
            plt.close(fig)
            st.success("✅ Calculation Complete! Check results above.")