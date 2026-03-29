import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== Streamlit 页面配置 ======
st.set_page_config(page_title="信号与系统教学演示", layout="wide")

# 缩小标题与分割线行距的 CSS
st.markdown("""
<style>
h2 {
    margin-bottom: 0.3rem !important;
}
hr {
    margin-top: 0.3rem !important;
}
</style>
""", unsafe_allow_html=True)

# ===================== 主页面：顶层选项卡 =====================
st.title("📚 信号与系统教学演示动画")
# 创建顶层选项卡：混叠仿真台 + 预留其他教学模块
tab_alias, tab_other = st.tabs(["🔧 混叠仿真台", "📌 其他模块（预留）"])

# ===================== 选项卡1：混叠仿真台 =====================
with tab_alias:
    st.header("混叠仿真台")
    st.divider()

    # ========== 混叠仿真台内部：子选项卡（核心升级） ==========
    sub_tab1, sub_tab2 = st.tabs(["📊 采样混叠演示", "🛡 抗混叠滤波器演示"])

    # ===================== 子选项卡1：采样混叠演示（原代码） =====================
    with sub_tab1:
        st.subheader("采样混叠演示 (Nyquist Theorem)")
        # ====== 中文字体修复 ======
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

        # ====== 交互滑块 ======
        col1, col2 = st.columns(2)
        with col1:
            f = st.slider("信号频率 f (Hz)", min_value=1, max_value=50, value=15, key="f1")
        with col2:
            fs = st.slider("采样频率 fs (Hz)", min_value=1, max_value=50, value=45, key="fs1")

        # ====== 计算信号 ======
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * f * t)

        # 采样点
        ts = np.arange(0, 1, 1 / fs)
        xs = np.sin(2 * np.pi * f * ts)

        # ====== 绘图 ======
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, x, label="Original Signal", color="red")
        ax.vlines(ts, 0, xs, colors='green')
        ax.scatter(ts, xs, color='green', label="Sampling Points")

        # 混叠判断
        if fs < 2 * f:
            f_mod = f % fs
            f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
            x_alias = np.sin(2 * np.pi * f_alias * t)
            x_alias = -np.sin(2 * np.pi * f_alias * t) if f_mod > fs / 2 else x_alias
            if np.max(np.abs(xs)) < 1e-6:
                x_alias = np.zeros_like(t)
            ax.plot(t, x_alias, '--', color='blue', label="Aliased Signal")
            st.warning("⚠ 发生混叠 (Aliasing Occurred)")
        else:
            st.success("✅ 满足采样定理 (Satisfy Nyquist Theorem)")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        # ===================== 子选项卡2：抗混叠滤波器演示（新增功能） =====================
    with sub_tab2:
        st.subheader("抗混叠滤波器演示 (Anti-aliasing Filter)")
        st.markdown("**原理**：采样前通过低通滤波器滤除高频分量，从根源避免混叠")

        # ====== 交互参数 ======
        col1, col2, col3 = st.columns(3)
        with col1:
            f_main = st.slider("主信号频率 (Hz)", 1, 30, 10, key="f_main")
        with col2:
            f_noise = st.slider("高频干扰频率 (Hz)", 40, 100, 60, key="f_noise")
        with col3:
            fs = st.slider("采样频率 fs (Hz)", 1, 50, 25, key="fs_filter")

        fc = st.slider("滤波器截止频率 (Hz)", 1, 50, 20, key="fc")

        # ====== 信号生成 ======
        t = np.linspace(0, 1, 1000)
        # 混合信号：主信号 + 高频干扰（会导致混叠）
        x_mix = np.sin(2 * np.pi * f_main * t) + 0.3 * np.sin(2 * np.pi * f_noise * t)
        # 理想低通抗混叠滤波
        x_filtered = np.sin(2 * np.pi * f_main * t)  # 滤除高频干扰

        # 采样点
        ts = np.arange(0, 1, 1 / fs)
        xs_mix = np.sin(2 * np.pi * f_main * ts) + 0.3 * np.sin(2 * np.pi * f_noise * ts)
        xs_filtered = np.sin(2 * np.pi * f_main * ts)

        # ====== 绘图对比 ======
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 子图1：未滤波 - 直接采样
        ax1.plot(t, x_mix, color="red", label="混合信号(含高频干扰)")
        ax1.scatter(ts, xs_mix, color="green", label="采样点")
        ax1.set_title("未使用抗混叠滤波器（易发生混叠）")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)
        ax1.legend()

        # 子图2：滤波后 - 采样
        ax2.plot(t, x_filtered, color="blue", label="滤波后信号(无高频干扰)")
        ax2.scatter(ts, xs_filtered, color="green", label="采样点")
        ax2.set_title("使用抗混叠滤波器（消除混叠）")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        ax2.legend()

        st.pyplot(fig)

        # 效果说明
        if f_noise > fs / 2:
            st.success("✅ 抗混叠滤波器已滤除高频干扰，避免混叠！")
        else:
            st.info("ℹ️ 干扰频率低于奈奎斯特频率，无明显混叠")

        # ===================== 选项卡2：预留模块 =====================
with tab_other:
    st.info("ℹ️ 此处可扩展其他信号与系统教学演示模块")