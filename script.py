import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.font_manager import FontProperties

# ====== 固定中文字体（只给图表用，不影响页面）======
font = FontProperties(family='SimHei', size=12)  # 黑体，Windows/mac通用

# ====== Streamlit 页面配置 ======
st.set_page_config(page_title="信号与系统教学演示", layout="wide")

# 页面中文（不动）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Noto Sans SC', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 优化样式
st.markdown("""
<style>
h2 {margin-bottom: 0.3rem !important;}
hr {margin-top: 0.3rem !important;}
</style>
""", unsafe_allow_html=True)

# ===================== 主页面 =====================
st.title("📚 信号与系统教学演示动画")
tab_alias, tab_other = st.tabs(["🔧 混叠仿真台", "📌 其他模块（预留）"])

with tab_alias:
    st.header("混叠仿真台")
    st.divider()
    sub_tab1, sub_tab2 = st.tabs(["📊 采样混叠演示", "🛡 抗混叠滤波器演示"])

    with sub_tab1:
        st.subheader("采样混叠演示 (奈奎斯特采样定理)")
        col1, col2 = st.columns(2)
        with col1:
            f = st.slider(r"信号频率 $f\ (\rm Hz)$", 1, 50, 15, key="f_signal")
        with col2:
            fs = st.slider(r"采样频率 $f_{\rm s}\ (\rm Hz)$", 1, 50, 45, key="fs_sample")

        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * f * t)
        ts = np.arange(0, 1, 1 / fs)
        xs = np.sin(2 * np.pi * f * ts)

        # ================= 图1 手动加字体！=================
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t, x, label="原始信号", color="red")
        ax1.vlines(ts, 0, xs, colors='green')
        ax1.scatter(ts, xs, color='green', label="采样点")

        f_alias = 0
        if fs < 2 * f:
            f_mod = f % fs
            f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
            x_alias = np.sin(2 * np.pi * f_alias * t)
            x_alias = -np.sin(2 * np.pi * f_alias * t) if f_mod > fs / 2 else x_alias
            ax1.plot(t, x_alias, '--', color='blue', label="混叠信号")
            st.warning("⚠ 发生频率混叠！")
        else:
            st.success("✅ 满足奈奎斯特采样定理！")

        # ✅ 就是你要的写法：ax.set_title(..., fontproperties=font)
        ax1.set_title("时域信号波形", fontproperties=font)
        ax1.set_xlabel("时间(s)", fontproperties=font)
        ax1.set_ylabel("幅值", fontproperties=font)
        ax1.legend(prop=font)
        ax1.grid(True)
        st.pyplot(fig1)

        # 频谱函数
        def fft_signal(sig, t):
            N = len(sig)
            dt = t[1] - t[0]
            freq = np.fft.fftfreq(N, dt)
            spectrum = np.abs(np.fft.fft(sig)) / N
            return np.fft.fftshift(freq), np.fft.fftshift(spectrum)

        freq_x, X = fft_signal(x, t)
        x_sampled = np.zeros_like(t)
        indices = (ts / (t[1] - t[0])).astype(int)
        indices = indices[indices < len(t)]
        x_sampled[indices] = xs[:len(indices)]
        freq_s, Xs = fft_signal(x_sampled, t)

        # ================= 图2 手动加字体！=================
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(freq_x, X, label="原始信号频谱", color='red')
        ax2.set_xlim(-60, 60)
        # ✅ 严格按你要求
        ax2.set_title("原始信号频谱", fontproperties=font)
        ax2.set_xlabel("频率(Hz)", fontproperties=font)
        ax2.set_ylabel("幅值", fontproperties=font)
        ax2.legend(prop=font)
        ax2.grid(True)
        st.pyplot(fig2)

        # ================= 图3 手动加字体！=================
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(freq_s, Xs, label="采样信号频谱", color='blue')
        ax3.set_xlim(-60, 60)
        ymax = np.max(Xs)
        ax3.set_ylim(0, ymax * 1.2)

        if fs < 2 * f:
            f_overlap_start = fs - f
            f_overlap_end = f
            if f_overlap_start < f_overlap_end:
                ax3.fill_betweenx([0, ymax], f_overlap_start, f_overlap_end, color='blue', alpha=0.2, label="混叠区域")
                ax3.fill_betweenx([0, ymax], -f_overlap_end, -f_overlap_start, color='blue', alpha=0.2)

        def vline(x):
            ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')

        vline(f)
        vline(-f)
        vline(fs)
        vline(-fs)

        # ✅ 严格按你要求
        ax3.set_title("周期采样信号频谱", fontproperties=font)
        ax3.set_xlabel("频率(Hz)", fontproperties=font)
        ax3.set_ylabel("幅值", fontproperties=font)
        ax3.legend(prop=font)
        ax3.grid(True)
        st.pyplot(fig3)

    with sub_tab2:
        st.subheader("抗混叠滤波器演示")
        st.markdown("**原理**：采样前通过低通滤波器滤除高频分量，从根源避免混叠")

        col1, col2, col3 = st.columns(3)
        with col1:
            f_main = st.slider("主信号频率 (Hz)", 1, 20, 10, key="f_main")
        with col2:
            f_noise = st.slider("高频干扰频率 (Hz)", 45, 100, 60, key="f_noise")
        with col3:
            fs = st.slider("采样频率 fs (Hz)", 75, 100, 80, key="fs_filter")

        fc = st.slider("滤波器截止频率 (Hz)", 25, 40, 30, key="fc")

        t = np.linspace(0, 1, 1000)
        x_mix = np.sin(2 * np.pi * f_main * t) + 0.3 * np.sin(2 * np.pi * f_noise * t)
        x_filtered = np.sin(2 * np.pi * f_main * t)

        ts = np.arange(0, 1, 1 / fs)
        xs_mix = np.sin(2 * np.pi * f_main * ts) + 0.3 * np.sin(2 * np.pi * f_noise * ts)
        xs_filtered = np.sin(2 * np.pi * f_main * ts)

        # ================= 图4 手动加字体！=================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(t, x_mix, color="red", label="混合信号（含高频干扰）")
        ax1.scatter(ts, xs_mix, color="green", label="采样点")
        ax1.set_title("无抗混叠滤波器（易发生混叠）", fontproperties=font)
        ax1.set_ylabel("幅值", fontproperties=font)
        ax1.legend(prop=font)
        ax1.grid(True)

        ax2.plot(t, x_filtered, color="blue", label="滤波后信号（无高频干扰）")
        ax2.scatter(ts, xs_filtered, color="green", label="采样点")
        ax2.set_title("添加抗混叠滤波器（消除混叠）", fontproperties=font)
        ax2.set_xlabel("时间(s)", fontproperties=font)
        ax2.set_ylabel("幅值", fontproperties=font)
        ax2.legend(prop=font)
        ax2.grid(True)

        st.pyplot(fig)

        if f_noise > fs / 2:
            st.success("✅ 抗混叠滤波器已移除高频干扰，彻底避免混叠！")
        else:
            st.info("ℹ️ 干扰频率低于奈奎斯特频率，无明显混叠")

with tab_other:
    st.info("ℹ️ 此处可扩展其他信号与系统教学演示模块")