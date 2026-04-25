import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== Streamlit 页面配置 ======
st.set_page_config(page_title="信号与系统教学演示", layout="wide")

# ====== 核心修复：全平台中文支持（解决Streamlit中文乱码）======
plt.rcParams['font.sans-serif'] = ["WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示

import streamlit as st

# 全局设置中文字体
# 全局设置中文字体（包含表格、数据框全组件）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

    /* 全局字体 + 表格全场景字体统一 */
    html, body, [class*="css"], 
    table, th, td,           /* 原生表格 */
    .stTable, .stDataFrame   /* Streamlit表格/数据框 */
    {
        font-family: 'Noto Sans SC', sans-serif !important;
        font-size: 14px;     /* 可调整表格字体大小 */
    }
</style>
""", unsafe_allow_html=True)

# 测试中文显示
st.title("中文标题测试")
st.write("这是一段中文文本，使用Noto Sans SC字体显示")

# 优化页面样式
st.markdown("""
<style>
h2 {margin-bottom: 0.3rem !important;}
hr {margin-top: 0.3rem !important;}
</style>
""", unsafe_allow_html=True)

# ===================== 主页面：顶层选项卡 =====================
st.title("📚 信号与系统教学演示动画")
# 顶层标签：混叠仿真台 + 预留模块
tab_alias, tab_other = st.tabs(["🔧 混叠仿真台", "📌 其他模块（预留）"])

# ===================== 选项卡1：混叠仿真台 =====================
with tab_alias:
    st.header("混叠仿真台")
    st.divider()

    # 子标签：采样混叠演示（融合程序1完整功能） + 抗混叠滤波器演示（程序2原功能）
    sub_tab1, sub_tab2 = st.tabs(["📊 采样混叠演示", "🛡 抗混叠滤波器演示"])

    # ===================== 子标签1：采样混叠演示（完整补充程序1功能）=====================
    with sub_tab1:
        st.subheader("采样混叠演示 (奈奎斯特采样定理)")

        # 交互参数滑块
        col1, col2 = st.columns(2)
        with col1:
            f = st.slider(r"信号频率 $f\ (\rm Hz)$", 1, 50, 15, key="f_signal")
        with col2:
            fs = st.slider(r"采样频率 $f_{\rm s}\ (\rm Hz)$", 1, 50, 45, key="fs_sample")

        # ================= 时域信号计算 =================
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * f * t)
        ts = np.arange(0, 1, 1 / fs)
        xs = np.sin(2 * np.pi * f * ts)

        # ================= 图1：时域波形图 =================
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t, x, label="原始信号", color="red")
        ax1.vlines(ts, 0, xs, colors='green')
        ax1.scatter(ts, xs, color='green', label="采样点")

        f_alias = 0
        # 混叠信号计算
        if fs < 2 * f:
            f_mod = f % fs
            f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
            x_alias = np.sin(2 * np.pi * f_alias * t)
            x_alias = -np.sin(2 * np.pi * f_alias * t) if f_mod > fs / 2 else x_alias
            ax1.plot(t, x_alias, '--', color='blue', label="混叠信号")
            st.warning("⚠ 发生频率混叠！")
        else:
            st.success("✅ 满足奈奎斯特采样定理！")

        ax1.set_title("时域信号波形")
        ax1.set_xlabel("时间(s)")
        ax1.set_ylabel("幅值")
        ax1.legend(loc="upper right")
        ax1.grid(True)
        st.pyplot(fig1)

        # ================= 频谱计算函数 =================
        def fft_signal(sig, t):
            N = len(sig)
            dt = t[1] - t[0]
            freq = np.fft.fftfreq(N, dt)
            spectrum = np.abs(np.fft.fft(sig)) / N
            return np.fft.fftshift(freq), np.fft.fftshift(spectrum)

        # 原信号频谱
        freq_x, X = fft_signal(x, t)
        # 采样信号频谱
        x_sampled = np.zeros_like(t)
        indices = (ts / (t[1] - t[0])).astype(int)
        indices = indices[indices < len(t)]
        x_sampled[indices] = xs[:len(indices)]
        freq_s, Xs = fft_signal(x_sampled, t)

        # ================= 图2：原始信号频谱 =================
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(freq_x, X, label="原始信号频谱", color='red')
        ax2.set_xlim(-60, 60)
        ax2.set_title("原始信号频谱")
        ax2.set_xlabel("频率(Hz)")
        ax2.set_ylabel("幅值")
        ax2.grid(True)
        ax2.legend(loc="upper right")
        st.pyplot(fig2)

        # ================= 图3：采样信号频谱 =================
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(freq_s, Xs, label="采样信号频谱", color='blue')
        ax3.set_xlim(-60, 60)
        ymax = np.max(Xs)
        ax3.set_ylim(0, ymax * 1.2)

        # 混叠区域标注
        if fs < 2 * f:
            f_overlap_start = fs - f
            f_overlap_end = f
            if f_overlap_start < f_overlap_end:
                ax3.fill_betweenx([0, ymax], f_overlap_start, f_overlap_end, color='blue', alpha=0.2, label="混叠区域")
                ax3.fill_betweenx([0, ymax], -f_overlap_end, -f_overlap_start, color='blue', alpha=0.2)

        # 频率虚线标注函数
        def vline(x, label=None):
            ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')
            if label:
                ax3.text(x, ymax * 1.05, label, ha='center')

        # 标注关键频率
        vline(f, r"$+f$")
        vline(-f, r"$-f$")
        vline(fs, r"$+f_s$")
        vline(-fs, r"$-f_s$")

        ax3.set_title("周期采样信号频谱")
        ax3.set_xlabel("频率(Hz)")
        ax3.set_ylabel("幅值")
        ax3.grid(True)
        ax3.legend(loc="upper right")
        st.pyplot(fig3)

    # ===================== 子标签2：抗混叠滤波器演示（保留原功能）=====================
    with sub_tab2:
        st.subheader("抗混叠滤波器演示")
        st.markdown("**原理**：采样前通过低通滤波器滤除高频分量，从根源避免混叠")

        # 交互参数
        col1, col2, col3 = st.columns(3)
        with col1:
            f_main = st.slider("主信号频率 (Hz)", 1, 20, 10, key="f_main")
        with col2:
            f_noise = st.slider("高频干扰频率 (Hz)", 45, 100, 60, key="f_noise")
        with col3:
            fs = st.slider("采样频率 fs (Hz)", 75, 100, 80, key="fs_filter")

        fc = st.slider("滤波器截止频率 (Hz)", 25, 40, 30, key="fc")

        # 信号生成
        t = np.linspace(0, 1, 1000)
        x_mix = np.sin(2 * np.pi * f_main * t) + 0.3 * np.sin(2 * np.pi * f_noise * t)
        x_filtered = np.sin(2 * np.pi * f_main * t)

        # 采样点
        ts = np.arange(0, 1, 1 / fs)
        xs_mix = np.sin(2 * np.pi * f_main * ts) + 0.3 * np.sin(2 * np.pi * f_noise * ts)
        xs_filtered = np.sin(2 * np.pi * f_main * ts)

        # 绘图对比
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        # 未滤波
        ax1.plot(t, x_mix, color="red", label="混合信号（含高频干扰）")
        ax1.scatter(ts, xs_mix, color="green", label="采样点")
        ax1.set_title("无抗混叠滤波器（易发生混叠）")
        ax1.set_ylabel("幅值")
        ax1.grid(True)
        ax1.legend(loc="upper right")
        # 滤波后
        ax2.plot(t, x_filtered, color="blue", label="滤波后信号（无高频干扰）")
        ax2.scatter(ts, xs_filtered, color="green", label="采样点")
        ax2.set_title("添加抗混叠滤波器（消除混叠）")
        ax2.set_xlabel("时间(s)")
        ax2.set_ylabel("幅值")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        st.pyplot(fig)

        # 效果提示
        if f_noise > fs / 2:
            st.success("✅ 抗混叠滤波器已移除高频干扰，彻底避免混叠！")
        else:
            st.info("ℹ️ 干扰频率低于奈奎斯特频率，无明显混叠")

# ===================== 选项卡2：预留模块 =====================
with tab_other:
    st.info("ℹ️ 此处可扩展其他信号与系统教学演示模块")
