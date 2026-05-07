import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== Streamlit Page Configuration (English, No Chinese Fonts) ======
st.set_page_config(page_title="Signals & Systems Demo", layout="wide")
plt.rcParams["axes.unicode_minus"] = False

# ===================== Main Page =====================
st.title("📚 Signals & Systems Teaching Demonstration")
tab_alias, tab_other = st.tabs(["🔧 Sampling Alias Simulator", "📌 Other Modules (Reserved)"])

with tab_alias:
    st.header("Sampling Alias Simulator")
    st.divider()
    sub_tab1, sub_tab2 = st.tabs(["📊 Sampling Aliasing Demo", "🛡 Anti-Aliasing Filter Demo"])

    with sub_tab1:
        # ==============================================
        # 新增：信号类型选择下拉框（二选一展示）
        # ==============================================
        st.subheader("1. Signal Sampling & Aliasing")
        signal_type = st.selectbox(
            "Select Signal Type",
            options=["Sine Wave (正弦波)", "Sinc Signal (Sinc信号)"],
            index=0,
            key="signal_selector"
        )

        # 根据选择的信号类型，生成对应参数、时间轴、信号
        if signal_type == "Sine Wave (正弦波)":
            # 正弦信号参数
            col1, col2 = st.columns(2)
            with col1:
                f = st.slider(r"Signal Frequency $f\ (\rm Hz)$", 1, 50, 15, key="f_signal")
            with col2:
                fs = st.slider(r"Sampling Frequency $f_{\rm s}\ (\rm Hz)$", 1, 50, 45, key="fs_sample")

            # 正弦信号时间轴 + 信号生成
            t = np.linspace(0, 1, 1000)
            ts = np.arange(0, 1, 1 / fs)
            x = np.sin(2 * np.pi * f * t)
            xs = np.sin(2 * np.pi * f * ts)
            plot_label = "Original Sine Signal"

        else:
            # Sinc信号参数
            col1, col2 = st.columns(2)
            with col1:
                f = st.slider(r"Sinc Signal Frequency $f\ (\rm Hz)$", 1, 50, 15, key="f_sinc")
            with col2:
                fs = st.slider(r"Sinc Sampling Frequency $f_{\rm s}\ (\rm Hz)$", 1, 50, 25, key="fs_sinc")


            # Sinc函数定义
            def sinc_signal(t, f0):
                y = np.zeros_like(t)
                nonzero = t != 0
                y[nonzero] = np.sin(2 * np.pi * f0 * t[nonzero]) / (np.pi * t[nonzero])
                y[~nonzero] = 2 * f0
                return y


            # Sinc信号时间轴 + 信号生成
            t = np.linspace(-0.5, 0.5, 3000)
            ts = np.arange(-0.5, 0.5, 1 / fs)
            x = sinc_signal(t, f)
            xs = sinc_signal(ts, f)
            plot_label = "Original Sinc Signal"

        # ==============================================
        # 统一：时域波形绘制
        # ==============================================
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t, x, label=plot_label, color="red")
        ax1.vlines(ts, 0, xs, colors='green')
        ax1.scatter(ts, xs, color='green', label="Sampling Points")

        f_alias = 0
        if fs < 2 * f:
            f_mod = f % fs
            f_alias = fs - f_mod if f_mod > fs / 2 else f_mod

            # 混叠信号生成（区分正弦/Sinc）
            if signal_type == "Sine Wave (正弦波)":
                x_alias = np.sin(2 * np.pi * f_alias * t)
                x_alias = -x_alias if f_mod > fs / 2 else x_alias
            else:
                def sinc_signal(t, f0):
                    y = np.zeros_like(t)
                    nonzero = t != 0
                    y[nonzero] = np.sin(2 * np.pi * f0 * t[nonzero]) / (np.pi * t[nonzero])
                    y[~nonzero] = 2 * f0
                    return y


                x_alias = sinc_signal(t, f_alias)
                sign = np.sign(xs[1] / (sinc_signal(ts, f_alias)[1] + 1e-8))
                x_alias *= sign

            ax1.plot(t, x_alias, '--', color='blue', label="Aliased Signal")
            st.warning("⚠ Frequency Aliasing Occurred!")
        else:
            st.success("✅ Satisfies Nyquist Sampling Theorem!")

        ax1.set_title(f"Time Domain Signal Waveform ({signal_type.split(' ')[0]})")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)


        # ==============================================
        # 统一：FFT 频谱分析函数
        # ==============================================
        def fft_signal(sig, t):
            N = len(sig)
            dt = t[1] - t[0]
            freq = np.fft.fftfreq(N, dt)
            spectrum = np.abs(np.fft.fft(sig)) / N
            return np.fft.fftshift(freq), np.fft.fftshift(spectrum)


        # 原始信号频谱
        freq_x, X = fft_signal(x, t)
        x_sampled = np.zeros_like(t)
        indices = (ts / (t[1] - t[0])).astype(int)
        indices = indices[indices < len(t)]
        x_sampled[indices] = xs[:len(indices)]
        freq_s, Xs = fft_signal(x_sampled, t)

        # ==============================================
        # 统一：原始信号频谱图
        # ==============================================
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(freq_x, X, label=f"Original {signal_type.split(' ')[0]} Spectrum", color='red')
        ax2.set_xlim(-60, 60)
        ax2.set_title(f"Original Signal Spectrum ({signal_type.split(' ')[0]})")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # ==============================================
        # 统一：采样信号频谱图
        # ==============================================
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(freq_s, Xs, label=f"Sampled {signal_type.split(' ')[0]} Spectrum", color='blue')
        ax3.set_xlim(-60, 60)
        ymax = np.max(Xs)
        ax3.set_ylim(0, ymax * 1.2)

        # 混叠区域标注
        if fs < 2 * f:
            f_overlap_start = fs - f
            f_overlap_end = f
            if f_overlap_start < f_overlap_end:
                ax3.fill_betweenx([0, ymax], f_overlap_start, f_overlap_end, color='blue', alpha=0.2,
                                  label="Aliasing Region")
                ax3.fill_betweenx([0, ymax], -f_overlap_end, -f_overlap_start, color='blue', alpha=0.2)


        # 频率虚线标注
        def vline(x):
            ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')


        vline(f)
        vline(-f)
        vline(fs)
        vline(-fs)

        ax3.set_title(f"Sampled Signal Spectrum ({signal_type.split(' ')[0]})")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude")
        ax3.legend(loc="upper right")
        ax3.grid(True)
        st.pyplot(fig3)

    with sub_tab2:
        st.subheader("Anti-Aliasing Filter Demonstration")
        st.markdown(
            "**Principle**: Filter out high-frequency components before sampling to avoid aliasing fundamentally.")

        col1, col2, col3 = st.columns(3)
        with col1:
            f_main = st.slider("Main Signal Frequency (Hz)", 1, 20, 10, key="f_main")
        with col2:
            f_noise = st.slider("High-Frequency Noise Frequency (Hz)", 45, 100, 60, key="f_noise")
        with col3:
            fs = st.slider("Sampling Frequency fs (Hz)", 75, 100, 80, key="fs_filter")

        fc = st.slider("Filter Cut-off Frequency (Hz)", 25, 40, 30, key="fc")

        # Generate Signals
        t = np.linspace(0, 1, 1000)
        x_mix = np.sin(2 * np.pi * f_main * t) + 0.3 * np.sin(2 * np.pi * f_noise * t)
        x_filtered = np.sin(2 * np.pi * f_main * t)

        ts = np.arange(0, 1, 1 / fs)
        xs_mix = np.sin(2 * np.pi * f_main * ts) + 0.3 * np.sin(2 * np.pi * f_noise * ts)
        xs_filtered = np.sin(2 * np.pi * f_main * ts)

        # Figure 4: Anti-Aliasing Filter Comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(t, x_mix, color="red", label="Mixed Signal (with High-Frequency Noise)")
        ax1.scatter(ts, xs_mix, color="green", label="Sampling Points")
        ax1.set_title("Without Anti-Aliasing Filter (Prone to Aliasing)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(t, x_filtered, color="blue", label="Filtered Signal (No High-Frequency Noise)")
        ax2.scatter(ts, xs_filtered, color="green", label="Sampling Points")
        ax2.set_title("With Anti-Aliasing Filter (Eliminate Aliasing)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)

        # Prompt Messages
        if f_noise > fs / 2:
            st.success("✅ Anti-aliasing filter removed high-frequency noise, completely avoiding aliasing!")
        else:
            st.info("ℹ️ Noise frequency is below Nyquist frequency, no obvious aliasing.")

with tab_other:
    st.info("ℹ️ More signals and systems teaching modules can be extended here.")