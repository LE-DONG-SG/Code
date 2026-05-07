import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(page_title="Signals & Systems Demo", layout="wide")
plt.rcParams["axes.unicode_minus"] = False

# Main Page
st.title("📚 Signals & Systems Teaching Demonstration")
tab_alias, tab_other = st.tabs(["🔧 Sampling Alias Simulator", "📌 Other Modules (Reserved)"])

with tab_alias:
    st.header("Sampling Alias Simulator")
    st.divider()
    sub_tab1, sub_tab2 = st.tabs(["📊 Sampling Aliasing Demo", "🛡 Anti-Aliasing Filter Demo"])

    with sub_tab1:
        # Signal type selector
        st.subheader("Signal Sampling & Aliasing")
        select_signal = st.selectbox(
            "Please choose signal type",
            ["Sine Wave", "Sinc Signal", "Square Wave"],
            key="sig_select"
        )
        st.divider()

        # Sine Wave Module
        if select_signal == "Sine Wave":
            st.subheader("Sine Wave Sampling & Aliasing")
            col1, col2 = st.columns(2)
            with col1:
                f = st.slider(r"Signal Frequency $f\ (\rm Hz)$", 1, 50, 15, key="f_signal")
            with col2:
                fs = st.slider(r"Sampling Frequency $f_{\rm s}\ (\rm Hz)$", 1, 50, 45, key="fs_sample")

            # Time domain signal generation
            t = np.linspace(0, 1, 1000)
            x = np.sin(2 * np.pi * f * t)
            ts = np.arange(0, 1, 1 / fs)
            xs = np.sin(2 * np.pi * f * ts)

            # Time domain waveform plot
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(t, x, label="Original Sine Signal", color="red")
            ax1.vlines(ts, 0, xs, colors='green')
            ax1.scatter(ts, xs, color='green', label="Sampling Points")

            f_alias = 0
            if fs < 2 * f:
                f_mod = f % fs
                f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
                x_alias = np.sin(2 * np.pi * f_alias * t)
                x_alias = -np.sin(2 * np.pi * f_alias * t) if f_mod > fs / 2 else x_alias
                ax1.plot(t, x_alias, '--', color='blue', label="Aliased Signal")
                st.warning("⚠ Frequency Aliasing Occurred!")
            else:
                st.success("✅ Satisfies Nyquist Sampling Theorem!")

            ax1.set_title("Time Domain Signal Waveform (Sine)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.legend(loc="upper right")
            ax1.grid(True)
            st.pyplot(fig1)

            # FFT calculate function
            def fft_signal(sig, t):
                N = len(sig)
                dt = t[1] - t[0]
                freq = np.fft.fftfreq(N, dt)
                spectrum = np.abs(np.fft.fft(sig)) / N
                return np.fft.fftshift(freq), np.fft.fftshift(spectrum)

            # Original signal spectrum
            freq_x, X = fft_signal(x, t)
            x_sampled = np.zeros_like(t)
            indices = (ts / (t[1] - t[0])).astype(int)
            indices = indices[indices < len(t)]
            x_sampled[indices] = xs[:len(indices)]
            freq_s, Xs = fft_signal(x_sampled, t)

            # Original spectrum figure
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(freq_x, X, label="Original Signal Spectrum", color='red')
            ax2.set_xlim(-60, 60)
            ax2.set_title("Original Signal Spectrum (Sine)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude")
            ax2.legend(loc="upper right")
            ax2.grid(True)
            st.pyplot(fig2)

            # Sampled signal spectrum figure
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(freq_s, Xs, label="Sampled Signal Spectrum", color='blue')
            ax3.set_xlim(-60, 60)
            ymax = np.max(Xs)
            ax3.set_ylim(0, ymax * 1.2)

            # Mark aliasing region
            if fs < 2 * f:
                f_overlap_start = fs - f
                f_overlap_end = f
                if f_overlap_start < f_overlap_end:
                    ax3.fill_betweenx([0, ymax], f_overlap_start, f_overlap_end, color='blue', alpha=0.2, label="Aliasing Region")
                    ax3.fill_betweenx([0, ymax], -f_overlap_end, -f_overlap_start, color='blue', alpha=0.2)

            # Vertical dashed frequency line
            def vline(x):
                ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')
            vline(f)
            vline(-f)
            vline(fs)
            vline(-fs)

            ax3.set_title("Sampled Signal Spectrum (Periodic Extension)")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Magnitude")
            ax3.legend(loc="upper right")
            ax3.grid(True)
            st.pyplot(fig3)

        # Sinc Signal Module
        elif select_signal == "Sinc Signal":
            st.subheader("Sinc Signal Sampling & Aliasing")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                f_sinc = st.slider(r"Sinc Signal Frequency $f\ (\rm Hz)$", 1, 50, 15, key="f_sinc")
            with col_s2:
                fs_sinc = st.slider(r"Sinc Sampling Frequency $f_{\rm s}\ (\rm Hz)$", 1, 50, 25, key="fs_sinc")

            # Sinc function definition
            def sinc_signal(t, f0):
                y = np.zeros_like(t)
                nonzero = t != 0
                y[nonzero] = np.sin(2 * np.pi * f0 * t[nonzero]) / (np.pi * t[nonzero])
                y[~nonzero] = 2 * f0
                return y

            # Symmetric time axis for sinc signal
            t_sinc = np.linspace(-0.5, 0.5, 3000)
            ts_sinc = np.arange(-0.5, 0.5, 1 / fs_sinc)
            x_sinc = sinc_signal(t_sinc, f_sinc)
            xs_sinc = sinc_signal(ts_sinc, f_sinc)

            # Time domain plot for sinc
            fig_s1, ax_s1 = plt.subplots(figsize=(10, 5))
            ax_s1.plot(t_sinc, x_sinc, label="Original Sinc Signal", lw=1, color="red")
            ax_s1.vlines(ts_sinc, 0, xs_sinc, lw=1, colors='green')
            ax_s1.scatter(ts_sinc, xs_sinc, color='green', s=10, label="Sampling Points")

            if fs_sinc < 2 * f_sinc:
                f_mod = f_sinc % fs_sinc
                f_alias_sinc = fs_sinc - f_mod if f_mod > fs_sinc / 2 else f_mod
                x_alias_sinc = sinc_signal(t_sinc, f_alias_sinc)
                sign = np.sign(xs_sinc[1] / sinc_signal(ts_sinc, f_alias_sinc)[1])
                x_alias_sinc *= sign
                ax_s1.plot(t_sinc, x_alias_sinc, '--', lw=1, color='blue', label="Aliased Sinc Signal")
                st.warning("⚠ Aliasing Occurred in Sinc Signal")
            else:
                st.success("✅ Sinc Signal Satisfies Nyquist Theorem")

            ax_s1.set_title("Time Domain Sinc Signal Waveform")
            ax_s1.set_xlabel("Time (s)")
            ax_s1.set_ylabel("Amplitude")
            ax_s1.legend(loc="upper right")
            ax_s1.grid(True)
            st.pyplot(fig_s1)

            # FFT calculate function
            def fft_signal(sig, t):
                N = len(sig)
                dt = t[1] - t[0]
                freq = np.fft.fftfreq(N, dt)
                spectrum = np.abs(np.fft.fft(sig)) / N
                return np.fft.fftshift(freq), np.fft.fftshift(spectrum)

            # Sinc signal spectrum calculation
            freq_x_sinc, X_sinc = fft_signal(x_sinc, t_sinc)
            x_sampled_sinc = np.zeros_like(t_sinc)
            idx_sinc = (ts_sinc / (t_sinc[1] - t_sinc[0])).astype(int)
            idx_sinc = idx_sinc[idx_sinc < len(t_sinc)]
            x_sampled_sinc[idx_sinc] = xs_sinc[:len(idx_sinc)]
            freq_s_sinc, Xs_sinc = fft_signal(x_sampled_sinc, t_sinc)

            # Original sinc spectrum plot
            fig_s2, ax_s2 = plt.subplots(figsize=(10, 5))
            ax_s2.plot(freq_x_sinc, X_sinc, label="Original Sinc Spectrum", color='red')
            ax_s2.set_xlim(-60, 60)
            ax_s2.set_title("Original Sinc Signal Spectrum")
            ax_s2.set_xlabel("Frequency (Hz)")
            ax_s2.set_ylabel("Magnitude")
            ax_s2.legend(loc="upper right")
            ax_s2.grid(True)
            st.pyplot(fig_s2)

            # Sampled sinc spectrum plot
            fig_s3, ax_s3 = plt.subplots(figsize=(10, 5))
            ax_s3.plot(freq_s_sinc, Xs_sinc, label="Sampled Sinc Spectrum", color='blue')
            ax_s3.set_xlim(-60, 60)
            ymax_sinc = np.max(Xs_sinc)
            ax_s3.set_ylim(0, ymax_sinc * 1.2)

            if fs_sinc < 2 * f_sinc:
                f_start = fs_sinc - f_sinc
                f_end = f_sinc
                if f_start < f_end:
                    ax_s3.fill_betweenx([0, ymax_sinc], f_start, f_end, color='blue', alpha=0.2, label="Aliasing Region")
                    ax_s3.fill_betweenx([0, ymax_sinc], -f_end, -f_start, color='blue', alpha=0.2)

            # Vertical dashed line for sinc spectrum
            def vline_sinc(x):
                ax_s3.plot([x, x], [0, ymax_sinc], linestyle='--', linewidth=1, color='black')
            vline_sinc(f_sinc)
            vline_sinc(-f_sinc)
            vline_sinc(fs_sinc)
            vline_sinc(-fs_sinc)

            ax_s3.set_title("Sampled Sinc Signal Spectrum")
            ax_s3.set_xlabel("Frequency (Hz)")
            ax_s3.set_ylabel("Magnitude")
            ax_s3.legend(loc="upper right")
            ax_s3.grid(True)
            st.pyplot(fig_s3)

        # Square Wave Module (New Added)
        else:
            st.subheader("Square Wave Sampling & Aliasing")
            col1, col2, col3 = st.columns(3)
            with col1:
                f = st.slider(r"Signal Frequency $f\ (\rm Hz)$", 1, 50, 15, key="f_square")
            with col2:
                fs = st.slider(r"Sampling Frequency $f_{\rm s}\ (\rm Hz)$", 1, 50, 25, key="fs_square")
            with col3:
                duty = st.slider("Duty Cycle (%)", 5, 95, 50, key="duty_square")
            D = duty / 100

            # Square wave signal definition (Fourier series)
            def square_signal(t, f, duty=0.5, harmonics=40):
                y = np.zeros_like(t)
                for k in range(1, harmonics + 1):
                    ak = (2 / (k * np.pi)) * np.sin(k * np.pi * duty)
                    y += ak * np.cos(2 * np.pi * k * f * t)
                y += duty - 0.5
                return y

            # Time axis
            t = np.linspace(-0.5, 0.5, 3000)
            ts = np.arange(-0.5, 0.5, 1 / fs)
            x = square_signal(t, f, D)
            xs = square_signal(ts, f, D)

            # Time domain waveform plot
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(t, x, label="Original Square Signal", lw=1, color="red")
            ax1.vlines(ts, 0, xs, lw=1, colors='green')
            ax1.scatter(ts, xs, color='green', s=10, label="Sampling Points")

            # Aliasing calculation
            if fs < 2 * f:
                f_mod = f % fs
                f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
                x_alias = square_signal(t, f_alias, D)
                alias_sample = square_signal(ts, f_alias, D)
                if len(alias_sample) > 1 and abs(alias_sample[1]) > 1e-8:
                    sign = np.sign(xs[1] / alias_sample[1])
                    x_alias *= sign
                ax1.plot(t, x_alias, '--', lw=1, color='blue', label="Aliased Signal")
                st.warning("⚠ Frequency Aliasing Occurred!")
            else:
                st.success("✅ Satisfies Nyquist Sampling Theorem!")

            ax1.set_title(f"Time Domain Signal Waveform (Square, Duty {duty}%)")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.legend(loc="upper right")
            ax1.grid(True)
            st.pyplot(fig1)

            # FFT calculate function
            def fft_signal(sig, t):
                N = len(sig)
                dt = t[1] - t[0]
                freq = np.fft.fftfreq(N, dt)
                spectrum = np.abs(np.fft.fft(sig)) / N
                return np.fft.fftshift(freq), np.fft.fftshift(spectrum)

            # Spectrum calculation
            freq_x, X = fft_signal(x, t)
            x_sampled = np.zeros_like(t)
            indices = ((ts - t[0]) / (t[1] - t[0])).astype(int)
            indices = indices[(indices >= 0) & (indices < len(t))]
            x_sampled[indices] = xs[:len(indices)]
            freq_s, Xs = fft_signal(x_sampled, t)

            # Original spectrum figure
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(freq_x, X, label="Original Square Spectrum", color='red')
            ax2.set_xlim(-60, 60)
            ax2.set_title("Original Signal Spectrum (Square)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude")
            ax2.legend(loc="upper right")
            ax2.grid(True)
            st.pyplot(fig2)

            # Sampled signal spectrum figure
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(freq_s, Xs, label="Sampled Square Spectrum", color='blue')
            ax3.set_xlim(-60, 60)
            ymax = np.max(Xs)
            ax3.set_ylim(0, ymax * 1.2)

            # Mark aliasing region
            if fs < 2 * f:
                f_overlap_start = fs - f
                f_overlap_end = f
                if f_overlap_start < f_overlap_end:
                    ax3.fill_betweenx([0, ymax], f_overlap_start, f_overlap_end, color='blue', alpha=0.2, label="Aliasing Region")
                    ax3.fill_betweenx([0, ymax], -f_overlap_end, -f_overlap_start, color='blue', alpha=0.2)

            # Vertical dashed frequency line
            def vline(x):
                ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')
            vline(f)
            vline(-f)
            vline(fs)
            vline(-fs)

            ax3.set_title("Sampled Square Signal Spectrum")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Magnitude")
            ax3.legend(loc="upper right")
            ax3.grid(True)
            st.pyplot(fig3)

    with sub_tab2:
        st.subheader("Anti-Aliasing Filter Demonstration")
        st.markdown("**Principle**: Filter out high-frequency components before sampling to avoid aliasing fundamentally.")

        col1, col2, col3 = st.columns(3)
        with col1:
            f_main = st.slider("Main Signal Frequency (Hz)", 1, 20, 10, key="f_main")
        with col2:
            f_noise = st.slider("High-Frequency Noise Frequency (Hz)", 45, 100, 60, key="f_noise")
        with col3:
            fs = st.slider("Sampling Frequency fs (Hz)", 75, 100, 80, key="fs_filter")

        fc = st.slider("Filter Cut-off Frequency (Hz)", 25, 40, 30, key="fc")

        # Generate mixed signal and filtered signal
        t = np.linspace(0, 1, 1000)
        x_mix = np.sin(2 * np.pi * f_main * t) + 0.3 * np.sin(2 * np.pi * f_noise * t)
        x_filtered = np.sin(2 * np.pi * f_main * t)

        ts = np.arange(0, 1, 1 / fs)
        xs_mix = np.sin(2 * np.pi * f_main * ts) + 0.3 * np.sin(2 * np.pi * f_noise * ts)
        xs_filtered = np.sin(2 * np.pi * f_main * ts)

        # Anti-aliasing filter comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(t, x_mix, color="red", label="Mixed Signal (with High-Frequency Noise)")
        ax1.scatter(ts, xs_mix, color="green", label="Sampling Points")
        ax1.set_title("Without Anti-Aliasing Filter (Prone to Aliasing)")
        ax1.set_ylabel("Amplitude")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2.plot(t, x_filtered, color="blue", label="Filtered Signal (No High-Frequency Noise)")
        ax2.scatter(ts, xs_filtered, color="green", label="Sampling Points")
        ax2.set_title("With Anti-Aliasing Filter (Eliminate Aliasing)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend(loc="upper right")
        ax2.grid(True)

        st.pyplot(fig)

        # Tip message
        if f_noise > fs / 2:
            st.success("✅ Anti-aliasing filter removed high-frequency noise, completely avoiding aliasing!")
        else:
            st.info("ℹ️ Noise frequency is below Nyquist frequency, no obvious aliasing.")

with tab_other:
    st.info("ℹ️ More signals and systems teaching modules can be extended here.")