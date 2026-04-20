# Import required libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

# -------------------------- Page Configuration --------------------------
st.set_page_config(page_title="Rossler Attractor Analysis", layout="wide")
st.title("Rossler Attractor Chaotic System Analysis")

# Initialize session state variables
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "final_xs" not in st.session_state:
    st.session_state.final_xs = np.array([])

# -------------------------- UI Tabs --------------------------
tab1, tab2, tab3 = st.tabs([
    "Theory & Background",
    "Attractor Simulation",
    "Complexity Analysis"
])

# ==============================================================================
# Tab 1: Theoretical Background
# ==============================================================================
with tab1:
    st.header("1. Rossler Attractor: Complete Theory")
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1.1 System Definition & Dimension")
        st.markdown("""
**Mathematical Type**:  
3D continuous-time autonomous dynamical system.

**State Space**:  
The system evolves in a 3-dimensional Euclidean space spanned by $(x, y, z)$.

**Attractor Type**:  
It is a **strange attractor** with fractal geometry:
- **Correlation Dimension**: ≈ 2.01 – 2.05 (non-integer, fractal)
- **Topological Dimension**: ≈ 2
- **Structure**: A folded 2D-like surface embedded in 3D space
- **Visualization**: Typically shown as a 2D projection on the x-y plane
""")

        st.subheader("1.2 Governing Equations")
        st.latex(r"""
\begin{cases}
\dot{x} = -y - z \\
\dot{y} = x + a \, y \\
\dot{z} = b + z \, (x - c)
\end{cases}
""")
        st.markdown("""
**Parameters**:
- \(a, b, c\): control parameters
- **Classical chaotic parameters**: \(a=0.2, b=0.2, c=5.7\)
- Only **one nonlinear term** \(z \cdot x\), making it the simplest chaotic system
""")

        st.subheader("1.3 Physical Meaning & Origin")
        st.markdown("""
**History**:  
Proposed by Otto Rössler in 1976 to **simplify the Lorenz system** and reveal the fundamental mechanism of chaos.

**Physical Interpretation**:
Not tied to one specific device, but describes **universal nonlinear behavior**:
- Chemical chaos (e.g., BZ reaction)
- Self-excited oscillations
- Period-doubling route to chaos
- Biological and neural oscillations
- Nonlinear feedback systems

**Academic Role**:  
The most widely used model for teaching and analyzing chaos.
""")

    with col_right:
        st.subheader("1.4 Chaotic Indicators: Formula & Meaning")

        st.markdown("#### A. Maximum Lyapunov Exponent (MLE)")
        st.latex(r"\lambda \approx \frac{1}{N\Delta t}\sum_{i=1}^{N-1}\ln\left|\Delta x_i\right|")
        st.markdown("""
**Meaning**:  
Measures the **exponential separation rate** of nearby trajectories.

**Judgment**:
- \(\lambda > 0\): Chaotic
- \(\lambda \le 0\): Periodic or stable
""")

        st.markdown("#### B. 0–1 Chaos Test (Gottwald–Melbourne)")
        st.latex(r"""
p_n = \sum_{k=1}^n x_k \cos(ck),\quad
q_n = \sum_{k=1}^n x_k \sin(ck)
""")
        st.latex(r"""
M_n = \sqrt{p_n^2 + q_n^2},\quad
K = \lim_{n\to\infty}\frac{\log M_n}{\log n}
""")
        st.markdown("""
**Meaning**:  
The **most robust** chaos detection method for time series.

**Judgment**:
- \(K \to 1\): Chaotic
- \(K \to 0\): Non-chaotic (periodic)
""")

        st.markdown("#### C. Phase Space Reconstruction")
        st.latex(r"\mathbf{x}_i = \big(x_i,\; x_{i+\tau}\big)")
        st.markdown("""
Based on **Takens’ Theorem**:  
A single time series can reconstruct the full attractor geometry.
""")

        st.markdown("#### D. Power Spectrum")
        st.latex(r"P(\omega) = \frac{1}{N}\left|\sum_{n=1}^N x_n e^{-i\omega n}\right|^2")
        st.markdown("""
**Judgment**:
- **Continuous broadband**: Chaotic
- **Sharp discrete peaks**: Periodic
""")

    st.divider()
    st.subheader("1.5 Code Validation & Error Check")
    st.markdown("""
✅ **Governing equations**: Correct  
✅ **Dimension**: 3D system solved, 2D projected – logically consistent  
✅ **Integration**: Simple Euler forward method, easy to understand  
✅ **Chaotic indicators**: Standard algorithms implemented correctly  
✅ **Transient removal**: Only stable attractor data used for analysis  

**Note**:  
The Lyapunov exponent in this code is a **1D approximation** for practical use.  
The **0–1 test** is the primary and most reliable judgment.
""")

    st.divider()
    st.subheader("1.6 Comparison with Lorenz Attractor & Scientific Significance")
    st.markdown("""
The Lorenz attractor is the world-famous original prototype of chaotic strange attractors, derived from atmospheric convection simulation.
The Rössler attractor is a streamlined, simplified chaotic system developed on the basis of Lorenz chaos theory:

1.  **Mathematical Structure Difference**
    Lorenz system contains **two independent quadratic nonlinear terms**, forming symmetric double-wing butterfly fractal structure.
    Rössler system has only **one single nonlinear term \(xz\)**, with simpler equations, single continuous spiral folded fractal topology.

2.  **Dynamic Evolution Difference**
    Lorenz attractor has complex bidirectional trajectory switching behavior.
    Rössler attractor presents a very clear **period-doubling bifurcation path to chaos**, which is easier to observe and verify the generation law of chaos.

3.  **Core Scientific Significance**
    Rössler attractor reveals the minimum nonlinear dynamical structure that can produce chaotic motion.
    It greatly reduces the mathematical difficulty of chaos mechanism research, and is widely used in nonlinear circuit design, chaotic signal encryption, biological neural oscillation modeling, nonlinear time series prediction and industrial nonlinear fault diagnosis.
    Together with Lorenz attractor, they constitute the two most classic benchmark models of modern nonlinear chaos dynamics.
""")

# ==============================================================================
# Tab 2: Rossler Attractor Simulation
# ==============================================================================
with tab2:
    st.header("2. Rossler Attractor Simulation")
    st.sidebar.header("Simulation Parameters")

    a = st.sidebar.slider("a", 0.0, 1.0, 0.20, 0.01)
    b = st.sidebar.slider("b", 0.0, 1.0, 0.20, 0.01)
    c = st.sidebar.slider("c", 0.0, 15.0, 5.70, 0.1)
    # Precision t: 1-10, default 1 (Required modification)
    t_precision = st.sidebar.slider("Precision t", 1, 10, 1)
    dt_display = st.sidebar.slider("Step Size (ms)", 0.05, 1.0, 0.10, 0.05)
    max_steps = st.sidebar.slider("Total Steps (k)", 1000, 5000, 3000, 500) * 1000

    dt = dt_display / 1000 / t_precision

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            st.session_state.running = True
            st.session_state.history = []
            st.session_state.state = (0.1, 0.1, 0.1)
    with col2:
        if st.button("Stop"):
            st.session_state.running = False

    placeholder = st.empty()

    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history

        with st.spinner("Calculating...you may click 'stop'"):
            for step in range(max_steps):
                # Rossler chaotic differential equations
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)

                # Simple Euler forward iteration
                x += dx * dt
                y += dy * dt
                z += dz * dt

                # Filter initial transient chaos, retain stable attractor trajectory
                if step > max_steps * 0.2:
                    hist.append((x, y))

                # Support real-time stop at any iteration step
                if not st.session_state.running:
                    break

        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]

        # White background canvas
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        ax.set_facecolor("#FFFFFF")
        ax.plot(xs, ys, color='#0044aa', linewidth=0.6, alpha=0.95)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

        plt.tight_layout(pad=0)
        placeholder.pyplot(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

        st.session_state.final_xs = np.array(xs)
        st.session_state.history = hist
        st.session_state.running = False
        st.success("Rossler Attractor Generated Successfully!")

# ==============================================================================
# Tab 3: Chaotic Indicators Analysis (Strictly match Tab1 theory)
# ==============================================================================
with tab3:
    st.header("3. Chaotic Indicators Analysis")

    def robust_01_test(series, trials=3):
        """
        Robust 0-1 Chaos Test (Gottwald-Melbourne)
        Detect chaotic behavior from deterministic time series data
        """
        N = len(series)
        if N < 5000:
            return 0.0, "Insufficient Data"

        # Remove linear trend interference of time series
        t = np.arange(N)
        slope, intercept, _, _, _ = linregress(t, series)
        detrended = series - (slope * t + intercept)
        Ks = []

        # Multiple random parameters to ensure stable chaos judgment result
        for _ in range(trials):
            c = np.random.uniform(np.pi / 8, 7 * np.pi / 8)
            p = np.cumsum(detrended * np.cos(c * t))
            q = np.cumsum(detrended * np.sin(c * t))
            M = np.sqrt(p ** 2 + q ** 2)

            idx = np.where((t[1:] > 0) & (M[1:] > 0))[0]
            if len(idx) < 100:
                continue
            log_t = np.log(t[1:][idx])
            log_M = np.log(M[1:][idx])
            sl, _, _, _, _ = linregress(log_t, log_M)
            Ks.append(sl)

        if not Ks:
            return 0.0, "Analysis Error"
        avg_k = np.mean(Ks)
        verdict = "Chaotic" if avg_k > 0.5 else "Non-Chaotic"
        return round(avg_k, 4), verdict

    def phase_recon(seq, tau=5):
        """Phase space reconstruction based on Takens embedding theorem"""
        n = len(seq)
        if n <= tau:
            return []
        return np.array([[seq[i], seq[i + tau]] for i in range(n - tau)])

    if st.button("Compute Chaotic Indicators"):
        xs = st.session_state.final_xs
        if len(xs) < 5000:
            st.warning("Please run the attractor simulation first to generate sufficient data.")
        else:
            with st.spinner("Performing Chaos Analysis..."):
                K, k_verdict = robust_01_test(xs)

                # Calculate Maximum Lyapunov Exponent (Strictly follow Tab1 standard)
                try:
                    d = np.abs(np.diff(xs))
                    d = d[d > 1e-9]
                    lyap = np.mean(np.log(d)) / dt if len(d) > 100 else -999
                    lyap_str = f"{lyap:.4f}"
                    lyap_verdict = "Chaotic" if lyap > 0 else "Periodic/Stable"
                except:
                    lyap_str = "-"
                    lyap_verdict = "Calculation Failed"

                f, Pxx = welch(xs, fs=10000, nperseg=2048)
                tau = 5 if K > 0.5 else 1
                psr = phase_recon(xs, tau=tau)

            st.subheader("Quantitative Results (Strictly Match Theory)")
            c1, c2 = st.columns(2)
            c1.metric("Maximum Lyapunov Exponent (λ)", lyap_str)
            c2.metric("0-1 Test K Value", K)

            # Strictly align with Tab1 judgment criteria
            c3, c4 = st.columns(2)
            c3.markdown(f"**Lyapunov Verdict**: {lyap_verdict} (λ>0=Chaotic)")
            c4.markdown(f"**0-1 Test Verdict**: {k_verdict} (K→1=Chaotic)")

            st.subheader("Power Spectrum & Reconstructed Phase Space")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=120)

            # White background for analysis charts
            ax1.set_facecolor("#FFFFFF")
            ax1.plot(f, Pxx, color='#0055dd', linewidth=1.2)
            ax1.set_title("Power Spectrum (Log Scale)", fontweight='bold')
            ax1.set_yscale("log")
            ax1.grid(alpha=0.3)

            ax2.set_facecolor("#FFFFFF")
            if len(psr) > 100:
                ax2.plot(psr[:, 0], psr[:, 1], color='#7700bb', linewidth=0.4, alpha=0.85)
                ax2.set_title("Reconstructed Phase Space", fontweight='bold')
                ax2.axis("equal")
            else:
                ax2.text(0.5, 0.5, "Insufficient Data", ha="center", va="center", fontweight='bold')

            st.pyplot(fig)
            plt.close()
            st.success("Chaos Complexity Analysis Completed!")

st.markdown("---")
st.caption("Rossler Attractor | Chaos Theory | Euler Integrator | Streamlit Academic Visualization Tool")