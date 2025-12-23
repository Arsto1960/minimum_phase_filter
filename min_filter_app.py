import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Circle

# --- Page Config ---
st.set_page_config(
    page_title="System Control: Phase & Stability",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Theme CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117; 
        color: #e0e0e0;
    }
    .metric-container {
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        background-color: #1a1c24;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace;
        color: #00ff00; /* System Green */
        font-weight: bold;
    }
    h1, h2, h3 {
        font-family: 'Helvetica', sans-serif;
        color: #fafafa;
    }
    .stButton>button {
        border-radius: 5px;
        border: 1px solid #4b4b4b;
        background-color: #262730;
        color: #00ff00;
        font-family: 'Courier New';
    }
    .stButton>button:hover {
        border-color: #00ff00;
        box-shadow: 0 0 10px #00ff0040;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_min_phase(h_linear):
    # Method: Auto-correlation -> Minimum Phase
    # 1. Compute squared magnitude response (h * h_reverse)
    h_sq = signal.convolve(h_linear, h_linear[::-1])
    # 2. Extract minimum phase spectral factor
    h_min = signal.minimum_phase(h_sq)
    return h_min

def plot_dark_theme(fig, ax_list):
    fig.patch.set_facecolor('#0e1117')
    if not isinstance(ax_list, (list, np.ndarray)):
        ax_list = [ax_list]
    
    for ax in np.array(ax_list).flatten():
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#a0a0a0')
        ax.spines['bottom'].set_color('#404040')
        ax.spines['left'].set_color('#404040')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, color='#303030', linestyle=':')
        ax.xaxis.label.set_color('#e0e0e0')
        ax.yaxis.label.set_color('#e0e0e0')
        ax.title.set_color('#ffffff')
        
        legend = ax.get_legend()
        if legend:
            plt.setp(legend.get_texts(), color='#e0e0e0')
            legend.get_frame().set_facecolor('#262730')
            legend.get_frame().set_edgecolor('#404040')

# --- Sidebar ---
with st.sidebar:
    st.title("🎛️ SYSTEM CONTROL")
    st.markdown("Select Module:")
    module = st.radio(
        "",
        # ["MODULE 1: Phase Analyzer", "MODULE 2: Zero Reflector", "MODULE 3: The Equalizer"],
        ["MODULE 1: Phase Analyzer", "MODULE 2: The Equalizer"],
        index=0
    )
    st.divider()
    st.info("System Status: ONLINE")

# ==============================================================================
# MODULE 1: PHASE ANALYZER (Linear vs Minimum)
# ==============================================================================
if module == "MODULE 1: Phase Analyzer":
    st.header("Module 1: Latency Analysis")
    st.markdown("""
    **Objective:** Compare **Linear Phase** (Symmetric) vs **Minimum Phase** (Causal/Front-loaded).
    Observe that while Magnitude remains identical, Group Delay is minimized.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filter Config")
        taps = st.slider("Filter Taps", 5, 51, 21, step=2)
        window = st.selectbox("Window Type", ["hamming", "hann", "blackman"])
        
    with col2:
        # 1. Create Linear Phase Filter (Symmetric Windowed Sinc)
        h_lin = signal.firwin(taps, 0.4, window=window)
        
        # 2. Create Minimum Phase Equivalent
        h_min = get_min_phase(h_lin)
        
        # 3. Analyze
        w, H_lin = signal.freqz(h_lin, worN=1024)
        w, H_min = signal.freqz(h_min, worN=1024)
        
        # Group Delay
        w_gd, gd_lin = signal.group_delay((h_lin, 1), w=1024)
        w_gd, gd_min = signal.group_delay((h_min, 1), w=1024)
        
        # Plots
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        
        # Impulse Response
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.stem(h_lin, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='Linear Phase')
        ax1.stem(h_min, linefmt='C1-', markerfmt='C1x', basefmt=' ', label='Min Phase')
        ax1.set_title("Impulse Response h[n]")
        ax1.legend()
        
        # Magnitude (Should be same)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(w/np.pi, 20*np.log10(np.abs(H_lin)+1e-12), 'C0', label='Linear', linewidth=3, alpha=0.5)
        ax2.plot(w/np.pi, 20*np.log10(np.abs(H_min)+1e-12), 'C1--', label='Minimum')
        ax2.set_title("Magnitude Response (dB)")
        ax2.set_ylim(-60, 5)
        ax2.legend()
        
        # Group Delay
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(w_gd/np.pi, gd_lin, 'C0', label='Linear Phase Delay')
        ax3.plot(w_gd/np.pi, gd_min, 'C1', label='Minimum Phase Delay')
        ax3.fill_between(w_gd/np.pi, gd_min, gd_lin, color='green', alpha=0.1, label='Latency Saved')
        ax3.set_title("Group Delay (Samples)")
        ax3.set_xlabel("Normalized Frequency")
        ax3.legend()
        
        plot_dark_theme(fig, [ax1, ax2, ax3])
        st.pyplot(fig)
        
        # Metrics
        avg_delay_lin = np.mean(gd_lin)
        avg_delay_min = np.mean(gd_min)
        savings = (avg_delay_lin - avg_delay_min) / avg_delay_lin * 100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Linear Delay", f"{avg_delay_lin:.1f} smp")
        m2.metric("Min-Phase Delay", f"{avg_delay_min:.1f} smp")
        m3.metric("Latency Reduction", f"{savings:.1f}%", delta="Faster Response")

# ==============================================================================
# MODULE 2: ZERO REFLECTOR (Interactive Z-Plane)
# ==============================================================================
elif module == "MODULE 2: Zero Reflector":
    st.header("Module 2: Zero Reflection & Stability")
    st.markdown("""
    **Objective:** A system is "Minimum Phase" only if all Zeros are **inside** the unit circle.
    Zeros outside ($|z| > 1$) make the **Inverse System** unstable. 
    **Task:** Move the Zero Slider. If it goes outside (Red), click **Reflect** to bring it inside (Green).
    """)
    
    col_ctrl, col_plot = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Zero Controller")
        # Interactive Zero Location
        r = st.slider("Zero Radius (r)", 0.2, 2.0, 1.5, 0.1)
        theta = st.slider("Zero Angle (rad)", 0.0, 3.14, 0.8, 0.1)
        
        # Calculate Zero
        z0 = r * np.exp(1j * theta)
        
        # Check status
        is_stable_inverse = r < 1.0
        
        st.write("---")
        if not is_stable_inverse:
            st.error("⚠️ SYSTEM WARNING: Zero outside Unit Circle! Inverse Filter will be UNSTABLE.")
            st.metric("Current Radius", f"{r:.2f}")
            if st.button("🔄 REFLECT ZERO INWARD"):
                st.info(f"Reflected to r = {1/r:.2f}")
        else:
            st.success("✅ SYSTEM STABLE: Minimum Phase Condition Met.")
            st.metric("Current Radius", f"{r:.2f}")

    with col_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 1. Z-Plane Plot
        unit_circle = Circle((0,0), 1, color='#00ff00', fill=False, linestyle='--', linewidth=1.5)
        ax1.add_patch(unit_circle)
        
        # Plot Zero
        color = '#00ff00' if is_stable_inverse else '#ff0000'
        ax1.scatter(np.real(z0), np.imag(z0), s=150, c=color, marker='o', label='Zero $z_0$')
        
        # If reflected, show the ghost
        if r > 1.0:
            z_ref = (1/r) * np.exp(1j * theta)
            ax1.scatter(np.real(z_ref), np.imag(z_ref), s=100, c='#00ff00', marker='x', alpha=0.6, label='Reflection $1/z_0^*$')
            ax1.plot([0, np.real(z0)], [0, np.imag(z0)], 'gray', linestyle=':', alpha=0.5)
            
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax1.set_aspect('equal')
        ax1.set_title("Z-Plane (Pole-Zero Map)")
        ax1.legend()
        
        # 2. Inverse Impulse Response
        # Filter H(z) = 1 - z0 * z^-1
        # Inverse G(z) = 1 / (1 - z0 * z^-1) -> Geometric series z0^n
        n = np.arange(20)
        h_inv = np.real(z0**n) # Real part approximation for visualization
        
        # Check for explosion
        if np.max(np.abs(h_inv)) > 10:
            h_inv = np.clip(h_inv, -10, 10) # Clip for plotting
            ax2.text(10, 0, "UNSTABLE\n(Explodes)", color='red', ha='center', fontweight='bold')
            
        ax2.stem(n, h_inv, linefmt=color, basefmt=" ")
        ax2.set_title("Inverse Impulse Response $h_{inv}[n]$")
        ax2.set_ylim(-5, 5)
        
        plot_dark_theme(fig, [ax1, ax2])
        st.pyplot(fig)

# ==============================================================================
# MODULE 3: THE EQUALIZER (Application)
# ==============================================================================
elif module == "MODULE 2: The Equalizer":
    st.header("Module 2: Channel Equalization")
    st.markdown("""
    **Scenario:** A communication channel (e.g., a room or cable) distorts the signal.
    **Objective:** We need to invert the channel ($H_{eq} = 1/H_{channel}$) to recover the signal.
    **Constraint:** Direct inversion works ONLY if the channel is Minimum Phase. Otherwise, we must convert it first!
    """)
    
    col_eq1, col_eq2 = st.columns([1, 2])
    
    with col_eq1:
        st.subheader("Distortion Source")
        distortion_type = st.radio("Channel Type", ["Minimum Phase (Easy)", "Non-Minimum Phase (Hard)"])
        
        # Create Channel Filter
        if distortion_type == "Minimum Phase (Easy)":
            # Just a simple decay
            h_chan = [1.0, 0.5, 0.25] 
        else:
            # Maximum phase / Mixed phase (Zeros outside)
            # [0.25, 0.5, 1.0] puts most energy at end -> Non-min phase
            h_chan = [0.25, 0.5, 1.0] 
            
        st.code(f"h_channel = {h_chan}")
        
    with col_eq2:
        # Simulation
        # 1. Signal
        sig = np.zeros(50)
        sig[10] = 1.0 # Pulse
        sig[15] = 0.5
        sig[25] = -0.8
        
        # 2. Apply Distortion
        distorted = signal.lfilter(h_chan, 1, sig)
        
        # 3. Attempt Equalization (Inverse Filtering)
        # Inverse of H(z) is 1/H(z). In scipy lfilter(b, a, x), we swap: lfilter(1, h_chan, x)
        try:
            # We filter with a=[h_chan], b=[1]
            restored = signal.lfilter([1], h_chan, distorted)
        except:
            restored = np.zeros_like(distorted)
            
        # Check stability of restoration
        is_unstable = np.max(np.abs(restored)) > 10.0
        
        # --- Visualization ---
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
        
        # Plot 1: Original
        ax[0].stem(sig, linefmt='green', basefmt=" ", label="Original Data")
        ax[0].set_title("1. Original Transmission")
        ax[0].set_ylim(-1.5, 1.5)
        
        # Plot 2: Distorted
        ax[1].stem(distorted, linefmt='orange', basefmt=" ", label="Distorted Signal")
        ax[1].set_title("2. Received (Distorted)")
        ax[1].set_ylim(-1.5, 1.5)
        
        # Plot 3: Restored
        color = 'red' if is_unstable else '#00ccff'
        label = "Restoration FAILED (Unstable)" if is_unstable else "Restoration SUCCESS"
        
        # Clip for display if unstable
        display_restored = np.clip(restored, -2, 2)
        
        ax[2].stem(display_restored, linefmt=color, basefmt=" ", label=label)
        ax[2].set_title("3. Equalizer Output (1 / H_channel)")
        ax[2].legend()
        ax[2].set_ylim(-2, 2)
        
        plot_dark_theme(fig, ax)
        st.pyplot(fig)
        
        if is_unstable:
            st.error("""
            **SYSTEM FAILURE:** The channel was **Non-Minimum Phase**. 
            Its zeros were outside the unit circle. When inverted, they became **Unstable Poles**.
            **Solution:** We must reflect zeros to create a Minimum Phase approximation before inverting.
            """)
        else:
            st.success("""
            **SUCCESS:** The channel was **Minimum Phase**. 
            All zeros were inside. The inverse filter (Equalizer) is stable.
            """)
