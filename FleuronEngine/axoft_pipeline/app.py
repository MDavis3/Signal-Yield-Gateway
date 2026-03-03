"""
Axoft Signal Yield & Clinical Translation Gateway - Streamlit Dashboard
========================================================================

Production-grade BCI signal processing visualization with dual persona views:
- R&D Engineer View: Raw vs cleaned waveforms, latency metrics
- Clinical/FDA View: KPI cards, chronic stability index, system health

Playback Controls: Play/Pause/Step for frame-by-frame presentation

Author: Senior AI Software Engineer & BCI Data Architect
Date: 2026-03-02
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import our modular pipeline components
from data_simulator import generate_synthetic_chunk, reset_drift_phase
from dsp_pipeline import process_signal_streaming, reset_streaming_buffer
from metrics_engine import (
    calculate_signal_yield,
    calculate_active_channels,
    check_system_health,
    format_health_status,
    calculate_uptime,
    StabilityTracker
)
from storage_manager import create_storage, DEFAULT_STORAGE_BACKEND


# ============================================================================
# Streamlit Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Axoft Signal Yield Gateway",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Session State Initialization
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'storage' not in st.session_state:
        st.session_state.storage = create_storage(DEFAULT_STORAGE_BACKEND)

    if 'stability_tracker' not in st.session_state:
        st.session_state.stability_tracker = StabilityTracker(max_history=200)

    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False

    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = time.time()

    if 'raw_signal' not in st.session_state:
        st.session_state.raw_signal = None

    if 'cleaned_signal' not in st.session_state:
        st.session_state.cleaned_signal = None

    if 'latest_metrics' not in st.session_state:
        st.session_state.latest_metrics = {}


initialize_session_state()


# ============================================================================
# Sidebar - Control Panel
# ============================================================================

with st.sidebar:
    st.title("⚙️ Axoft Pipeline Controller")

    st.markdown("---")
    st.subheader("📋 Presets")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎯 Demo"):
            st.session_state.preset_noise = 0.10
            st.session_state.preset_drift = 0.20
            st.session_state.preset_alpha = 0.70
            st.rerun()

    with col2:
        if st.button("⚙️ Real"):
            st.session_state.preset_noise = 0.30
            st.session_state.preset_drift = 0.40
            st.session_state.preset_alpha = 1.00
            st.rerun()

    with col3:
        if st.button("🔥 Stress"):
            st.session_state.preset_noise = 0.50
            st.session_state.preset_drift = 1.00
            st.session_state.preset_alpha = 1.50
            st.rerun()

    st.caption("Presets set noise, drift, and alpha for optimal visualization")

    st.markdown("---")
    st.subheader("🔬 Signal Parameters")

    # Use preset values if available, otherwise use slider defaults
    default_drift = st.session_state.get('preset_drift', 0.40)
    default_noise = st.session_state.get('preset_noise', 0.10)

    drift_severity = st.slider(
        "Micromotion Drift Severity",
        min_value=0.0,
        max_value=2.0,
        value=default_drift,
        step=0.05,
        help="Simulates physical electrode movement from heartbeat/respiration (0.40 = realistic biological conditions)"
    )

    noise_level = st.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=default_noise,
        step=0.05,
        help="Background Gaussian noise amplitude (0.10 = good recording conditions, 0.30 = realistic biological, 0.50 = noisy/worst-case)"
    )

    # Use preset value if available, otherwise default to 1.0
    default_alpha = st.session_state.get('preset_alpha', 1.0)

    tanh_alpha = st.slider(
        "Tanh Alpha (Gain)",
        min_value=0.1,
        max_value=5.0,
        value=default_alpha,
        step=0.1,
        help="Compression steepness for visualization. Higher alpha = better spike separation in noisy signals. Recommended: 0.7 (Demo), 1.0 (Real), 1.5+ (Stress)"
    )

    st.markdown("---")
    st.subheader("🔧 DSP Parameters")

    # Polynomial detrending (replaces moving average)
    # Order hardcoded to 1 (linear) - optimal for 50ms chunks with 1Hz drift
    poly_order = 1

    # Smoothing no longer needed (polyfit eliminates ringing)
    # Keeping slider for optional noise reduction if desired
    smoothing_window = st.slider(
        "Smoothing Window (samples)",
        min_value=0,
        max_value=100,
        value=0,  # DEFAULT TO 0 (no smoothing needed with polyfit)
        step=5,
        help="Optional noise reduction. Polyfit detrending eliminates ringing, so smoothing is rarely needed. Set to 0 to preserve spike morphology (recommended)."
    )

    stability_window = st.slider(
        "Stability Window (epochs)",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Rolling window for chronic stability index"
    )

    st.markdown("---")
    st.subheader("👁️ View Persona")

    view_mode = st.radio(
        "Select View",
        options=["R&D Engineer View", "Clinical / FDA View"],
        index=1,  # Default to Clinical view
        help="Toggle between technical and clinical perspectives"
    )

    st.markdown("---")
    st.subheader("▶️ Playback Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Play" if not st.session_state.is_playing else "⏸ Pause"):
            st.session_state.is_playing = not st.session_state.is_playing

    with col2:
        if st.button("⏭ Step"):
            st.session_state.is_playing = False
            # Step will trigger one chunk generation below

    with col3:
        if st.button("🔄 Reset"):
            st.session_state.storage.reset()
            st.session_state.stability_tracker.reset()
            reset_streaming_buffer()
            reset_drift_phase()  # Reset continuous drift simulation
            st.session_state.session_start_time = time.time()

    st.markdown("---")
    st.subheader("📊 Session Stats")
    st.metric("Epochs Processed", st.session_state.storage.get_epoch_count())
    st.metric("Uptime", calculate_uptime(st.session_state.session_start_time, time.time()))


# ============================================================================
# Data Processing - Generate and Process Chunk
# ============================================================================

def process_chunk():
    """Generate synthetic chunk and run through DSP pipeline."""

    # Generate synthetic data from mock hardware
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=noise_level,
        drift_severity=drift_severity,
        spike_rate=20.0
    )

    # Process through DSP pipeline
    config = {
        'poly_order': poly_order,  # Polynomial order for detrending (1 = linear)
        'tanh_alpha': tanh_alpha,
        'spike_threshold': 5.0,  # Lowered from 20.0 to detect 100μV spikes (derivative ≈8.3μV)
        'smoothing_window': smoothing_window  # Optional (default 0, not needed with polyfit)
    }

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)

    # Calculate metrics
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    active_channels = calculate_active_channels(yield_pct, total_channels=10000)
    health_status = check_system_health(cleaned_tensor, metadata, yield_pct)

    # Store results
    st.session_state.storage.save_tensor(cleaned_tensor, yield_pct, metadata)
    st.session_state.stability_tracker.add_yield(yield_pct)

    # Update session state for UI rendering
    st.session_state.raw_signal = raw_chunk
    st.session_state.cleaned_signal = cleaned_tensor
    st.session_state.latest_metrics = {
        'latency_ms': latency_ms,
        'yield_pct': yield_pct,
        'active_channels': active_channels,
        'health_status': health_status,
        'spike_count': metadata['spike_count']
    }


# Trigger chunk processing (Step button or Play mode)
if st.session_state.is_playing or st.button("Generate", key="hidden_generate", disabled=True):
    process_chunk()


# ============================================================================
# Main UI - View-Specific Rendering
# ============================================================================

st.title("🧠 Axoft Signal Yield & Clinical Translation Gateway")

# ============================================================================
# Synthetic Data Warning Banner
# ============================================================================
st.warning(
    "⚠️ **DEMO MODE**: This dashboard uses **synthetic neural data** generated from "
    "mathematical models (Poisson spike trains, sinusoidal drift, Gaussian noise). "
    "Real patient data validation is required before clinical use.",
    icon="⚠️"
)

# ============================================================================
# Parameter Validation Warning
# ============================================================================
# Warn if alpha is too low for the current noise level (poor visual separation)
if noise_level > 0.25 and tanh_alpha < 0.80:
    recommended_alpha = max(1.0, noise_level * 2.5)
    st.warning(
        f"⚠️ **Visual Clarity Warning**: Low alpha ({tanh_alpha:.1f}) with high noise ({noise_level:.2f}) "
        f"may reduce spike visibility. **Recommendation**: Increase alpha to {recommended_alpha:.1f}+ "
        f"or click a preset button for optimal settings.",
        icon="⚠️"
    )

if view_mode == "R&D Engineer View":
    # ========================================================================
    # R&D Engineer View - Technical Waveforms and Latency
    # ========================================================================

    st.markdown("### 🔬 R&D Engineer View")

    if st.session_state.latest_metrics:
        st.metric(
            "Pipeline Latency",
            f"{st.session_state.latest_metrics['latency_ms']:.2f} ms",
            delta=f"Budget: <20ms" if st.session_state.latest_metrics['latency_ms'] < 20 else "OVER BUDGET",
            delta_color="normal" if st.session_state.latest_metrics['latency_ms'] < 20 else "inverse"
        )

    if st.session_state.raw_signal is not None and st.session_state.cleaned_signal is not None:
        # Create dual Y-axis plot to properly visualize both scales
        # Raw signal: -40 to 120 μV (primary Y-axis, left)
        # Tanh normalized: -1 to 1 (secondary Y-axis, right)
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        sample_indices = np.arange(len(st.session_state.raw_signal))

        # Raw signal trace (red) - PRIMARY Y-AXIS (left)
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=st.session_state.raw_signal,
                mode='lines',
                name='Raw Drifting Signal (μV)',
                line=dict(color='red', width=1),
                opacity=0.7
            ),
            secondary_y=False  # Primary Y-axis (left)
        )

        # Cleaned signal trace (cyan) - SECONDARY Y-AXIS (right)
        fig.add_trace(
            go.Scatter(
                x=sample_indices,
                y=st.session_state.cleaned_signal,
                mode='lines',
                name='Tanh-Normalized Output',
                line=dict(color='cyan', width=2)
            ),
            secondary_y=True  # Secondary Y-axis (right)
        )

        # Update layout and axes
        fig.update_layout(
            title="Raw vs. Cleaned Signal Comparison (50ms Chunk @ 40kHz)",
            xaxis_title="Sample Index (0-2000)",
            template="plotly_dark",
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99)
        )

        # Set Y-axis titles and ranges
        fig.update_yaxes(
            title_text="Raw Amplitude (μV)",
            secondary_y=False  # Primary Y-axis (left)
        )

        fig.update_yaxes(
            title_text="Tanh Normalized [-1, 1]",
            range=[-1.2, 1.2],  # STRICTLY LOCK to [-1.2, 1.2]
            secondary_y=True  # Secondary Y-axis (right)
        )

        st.plotly_chart(fig, width='stretch')

else:
    # ========================================================================
    # Clinical / FDA View - KPI Cards and Stability Trend
    # ========================================================================

    st.markdown("### 🏥 Clinical / FDA View")

    # KPI Metric Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.latest_metrics:
            st.metric("🎯 Live Signal Yield", f"{st.session_state.latest_metrics['yield_pct']:.1f}%")
        else:
            st.metric("🎯 Live Signal Yield", "—")

    with col2:
        if st.session_state.latest_metrics:
            st.metric("📡 Active Channels", f"{st.session_state.latest_metrics['active_channels']:,} / 10,000")
        else:
            st.metric("📡 Active Channels", "— / 10,000")

    with col3:
        st.metric("⏱️ System Uptime", calculate_uptime(st.session_state.session_start_time, time.time()))

    # System Health Indicator
    if st.session_state.latest_metrics:
        health_status = format_health_status(st.session_state.latest_metrics['health_status'])
        st.markdown(f"**System Health:** {health_status}")

    # Chronic Stability Index Chart
    yield_history = st.session_state.stability_tracker.get_full_history()
    smoothed_yield_history = st.session_state.stability_tracker.get_smoothed_history()

    if len(yield_history) > 0:
        stability_index, stability_variance = st.session_state.stability_tracker.calculate_stability_index(stability_window)

        # Create stability trend plot
        fig = go.Figure()

        epoch_indices = np.arange(len(yield_history))

        # Main stability line (SMOOTHED via EMA for FDA presentation)
        fig.add_trace(go.Scatter(
            x=epoch_indices,
            y=smoothed_yield_history,
            mode='lines',
            name='Signal Yield % (EMA Smoothed)',
            line=dict(color='dodgerblue', width=2)
        ))

        # Optional: Add raw yield as faint background trace
        fig.add_trace(go.Scatter(
            x=epoch_indices,
            y=yield_history,
            mode='lines',
            name='Raw Yield % (Unsmoothed)',
            line=dict(color='gray', width=1),
            opacity=0.3,
            showlegend=True
        ))

        # ±2σ confidence band - calculated from SMOOTHED yields for statistical consistency
        # Since we're plotting the EMA smoothed line, the envelope should reflect its variance
        smoothed_array = np.array(smoothed_yield_history)
        rolling_mean = np.convolve(smoothed_array, np.ones(min(stability_window, len(smoothed_array))) / min(stability_window, len(smoothed_array)), mode='same')
        smoothed_std = np.std(smoothed_array[-min(stability_window, len(smoothed_array)):])
        upper_bound = rolling_mean + 2 * smoothed_std
        lower_bound = rolling_mean - 2 * smoothed_std

        fig.add_trace(go.Scatter(
            x=np.concatenate([epoch_indices, epoch_indices[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±2σ Envelope',
            showlegend=True
        ))

        fig.update_layout(
            title=f"Chronic Stability Index (Rolling {stability_window} Epoch Window)",
            xaxis_title="Epoch Number",
            yaxis_title="Signal Yield %",
            template="plotly_dark",
            height=400,
            yaxis=dict(range=[0, 105])
        )

        st.plotly_chart(fig, width='stretch')

        st.markdown(f"**Business Value:** Array viability maintained at {stability_index:.1f}% without manual recalibration over {len(yield_history)} epochs.")


# ============================================================================
# Auto-refresh for Play Mode
# ============================================================================

if st.session_state.is_playing:
    time.sleep(0.05)  # 50ms delay between chunks
    st.rerun()
