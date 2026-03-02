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
import time

# Import our modular pipeline components
from data_simulator import generate_synthetic_chunk
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
    st.subheader("🔬 Signal Parameters")

    drift_severity = st.slider(
        "Micromotion Drift Severity",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Simulates physical electrode movement from heartbeat/respiration"
    )

    noise_level = st.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Background Gaussian noise amplitude"
    )

    tanh_alpha = st.slider(
        "Tanh Alpha (Gain)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Soft-clipping compression steepness"
    )

    st.markdown("---")
    st.subheader("🔧 DSP Parameters")

    moving_avg_window = st.slider(
        "Moving Avg Window (samples)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="Window size for baseline drift removal"
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
        'moving_avg_window': moving_avg_window,
        'tanh_alpha': tanh_alpha,
        'spike_threshold': 20.0
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
        # Create dual-trace comparison plot
        fig = go.Figure()

        sample_indices = np.arange(len(st.session_state.raw_signal))

        # Raw signal trace (red)
        fig.add_trace(go.Scatter(
            x=sample_indices,
            y=st.session_state.raw_signal,
            mode='lines',
            name='Raw Drifting Signal',
            line=dict(color='red', width=1),
            opacity=0.7
        ))

        # Cleaned signal trace (cyan)
        fig.add_trace(go.Scatter(
            x=sample_indices,
            y=st.session_state.cleaned_signal,
            mode='lines',
            name='Tanh-Normalized Output',
            line=dict(color='cyan', width=1.5)
        ))

        fig.update_layout(
            title="Raw vs. Cleaned Signal Comparison (50ms Chunk @ 40kHz)",
            xaxis_title="Sample Index (0-2000)",
            yaxis_title="Amplitude (μV / normalized)",
            template="plotly_dark",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

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

    if len(yield_history) > 0:
        stability_index, stability_variance = st.session_state.stability_tracker.calculate_stability_index(stability_window)

        # Create stability trend plot
        fig = go.Figure()

        epoch_indices = np.arange(len(yield_history))

        # Main stability line
        fig.add_trace(go.Scatter(
            x=epoch_indices,
            y=yield_history,
            mode='lines',
            name='Signal Yield %',
            line=dict(color='dodgerblue', width=2)
        ))

        # ±2σ confidence band
        rolling_mean = np.convolve(yield_history, np.ones(min(stability_window, len(yield_history))) / min(stability_window, len(yield_history)), mode='same')
        upper_bound = rolling_mean + 2 * stability_variance
        lower_bound = rolling_mean - 2 * stability_variance

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

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Business Value:** Array viability maintained at {stability_index:.1f}% without manual recalibration over {len(yield_history)} epochs.")


# ============================================================================
# Auto-refresh for Play Mode
# ============================================================================

if st.session_state.is_playing:
    time.sleep(0.05)  # 50ms delay between chunks
    st.rerun()
