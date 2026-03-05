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
from real_data_loader import RealDataLoader, generate_real_chunk
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

    # For same-scale visualization
    if 'centered_signal_uv' not in st.session_state:
        st.session_state.centered_signal_uv = None

    if 'raw_dc_offset' not in st.session_state:
        st.session_state.raw_dc_offset = 0.0

    # Real data loader
    if 'real_data_loader' not in st.session_state:
        st.session_state.real_data_loader = RealDataLoader("../data/physionet")

    if 'use_real_data' not in st.session_state:
        st.session_state.use_real_data = False

    if 'real_data_file' not in st.session_state:
        st.session_state.real_data_file = None

    if 'real_data_channel' not in st.session_state:
        st.session_state.real_data_channel = 0


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
    st.subheader("📊 Data Source")

    # Toggle between synthetic and real data
    data_source = st.radio(
        "Select Data Source",
        options=["Synthetic (Simulated)", "Real (PhysioNet EEG)"],
        index=0 if not st.session_state.use_real_data else 1,
        help="Use synthetic data for demos or real PhysioNet EEG recordings"
    )
    st.session_state.use_real_data = (data_source == "Real (PhysioNet EEG)")

    # Real data file selection
    if st.session_state.use_real_data:
        available_files = st.session_state.real_data_loader.get_available_files()

        if available_files:
            selected_file = st.selectbox(
                "EDF File",
                options=available_files,
                help="Select recording file (R01=baseline, R03/R07/R11=left/right motor imagery)"
            )

            # Load file if changed
            if selected_file != st.session_state.real_data_file:
                st.session_state.real_data_file = selected_file
                st.session_state.real_data_loader.load_file(selected_file, channel=0)

            # Channel selection (PhysioNet has 64 channels)
            channel_names = st.session_state.real_data_loader.get_channel_names()
            if channel_names:
                selected_channel = st.selectbox(
                    "EEG Channel",
                    options=list(range(len(channel_names))),
                    format_func=lambda x: channel_names[x] if x < len(channel_names) else f"Ch {x}",
                    help="Select EEG electrode (C3/C4 recommended for motor cortex)"
                )
                if selected_channel != st.session_state.real_data_channel:
                    st.session_state.real_data_channel = selected_channel
                    st.session_state.real_data_loader.load_file(
                        st.session_state.real_data_file,
                        channel=selected_channel
                    )

            # Show data info
            info = st.session_state.real_data_loader.get_info()
            if info.get('loaded'):
                st.caption(f"📍 {info['current_channel']} | {info['duration_seconds']:.1f}s @ {info['sampling_rate']}Hz")
        else:
            st.warning("No EDF files found in data/physionet/")

    st.markdown("---")
    st.subheader("🔧 Processing Mode")

    # Processing mode selection (EEG vs Intracortical)
    if st.session_state.use_real_data:
        processing_mode = st.radio(
            "Algorithm",
            options=["EEG (Highpass Filter)", "Intracortical (Polynomial)"],
            index=0,  # Default to EEG for real data
            help="**EEG mode**: Preserves brain rhythms (8-12Hz alpha), removes only DC drift. "
                 "**Intracortical mode**: Removes all low-frequency content (for spike trains)."
        )
        mode_key = "eeg" if "EEG" in processing_mode else "intracortical"

        # Highpass cutoff slider (only for EEG mode)
        if mode_key == "eeg":
            highpass_cutoff = st.slider(
                "Highpass Cutoff (Hz)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Frequencies below this are removed as drift. 0.5Hz preserves all brain rhythms (alpha, theta, beta)."
            )
        else:
            highpass_cutoff = 0.5  # Default, not used
    else:
        # Synthetic data always uses intracortical mode
        mode_key = "intracortical"
        highpass_cutoff = 0.5
        st.caption("Using polynomial detrending for synthetic data")

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

    # Chunk duration for visualization (longer = more visible cycles)
    if st.session_state.use_real_data:
        chunk_duration_ms = st.slider(
            "Chunk Duration (ms)",
            min_value=50,
            max_value=1000,
            value=500,  # Default 500ms for EEG (shows ~5 alpha cycles)
            step=50,
            help="Longer chunks show more alpha cycles. 500ms recommended for clear visualization of drift removal."
        )
    else:
        chunk_duration_ms = 50.0  # Keep 50ms for synthetic data (40kHz)

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
    """Generate chunk (synthetic or real) and run through DSP pipeline with dual-mode support."""

    # Choose data source and build appropriate config
    if st.session_state.use_real_data and st.session_state.real_data_loader.get_info().get('loaded'):
        # Use REAL neural data from PhysioNet
        raw_chunk, chunk_meta = generate_real_chunk(
            st.session_state.real_data_loader,
            chunk_duration_ms=chunk_duration_ms,  # Use slider value
            add_synthetic_drift=True,  # Add drift to demonstrate removal
            drift_amplitude=drift_severity * 50,  # Scale drift to severity slider
            native_rate=True  # Use native 160Hz sample rate
        )

        # Build config for real EEG data
        config = {
            'processing_mode': mode_key,  # 'eeg' or 'intracortical'
            'sample_rate': chunk_meta.get('sample_rate', 160.0),
            'highpass_cutoff': highpass_cutoff,
            'poly_order': poly_order,
            'tanh_alpha': tanh_alpha,
            'spike_threshold': 5.0,
            'smoothing_window': smoothing_window,
            'signal_type': chunk_meta.get('signal_type', 'eeg')
        }
    else:
        # Generate synthetic data from mock hardware
        raw_chunk = generate_synthetic_chunk(
            duration_ms=chunk_duration_ms,  # Use variable (50ms for synthetic)
            sample_rate=40000,
            noise_level=noise_level,
            drift_severity=drift_severity,
            spike_rate=20.0
        )

        # Build config for synthetic data (always intracortical mode)
        config = {
            'processing_mode': 'intracortical',
            'poly_order': poly_order,
            'tanh_alpha': tanh_alpha,
            'spike_threshold': 5.0,
            'smoothing_window': smoothing_window,
            'signal_type': 'synthetic'
        }

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)

    # Calculate metrics
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    active_channels = calculate_active_channels(yield_pct, total_channels=10000)
    health_status = check_system_health(cleaned_tensor, metadata, yield_pct)

    # Store results
    st.session_state.storage.save_tensor(cleaned_tensor, yield_pct, metadata)
    st.session_state.stability_tracker.add_yield(yield_pct)

    # Calculate centered signal in μV (for same-scale visualization)
    # This shows what the highpass filter does: removes DC offset
    raw_mean = float(np.mean(raw_chunk))
    centered_signal_uv = raw_chunk - raw_mean  # Centered at 0 in μV

    # Update session state for UI rendering
    st.session_state.raw_signal = raw_chunk
    st.session_state.cleaned_signal = cleaned_tensor
    st.session_state.centered_signal_uv = centered_signal_uv  # For same-scale plot
    st.session_state.raw_dc_offset = raw_mean  # For DC indicator
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
# Data Source Banner
# ============================================================================
if st.session_state.use_real_data:
    mode_description = "IIR Highpass (preserves brain rhythms)" if mode_key == "eeg" else "Polynomial Detrending"
    st.success(
        f"✅ **REAL DATA MODE**: Processing **actual PhysioNet EEG recordings** "
        f"(File: {st.session_state.real_data_file or 'None'}, "
        f"Channel: {st.session_state.real_data_loader.get_info().get('current_channel', 'N/A')}) | "
        f"**Algorithm**: {mode_description}" +
        (f" @ {highpass_cutoff}Hz cutoff" if mode_key == "eeg" else ""),
        icon="✅"
    )
else:
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

    # Latency and DC Offset metrics side by side
    if st.session_state.latest_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Pipeline Latency",
                f"{st.session_state.latest_metrics['latency_ms']:.2f} ms",
                delta=f"Budget: <20ms" if st.session_state.latest_metrics['latency_ms'] < 20 else "OVER BUDGET",
                delta_color="normal" if st.session_state.latest_metrics['latency_ms'] < 20 else "inverse"
            )
        with col2:
            # DC Offset Indicator - shows value of drift removal
            dc_offset = st.session_state.raw_dc_offset
            st.metric(
                "DC Offset Removed",
                f"{dc_offset:.1f} μV",
                delta="→ Centered at 0",
                delta_color="normal"
            )
        with col3:
            if st.session_state.raw_signal is not None:
                raw_range = float(np.max(st.session_state.raw_signal) - np.min(st.session_state.raw_signal))
                st.metric(
                    "Signal Range",
                    f"{raw_range:.1f} μV",
                    delta="Peak-to-peak"
                )

    if st.session_state.raw_signal is not None and st.session_state.centered_signal_uv is not None:
        # ====================================================================
        # SAME Y-SCALE VISUALIZATION
        # Shows raw signal WITH DC offset vs centered signal (offset removed)
        # This clearly demonstrates what the highpass filter does!
        # ====================================================================
        fig = go.Figure()

        num_samples = len(st.session_state.raw_signal)

        # Convert sample indices to time in milliseconds for better readability
        if st.session_state.use_real_data:
            info = st.session_state.real_data_loader.get_info()
            sample_rate = info.get('sampling_rate', 160.0)
            time_ms = np.arange(num_samples) / sample_rate * 1000
            x_axis = time_ms
            x_label = "Time (ms)"
            duration_ms = chunk_duration_ms
        else:
            sample_rate = 40000
            time_ms = np.arange(num_samples) / sample_rate * 1000
            x_axis = time_ms
            x_label = "Time (ms)"
            duration_ms = 50.0

        # Raw signal trace (red) - with DC offset
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=st.session_state.raw_signal,
                mode='lines',
                name=f'Raw Signal (DC offset: {st.session_state.raw_dc_offset:.1f}μV)',
                line=dict(color='red', width=1.5),
                opacity=0.8
            )
        )

        # Centered signal trace (cyan) - DC offset removed
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=st.session_state.centered_signal_uv,
                mode='lines',
                name='Centered Signal (DC removed)',
                line=dict(color='cyan', width=2)
            )
        )

        # Add horizontal line at y=0 to show centering
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text="Zero baseline",
            annotation_position="right"
        )

        # Add horizontal line at raw DC offset to show original baseline
        fig.add_hline(
            y=st.session_state.raw_dc_offset,
            line_dash="dot",
            line_color="red",
            opacity=0.3,
            annotation_text=f"Original DC: {st.session_state.raw_dc_offset:.1f}μV",
            annotation_position="left"
        )

        # Update layout
        fig.update_layout(
            title=f"DC Drift Removal Demonstration ({duration_ms:.0f}ms @ {sample_rate:.0f}Hz) - Same Y-Scale",
            xaxis_title=x_label,
            yaxis_title="Amplitude (μV)",
            template="plotly_dark",
            height=500,
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
            # Add annotation explaining what's shown
            annotations=[
                dict(
                    x=0.5,
                    y=-0.12,
                    xref='paper',
                    yref='paper',
                    text=f"<b>IIR Highpass filter removes DC drift while preserving brain rhythms</b> | "
                         f"Samples: {num_samples} | DC removed: {abs(st.session_state.raw_dc_offset):.1f}μV",
                    showarrow=False,
                    font=dict(size=11, color='gray')
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation text
        if st.session_state.use_real_data:
            st.caption(
                f"**What you're seeing**: Red = raw EEG with electrode drift (DC offset: {st.session_state.raw_dc_offset:.1f}μV). "
                f"Cyan = same signal centered at 0μV after highpass filtering. "
                f"The **brain rhythms (alpha/theta) are preserved** - only the DC baseline drift is removed."
            )

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

        # ±2σ confidence band - only show after enough data accumulated
        # This prevents the "triangle artifact" that appears when few data points exist
        smoothed_array = np.array(smoothed_yield_history)

        # Only show envelope after at least 20 epochs (enough for stable statistics)
        min_epochs_for_envelope = 20
        if len(smoothed_array) >= min_epochs_for_envelope:
            # Use a fixed window size for consistent envelope calculation
            window = min(stability_window, len(smoothed_array))
            kernel = np.ones(window) / window
            rolling_mean = np.convolve(smoothed_array, kernel, mode='same')

            # Calculate TRUE rolling std (per-epoch) to avoid expanding wedge artifact
            # This ensures envelope width stays consistent across all epochs
            def calculate_rolling_std(arr, win):
                """Calculate rolling standard deviation for each epoch."""
                result = np.zeros(len(arr))
                for i in range(len(arr)):
                    start = max(0, i - win + 1)
                    chunk = arr[start:i + 1]
                    result[i] = np.std(chunk) if len(chunk) > 1 else 0.0
                return result

            rolling_std_values = calculate_rolling_std(smoothed_array, window)
            upper_bound = rolling_mean + 2 * rolling_std_values
            lower_bound = rolling_mean - 2 * rolling_std_values

            # Clip bounds to valid range
            upper_bound = np.clip(upper_bound, 0, 105)
            lower_bound = np.clip(lower_bound, 0, 105)

            fig.add_trace(go.Scatter(
                x=np.concatenate([epoch_indices, epoch_indices[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(135, 206, 250, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±2σ Envelope',
                showlegend=True
            ))
        else:
            # Show accumulation message instead of broken envelope
            st.caption(f"Accumulating data... ({len(smoothed_array)}/{min_epochs_for_envelope} epochs for stability envelope)")

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
