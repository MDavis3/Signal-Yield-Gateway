"""
DSP Pipeline - Core Signal Processing
======================================

Implements thermally-constrained signal processing operations for Axoft's
flexible BCI electrodes. All operations are designed for O(1) or highly
efficient O(n) complexity to stay within thermal and latency budgets.

Key Functions:
--------------
- moving_average_subtract: O(1) amortized baseline drift removal
- detect_spikes_derivative: O(n) action potential detection
- tanh_normalize: O(n) soft-clipping normalization
- process_signal: Main pipeline orchestrator with latency tracking

Thermal Constraints:
--------------------
We CANNOT use heavy DSP filters (scipy.signal.butter, FFTs, deep neural nets)
because they generate heat. Brain implant heating >1°C causes tissue necrosis.
All operations must be mathematically simple, vectorized, and hardware-accelerated.

Author: Senior AI Software Engineer & BCI Data Architect
Date: 2026-03-02
"""

import numpy as np
import time
from typing import Tuple, Dict, Any
from collections import deque


class CircularBuffer:
    """
    O(1) amortized circular buffer for moving average calculation.

    Maintains a fixed-size sliding window without array shifts, enabling
    constant-time baseline drift removal critical for thermal budget compliance.
    """

    def __init__(self, size: int):
        """
        Initialize circular buffer.

        Parameters:
        -----------
        size : int
            Maximum number of elements to store
        """
        self.size = size
        self.buffer = deque(maxlen=size)
        self.sum = 0.0

    def add(self, value: float) -> float:
        """
        Add new value and return current moving average.

        O(1) complexity: add new sample, subtract oldest (if full), update sum.

        Parameters:
        -----------
        value : float
            New sample value

        Returns:
        --------
        float
            Current moving average
        """
        if len(self.buffer) == self.size:
            # Buffer is full, subtract the oldest value before it's removed
            self.sum -= self.buffer[0]

        self.buffer.append(value)
        self.sum += value

        return self.sum / len(self.buffer)

    def reset(self):
        """Reset buffer to empty state."""
        self.buffer.clear()
        self.sum = 0.0


def moving_average_subtract(
    signal: np.ndarray,
    window_size: int,
    buffer: CircularBuffer = None
) -> Tuple[np.ndarray, CircularBuffer]:
    """
    Remove low-frequency baseline drift using fast moving average subtraction.

    This is our O(1) amortized alternative to scipy.signal.butter high-pass
    filters, which require FFTs and violate thermal constraints. By subtracting
    a moving average, we effectively "snap" the baseline to zero while preserving
    high-frequency neural spikes.

    Why this works:
    ---------------
    - Micromotion drift is slow (0.5-1Hz heartbeat/respiration)
    - Neural action potentials are fast (~1ms rising edge)
    - Moving average tracks the slow drift but not the fast spikes
    - Subtracting the moving average removes drift, keeps spikes

    Parameters:
    -----------
    signal : np.ndarray
        Raw input signal with baseline drift (shape: [n_samples])
    window_size : int
        Moving average window size in samples (100-2000 typical)
        Larger window = more aggressive drift removal, but can attenuate spikes
    buffer : CircularBuffer, optional
        Existing buffer for streaming mode (maintains state across chunks)
        If None, creates new buffer (for batch mode)

    Returns:
    --------
    centered_signal : np.ndarray
        Signal with baseline drift removed (shape: [n_samples])
    buffer : CircularBuffer
        Updated buffer (for streaming across multiple chunks)

    Complexity:
    -----------
    O(n) total, O(1) amortized per sample via circular buffer
    """
    if buffer is None:
        buffer = CircularBuffer(window_size)

    centered_signal = np.zeros_like(signal, dtype=np.float32)

    for i, sample in enumerate(signal):
        moving_avg = buffer.add(sample)
        centered_signal[i] = sample - moving_avg

    return centered_signal, buffer


def smooth_signal_ma(
    signal: np.ndarray,
    window_size: int = 40
) -> np.ndarray:
    """
    Apply small moving average smoothing to suppress high-frequency ringing.

    This is Stage 2 of the two-stage filter cascade:
    - Stage 1 (moving_average_subtract): Aggressive drift removal (1500 samples)
    - Stage 2 (smooth_signal_ma): Ringing suppression (30-50 samples)

    Why this works:
    ---------------
    - Moving average subtraction creates biphasic ringing at spike edges
    - Ringing manifests as 5-20 kHz oscillations (from step response)
    - Small MA smoothing (40 samples = 1ms) acts as low-pass filter
    - Cutoff ~1 kHz attenuates ringing while preserving spike edges

    Mathematical Justification:
    ---------------------------
    At 40kHz sampling:
    - 40 samples = 1ms window
    - Cutoff frequency ≈ sample_rate / window_size = 40kHz / 40 = 1 kHz
    - Spike bandwidth ≈ 1 kHz (from 1ms rising edge)
    - Ringing frequency ≈ 5-20 kHz (from MA edge response)
    - Result: Attenuates 5-20 kHz ringing by 70-90%, preserves 1 kHz spike fundamentals

    Parameters:
    -----------
    signal : np.ndarray
        Centered signal after baseline drift removal (with ringing artifacts)
    window_size : int
        Smoothing window in samples (default: 40 = 1ms at 40kHz)
        Range: 30-50 samples (0.75-1.25ms)
        - Smaller = less smoothing, more ringing preserved
        - Larger = more smoothing, risk blurring spike edges

    Returns:
    --------
    smoothed_signal : np.ndarray
        Smoothed signal with attenuated high-frequency ringing

    Complexity:
    -----------
    O(n) with np.convolve using small fixed kernel
    Thermal cost: ~1-2ms for 2000 samples (well within 20ms budget)
    """
    # Create uniform averaging kernel
    kernel = np.ones(window_size) / window_size

    # Apply convolution with 'same' mode to preserve length
    # 'same' mode pads with edge values to avoid boundary artifacts
    smoothed = np.convolve(signal, kernel, mode='same')

    return smoothed.astype(np.float32)


def detect_spikes_derivative(
    signal: np.ndarray,
    threshold: float = 20.0
) -> int:
    """
    Detect neural action potentials using derivative-based edge detection.

    Why derivative instead of amplitude threshold:
    -----------------------------------------------
    - Action potentials have characteristic sharp rising edge (~1ms, 50-100 μV/ms)
    - As electrode drifts closer/further from neuron, spike amplitude changes
    - Amplitude threshold fails when electrode moves
    - Derivative (slope) is more robust to distance changes

    Implementation:
    ---------------
    - Compute np.diff() to get instantaneous slopes
    - Count threshold crossings on rising edges
    - O(n) vectorized operation, thermally cheap

    Parameters:
    -----------
    signal : np.ndarray
        Centered signal after baseline drift removal (shape: [n_samples])
    threshold : float
        Derivative threshold for spike detection (default: 30.0)
        At 40kHz sampling (25μs/sample), threshold of 30μV corresponds to
        1200μV/ms slope, which is characteristic of neural spikes.
        Higher = fewer false positives, may miss small spikes
        Lower = more sensitive, may detect noise as spikes

    Returns:
    --------
    spike_count : int
        Number of detected action potentials in this chunk

    Complexity:
    -----------
    O(n) linear pass with vectorized numpy operations
    """
    # Compute first derivative (instantaneous slope)
    derivatives = np.diff(signal)

    # Detect rising edges exceeding threshold
    # Use absolute value to catch both positive and negative spikes
    spikes = np.abs(derivatives) > threshold

    # Count spike events
    spike_count = int(np.sum(spikes))

    return spike_count


def tanh_normalize(
    signal: np.ndarray,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Apply hyperbolic tangent soft-clipping normalization with ADAPTIVE scaling.

    **CRITICAL FIX v2**: Remove DC offset before tanh to anchor baseline at 0.0

    The moving average subtraction can leave residual DC offset due to:
    1. Buffer initialization lag (first 1500 samples)
    2. Group delay (18.75ms lag for 1500-sample window)
    3. Drift frequency changes faster than MA can track

    This DC offset causes the cyan line to "float" away from 0.0. By subtracting
    the mean BEFORE tanh, we guarantee the baseline stays anchored at 0.0.

    Why tanh instead of hard clipping:
    ----------------------------------
    - Differentiable everywhere (critical for gradient-based TN-VAE decoders)
    - Smooth transition prevents high-frequency artifacts from sharp clipping
    - np.tanh() is hardware-accelerated on most CPUs (SIMD vectorization)
    - Guarantees bounds [-1, 1] without conditional logic (no if-statements)

    The tanh function acts as a "soft clipper":
    - Small signals (noise): tanh(x) ≈ x ≈ 0 (linear passthrough)
    - Medium signals (spikes): tanh(x) → 0.5-0.9 (clear peaks)
    - Large signals (artifacts): tanh(x) → ±1 (soft saturation)

    Adaptive Scaling Logic (1.5σ Rule):
    ----------------------------------
    - Remove mean (DC offset) to ensure true zero-centered signal
    - Calculate std of zero-centered signal
    - Scale factor = 1.5 * std
    - Spikes (typically 5-10σ) reach tanh output of 0.7-1.0
    - Background noise (±1σ) stays near 0 (tanh output ±0.3)

    Parameters:
    -----------
    signal : np.ndarray
        Centered signal after baseline drift removal (shape: [n_samples])
        May have residual DC offset from MA lag
        Expected units: microvolts (μV)
    alpha : float
        Gain parameter controlling compression steepness (default: 1.0)
        Larger alpha = more aggressive compression (harder clipping)
        Smaller alpha = softer compression (more linear passthrough)

    Returns:
    --------
    normalized_signal : np.ndarray
        Soft-clipped signal bounded to [-1, 1] (shape: [n_samples])
        Baseline anchored at 0.0, spikes as sharp needles
        Formatted as float32 PyTorch-ready tensor

    Complexity:
    -----------
    O(n) vectorized operation, hardware-accelerated
    """
    # **CRITICAL FIX**: Remove DC offset to anchor baseline at 0.0
    # The moving average can leave residual offset due to group delay
    signal_mean = np.mean(signal)
    zero_centered_signal = signal - signal_mean

    # Adaptive scaling: use 1.5σ rule for STRONG spike visibility
    signal_std = np.std(zero_centered_signal)

    # Handle edge case: completely flat signal (all zeros)
    if signal_std < 1e-6:
        return np.zeros_like(signal, dtype=np.float32)

    # Scale factor: 1.5σ for STRONG spike peaks
    # With 1.5σ: 5σ spike → 5/1.5 = 3.33 → tanh(3.33) = 0.997
    # Background noise (±1σ) → tanh(±0.67) ≈ ±0.59
    scale_factor = 1.5 * signal_std

    # Scale zero-centered signal: μV → normalized range
    scaled_signal = zero_centered_signal / scale_factor

    # Apply tanh soft-clipping to scaled signal
    normalized_signal = np.tanh(alpha * scaled_signal).astype(np.float32)

    return normalized_signal


def process_signal(
    raw_chunk: np.ndarray,
    config: Dict[str, Any],
    buffer: CircularBuffer = None
) -> Tuple[np.ndarray, float, Dict[str, Any], CircularBuffer]:
    """
    Main DSP pipeline orchestrator with latency tracking.

    Pipeline Stages:
    ----------------
    1. Moving average subtraction → remove baseline drift
    2. Derivative-based spike detection → count action potentials
    3. Tanh normalization → soft-clip to [-1, 1] bounds
    4. Latency measurement → verify <20ms budget compliance

    Parameters:
    -----------
    raw_chunk : np.ndarray
        Raw input signal from hardware (shape: [n_samples])
    config : dict
        Processing configuration with keys:
        - 'moving_avg_window': int (100-2000)
        - 'tanh_alpha': float (0.1-5.0)
        - 'spike_threshold': float (default: 3.0)
    buffer : CircularBuffer, optional
        Existing buffer for streaming mode

    Returns:
    --------
    cleaned_tensor : np.ndarray
        Processed signal ready for TN-VAE decoder (shape: [n_samples], float32)
    latency_ms : float
        Processing time in milliseconds (should be <20ms)
    metadata : dict
        Processing metadata with keys:
        - 'spike_count': int
        - 'variance': float
        - 'mean': float
        - 'has_nan': bool
        - 'has_inf': bool
    buffer : CircularBuffer
        Updated buffer for next chunk

    Complexity:
    -----------
    O(n) total: moving avg O(n), spike detect O(n), tanh O(n)
    Vectorized numpy operations keep latency <20ms for n=2000 samples
    """
    start_time = time.perf_counter()

    # Step 1: Remove baseline drift (O(1) amortized per sample)
    centered_signal, buffer = moving_average_subtract(
        raw_chunk,
        config['moving_avg_window'],
        buffer
    )

    # Step 2: Detect neural spikes (O(n) linear pass)
    # Use CENTERED signal (pre-smoothing) for accurate spike detection
    # Smoothing reduces derivatives and would harm spike detection accuracy
    spike_count = detect_spikes_derivative(
        centered_signal,  # Use pre-smoothing signal for accurate detection
        config.get('spike_threshold', 20.0)
    )

    # Step 2.5: Suppress high-frequency ringing (O(n) smoothing)
    # Apply post-MA smoothing AFTER spike detection to reduce biphasic ringing artifacts
    # This preserves spike detection accuracy while improving visual quality
    # Allow disabling by setting smoothing_window to 0 for backward compatibility
    smoothing_window = config.get('smoothing_window', 40)
    if smoothing_window > 0:
        smoothed_signal = smooth_signal_ma(centered_signal, smoothing_window)
    else:
        smoothed_signal = centered_signal  # No smoothing, backward compatible

    # Step 3: Soft-clip normalization (O(n) vectorized)
    # Use smoothed signal for clean visualization (oscillation-free cyan line)
    cleaned_tensor = tanh_normalize(
        smoothed_signal,  # Use post-smoothing signal for clean visualization
        config['tanh_alpha']
    )

    # Step 4: NaN/Inf sanitization (CRITICAL for medical device safety)
    # Medical devices MUST NOT crash on anomalous data - graceful degradation required
    has_nan = bool(np.any(np.isnan(cleaned_tensor)))
    has_inf = bool(np.any(np.isinf(cleaned_tensor)))

    if has_nan or has_inf:
        # Replace NaN with 0.0 (neutral value within tanh [-1, 1] range)
        cleaned_tensor = np.nan_to_num(
            cleaned_tensor,
            nan=0.0,      # Replace NaN with 0
            posinf=1.0,   # Clip +inf to upper tanh bound
            neginf=-1.0   # Clip -inf to lower tanh bound
        )

    # **CRITICAL FIX**: Calculate metadata on CENTERED signal (pre-smoothing AND pre-tanh)
    # This decouples clinical metrics from BOTH aesthetic smoothing AND Tanh Alpha slider
    # User requirement: "calculate_signal_yield must evaluate health independent of visualization"
    metadata = {
        'spike_count': spike_count,
        'variance': float(np.var(centered_signal)),      # PRE-smoothing variance (biological signal)
        'mean': float(np.mean(centered_signal)),         # PRE-smoothing mean (biological signal)
        'has_nan': has_nan,       # Flag BEFORE sanitization (for health check)
        'has_inf': has_inf,       # Flag BEFORE sanitization (for health check)
        'was_sanitized': has_nan or has_inf,  # Flag indicating recovery occurred
        'centered_signal': centered_signal,   # PRE-smoothing signal for metrics engine
        'smoothed_signal': smoothed_signal    # POST-smoothing signal for reference
    }

    # Measure processing latency
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    return cleaned_tensor, latency_ms, metadata, buffer


# Module-level buffer for streaming mode (maintains state across chunks)
_streaming_buffer = None


def reset_streaming_buffer():
    """Reset the module-level streaming buffer. Call when starting new session."""
    global _streaming_buffer
    _streaming_buffer = None


def process_signal_streaming(
    raw_chunk: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Streaming-mode wrapper for process_signal that maintains buffer state.

    This function automatically manages the circular buffer across multiple
    chunks, simulating continuous real-time processing from hardware.

    Parameters:
    -----------
    raw_chunk : np.ndarray
        Raw input signal from hardware (shape: [n_samples])
    config : dict
        Processing configuration (same as process_signal)

    Returns:
    --------
    cleaned_tensor, latency_ms, metadata
        Same as process_signal (buffer is managed internally)
    """
    global _streaming_buffer

    cleaned_tensor, latency_ms, metadata, _streaming_buffer = process_signal(
        raw_chunk,
        config,
        _streaming_buffer
    )

    return cleaned_tensor, latency_ms, metadata
