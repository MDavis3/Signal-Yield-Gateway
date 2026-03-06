"""
Data Simulator - Mock Hardware Signal Generation
=================================================

Generates synthetic neural data that mimics flexible polymer electrode
output, including realistic micromotion artifacts for testing the DSP pipeline.

Key Functions:
--------------
- generate_synthetic_chunk: Creates 40kHz mock data with noise, spikes, and drift

Simulation Components:
----------------------
1. Background Gaussian noise (thermal/electrical noise)
2. Randomly injected sharp neural action potentials (spikes)
3. Severe low-frequency rolling sine wave (micromotion/respiration drift)

Author: Senior AI Software Engineer & BCI Data Architect
Date: 2026-03-02
"""

import numpy as np
from typing import Tuple

# Global phase state for continuous drift simulation across chunks
_drift_phase_state = {
    'heartbeat_phase': 0.0,
    'respiration_phase': 0.0,
    'initialized': False
}


def generate_synthetic_chunk(
    duration_ms: float = 50.0,
    sample_rate: int = 40000,
    noise_level: float = 0.3,
    drift_severity: float = 1.0,
    spike_rate: float = 20.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate a synthetic chunk of neural data mimicking flexible electrode output.

    This function simulates the three main components of real BCI data:

    1. **Background Noise** (Gaussian white noise)
       - Represents thermal noise from electronics + biological background activity
       - Typical amplitude: 10-50 μV RMS

    2. **Neural Action Potentials** (Sharp spike waveforms)
       - Simulates neurons firing near the electrode
       - Characteristic shape: sharp rising edge (~1ms), slower falling edge
       - Amplitude: 50-200 μV (varies with distance from neuron)
       - Rate: 5-50 spikes/second typical for single unit

    3. **Micromotion Baseline Drift** (Low-frequency sinusoid)
       - Simulates physical electrode movement from heartbeat/respiration
       - Frequency: 0.5-1Hz (heartbeat ~1Hz, respiration ~0.3Hz)
       - Amplitude: ±100-500 μV (severe for soft electrodes)
       - This is the PRIMARY PROBLEM we're solving with the DSP pipeline

    Parameters:
    -----------
    duration_ms : float
        Chunk duration in milliseconds (default: 50ms per chunk)
    sample_rate : int
        Sampling frequency in Hz (default: 40kHz = 40,000 samples/second)
        Hardware spec: 40kHz per channel
    noise_level : float
        Gaussian noise amplitude scaling factor (0.0-1.0)
        0.0 = pristine (unrealistic), 1.0 = very noisy
    drift_severity : float
        Micromotion drift amplitude scaling factor (0.0-2.0)
        0.0 = no drift (ideal case), 2.0 = severe drift (worst case)
    spike_rate : float
        Average number of spikes per second (default: 20 Hz)
        Typical single-unit firing rates: 5-50 Hz
    seed : int, optional
        Random seed for reproducibility (default: None = random)

    Returns:
    --------
    signal : np.ndarray
        Synthetic neural signal (shape: [n_samples], dtype: float32)
        Units: microvolts (μV)
        Contains: noise + spikes + drift

    Example:
    --------
    >>> # Generate 50ms chunk with severe drift and moderate noise
    >>> chunk = generate_synthetic_chunk(
    ...     duration_ms=50.0,
    ...     sample_rate=40000,
    ...     noise_level=0.5,
    ...     drift_severity=1.5,
    ...     spike_rate=25.0
    ... )
    >>> chunk.shape
    (2000,)  # 50ms * 40kHz = 2000 samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate number of samples
    n_samples = int((duration_ms / 1000.0) * sample_rate)

    # Time axis for waveform generation
    time_axis = np.linspace(0, duration_ms / 1000.0, n_samples)

    # ======================
    # Component 1: Background Gaussian Noise
    # ======================
    # Thermal/electrical noise is modeled as white Gaussian noise
    # Typical RMS: 10-50 μV
    noise_amplitude = 30.0  # μV baseline
    noise = noise_level * noise_amplitude * np.random.randn(n_samples)

    # ======================
    # Component 2: Neural Action Potentials (Spikes)
    # ======================
    # Generate random spike times based on Poisson process
    expected_spike_count = int(spike_rate * (duration_ms / 1000.0))
    spike_times = np.sort(np.random.uniform(0, duration_ms / 1000.0, expected_spike_count))

    # Initialize spike waveform array
    spikes = np.zeros(n_samples)

    # Spike waveform parameters
    spike_amplitude = 100.0  # μV (typical range: 50-200 μV)
    spike_width = 0.001  # 1ms characteristic width
    spike_rise_time = 0.0003  # 0.3ms rising edge (sharp)
    spike_fall_time = 0.0007  # 0.7ms falling edge (slower)

    # Inject each spike as a realistic action potential waveform
    for spike_time in spike_times:
        # Find closest sample index
        spike_idx = int(spike_time * sample_rate)

        if spike_idx >= n_samples:
            continue

        # Add amplitude jitter to simulate varying distance from neuron
        # Apply PER SPIKE, not to entire array (avoids compounding jitter)
        amplitude_jitter = np.random.uniform(0.7, 1.3)
        jittered_amplitude = spike_amplitude * amplitude_jitter

        # Generate realistic spike waveform (asymmetric)
        # Rising edge: exponential rise
        # Falling edge: exponential decay
        for i in range(n_samples):
            t_rel = time_axis[i] - spike_time  # Time relative to spike onset

            if t_rel >= 0 and t_rel < spike_rise_time:
                # Rising phase (sharp)
                spikes[i] += jittered_amplitude * (t_rel / spike_rise_time)
            elif t_rel >= spike_rise_time and t_rel < (spike_rise_time + spike_fall_time):
                # Falling phase (slower decay)
                decay_time = t_rel - spike_rise_time
                spikes[i] += jittered_amplitude * np.exp(-decay_time / (spike_fall_time / 3))

    # ======================
    # Component 3: Micromotion Baseline Drift
    # ======================
    # Low-frequency sinusoidal drift simulating electrode movement
    # from heartbeat (1 Hz) and respiration (0.3 Hz)

    # CONTINUOUS DRIFT: Maintain phase continuity across chunks for realistic streaming
    global _drift_phase_state

    # Initialize phase on first call (or after reset)
    if not _drift_phase_state['initialized']:
        _drift_phase_state['heartbeat_phase'] = np.random.uniform(0, 2 * np.pi)
        _drift_phase_state['respiration_phase'] = np.random.uniform(0, 2 * np.pi)
        _drift_phase_state['initialized'] = True

    # Use continuous phase from previous chunk
    heartbeat_phase = _drift_phase_state['heartbeat_phase']
    respiration_phase = _drift_phase_state['respiration_phase']

    # Primary drift: heartbeat component (1 Hz)
    heartbeat_freq = 1.0  # Hz
    heartbeat_amplitude = 200.0 * drift_severity  # μV (scales with severity)
    heartbeat_drift = heartbeat_amplitude * np.sin(2 * np.pi * heartbeat_freq * time_axis + heartbeat_phase)

    # Secondary drift: respiration component (0.3 Hz)
    respiration_freq = 0.3  # Hz
    respiration_amplitude = 150.0 * drift_severity  # μV
    respiration_drift = respiration_amplitude * np.sin(2 * np.pi * respiration_freq * time_axis + respiration_phase)

    # Total drift is combination of both components
    drift = heartbeat_drift + respiration_drift

    # Update phase state for next chunk (advance by chunk duration)
    _drift_phase_state['heartbeat_phase'] += 2 * np.pi * heartbeat_freq * (duration_ms / 1000.0)
    _drift_phase_state['respiration_phase'] += 2 * np.pi * respiration_freq * (duration_ms / 1000.0)

    # Wrap phases to [0, 2π) to prevent overflow
    _drift_phase_state['heartbeat_phase'] %= (2 * np.pi)
    _drift_phase_state['respiration_phase'] %= (2 * np.pi)

    # ======================
    # Combine All Components
    # ======================
    signal = noise + spikes + drift

    # Convert to float32 for memory efficiency (PyTorch-compatible)
    signal = signal.astype(np.float32)

    return signal


def generate_batch(
    batch_size: int = 10,
    duration_ms: float = 50.0,
    sample_rate: int = 40000,
    noise_level: float = 0.3,
    drift_severity: float = 1.0,
    spike_rate: float = 20.0
) -> np.ndarray:
    """
    Generate a batch of synthetic chunks for testing.

    Useful for benchmarking DSP pipeline throughput and stability over
    multiple consecutive chunks.

    Parameters:
    -----------
    batch_size : int
        Number of chunks to generate (default: 10)
    (other parameters same as generate_synthetic_chunk)

    Returns:
    --------
    batch : np.ndarray
        Batch of synthetic signals (shape: [batch_size, n_samples], float32)

    Example:
    --------
    >>> batch = generate_batch(batch_size=100, drift_severity=1.5)
    >>> batch.shape
    (100, 2000)
    """
    n_samples = int((duration_ms / 1000.0) * sample_rate)
    batch = np.zeros((batch_size, n_samples), dtype=np.float32)

    for i in range(batch_size):
        batch[i] = generate_synthetic_chunk(
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            noise_level=noise_level,
            drift_severity=drift_severity,
            spike_rate=spike_rate,
            seed=None  # Different random seed for each chunk
        )

    return batch


def reset_drift_phase():
    """
    Reset the global drift phase state.

    Call this when starting a new recording session to randomize the initial
    phase offsets. This ensures each session starts with different drift patterns.
    """
    global _drift_phase_state
    _drift_phase_state['heartbeat_phase'] = 0.0
    _drift_phase_state['respiration_phase'] = 0.0
    _drift_phase_state['initialized'] = False


def estimate_snr(signal: np.ndarray, spike_amplitude: float = 100.0) -> float:
    """
    Estimate Signal-to-Noise Ratio (SNR) of the generated signal.

    SNR = 10 * log10(P_signal / P_noise)

    Parameters:
    -----------
    signal : np.ndarray
        Generated synthetic signal
    spike_amplitude : float
        Expected spike amplitude in μV (default: 100.0)

    Returns:
    --------
    snr_db : float
        Estimated SNR in decibels (dB)

    Note:
    -----
    This is an approximate SNR estimate assuming:
    - Signal power ≈ spike power
    - Noise power ≈ variance of residual after removing spikes
    """
    signal_power = spike_amplitude ** 2
    noise_power = np.var(signal)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
