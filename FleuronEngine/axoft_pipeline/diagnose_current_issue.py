"""
Diagnose why cyan line is crazy with current parameters.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal

print("=" * 80)
print("DIAGNOSTIC: Current Issue with Cyan Line")
print("=" * 80)

# User's exact parameters
drift_severity = 0.40
noise_level = 0.30
tanh_alpha = 1.0
smoothing_window = 0
poly_order = 1

print(f"\nParameters:")
print(f"  Drift Severity: {drift_severity}")
print(f"  Noise Level: {noise_level}")
print(f"  Tanh Alpha: {tanh_alpha}")
print(f"  Smoothing Window: {smoothing_window}")
print(f"  Poly Order: {poly_order}")

# Generate synthetic chunk
raw_chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=noise_level,
    drift_severity=drift_severity,
    spike_rate=20.0
)

print(f"\nRaw Signal Statistics:")
print(f"  Mean: {np.mean(raw_chunk):.2f} uV")
print(f"  Std: {np.std(raw_chunk):.2f} uV")
print(f"  Min: {np.min(raw_chunk):.2f} uV")
print(f"  Max: {np.max(raw_chunk):.2f} uV")
print(f"  Range: {np.max(raw_chunk) - np.min(raw_chunk):.2f} uV")

# Process through pipeline
config = {
    'poly_order': poly_order,
    'tanh_alpha': tanh_alpha,
    'spike_threshold': 5.0,
    'smoothing_window': smoothing_window
}

cleaned, latency_ms, metadata, _ = process_signal(raw_chunk, config, buffer=None)

print(f"\nCleaned Signal Statistics:")
print(f"  Mean: {np.mean(cleaned):.6f}")
print(f"  Std: {np.std(cleaned):.6f}")
print(f"  Min: {np.min(cleaned):.6f}")
print(f"  Max: {np.max(cleaned):.6f}")
print(f"  Range: {np.max(cleaned) - np.min(cleaned):.6f}")

# Check for extreme oscillations
derivatives = np.diff(cleaned)
rapid_oscillations = np.sum(np.abs(derivatives) > 0.1)
oscillation_pct = (rapid_oscillations / len(cleaned)) * 100

print(f"\nOscillation Metrics:")
print(f"  Rapid oscillations (|diff| > 0.1): {oscillation_pct:.1f}% of samples")
print(f"  Max derivative: {np.max(np.abs(derivatives)):.4f}")

# Sample values from cleaned signal
print(f"\nFirst 20 samples of cleaned signal:")
print(cleaned[:20])

print(f"\nLast 20 samples of cleaned signal:")
print(cleaned[-20:])

# Check if signal is mostly noise
signal_power = np.var(cleaned)
print(f"\nSignal Power (variance): {signal_power:.6f}")

if oscillation_pct > 50:
    print("\n" + "=" * 80)
    print("[PROBLEM DETECTED] Cleaned signal has excessive oscillations!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("[OK] Cleaned signal looks reasonable")
    print("=" * 80)
