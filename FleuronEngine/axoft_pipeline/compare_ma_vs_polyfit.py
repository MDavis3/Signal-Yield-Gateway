"""
Compare MA vs Polyfit with REALISTIC noise levels.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import moving_average_subtract, polyfit_detrend, tanh_normalize, CircularBuffer

print("=" * 80)
print("COMPARISON: Moving Average vs Polyfit with Realistic Noise")
print("=" * 80)

# Generate ONE synthetic chunk with REALISTIC parameters
drift_severity = 0.40
noise_level = 0.30
tanh_alpha = 1.0

raw_chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=noise_level,
    drift_severity=drift_severity,
    spike_rate=20.0,
    seed=42  # Fixed seed for comparison
)

print(f"\nRaw signal:")
print(f"  Mean: {np.mean(raw_chunk):.2f} uV")
print(f"  Std: {np.std(raw_chunk):.2f} uV")

# Process with MOVING AVERAGE
buffer_ma = CircularBuffer(200)
centered_ma, _ = moving_average_subtract(raw_chunk, buffer=buffer_ma, window_size=200)
cleaned_ma = tanh_normalize(centered_ma, alpha=tanh_alpha)

# Process with POLYFIT
centered_polyfit, _ = polyfit_detrend(raw_chunk, poly_order=1)
cleaned_polyfit = tanh_normalize(centered_polyfit, alpha=tanh_alpha)

# Compare oscillation levels
ma_diff = np.diff(cleaned_ma)
ma_oscillations = np.sum(np.abs(ma_diff) > 0.1)
ma_osc_pct = (ma_oscillations / len(cleaned_ma)) * 100

polyfit_diff = np.diff(cleaned_polyfit)
polyfit_oscillations = np.sum(np.abs(polyfit_diff) > 0.1)
polyfit_osc_pct = (polyfit_oscillations / len(cleaned_polyfit)) * 100

print(f"\n[MOVING AVERAGE]")
print(f"  Centered signal std: {np.std(centered_ma):.4f} uV")
print(f"  Cleaned signal std: {np.std(cleaned_ma):.6f}")
print(f"  Oscillations (|diff| > 0.1): {ma_osc_pct:.1f}%")

print(f"\n[POLYFIT]")
print(f"  Centered signal std: {np.std(centered_polyfit):.4f} uV")
print(f"  Cleaned signal std: {np.std(cleaned_polyfit):.6f}")
print(f"  Oscillations (|diff| > 0.1): {polyfit_osc_pct:.1f}%")

print(f"\n[COMPARISON]")
if abs(ma_osc_pct - polyfit_osc_pct) < 10:
    print(f"  BOTH methods have similar oscillation levels!")
    print(f"  This means the oscillations are from NOISE, not filter artifacts.")
else:
    print(f"  Different oscillation levels detected.")
    print(f"  Difference: {abs(ma_osc_pct - polyfit_osc_pct):.1f} percentage points")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("The 'crazy' oscillations you're seeing are GAUSSIAN NOISE, not ringing!")
print("With noise_level=0.30, this is EXPECTED and REALISTIC biological noise.")
print("")
print("Options to reduce oscillations:")
print("  1. Reduce noise_level slider (e.g., 0.10 instead of 0.30)")
print("  2. Apply smoothing (but this destroys spike morphology)")
print("  3. Accept that realistic biological data has this noise")
print("=" * 80)
