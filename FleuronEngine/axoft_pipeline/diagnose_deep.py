"""
Deep diagnostic to trace through each pipeline stage.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import polyfit_detrend, tanh_normalize, detect_spikes_derivative

print("=" * 80)
print("DEEP DIAGNOSTIC: Tracing Through Each Pipeline Stage")
print("=" * 80)

# User's exact parameters
drift_severity = 0.40
noise_level = 0.30
tanh_alpha = 1.0

# Generate synthetic chunk
raw_chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=noise_level,
    drift_severity=drift_severity,
    spike_rate=20.0
)

print(f"\n[STAGE 1] Raw Signal:")
print(f"  Mean: {np.mean(raw_chunk):.2f} uV")
print(f"  Std: {np.std(raw_chunk):.2f} uV")
print(f"  Range: [{np.min(raw_chunk):.2f}, {np.max(raw_chunk):.2f}]")
print(f"  First 10: {raw_chunk[:10]}")

# Apply polyfit detrending
centered_signal, _ = polyfit_detrend(raw_chunk, poly_order=1, buffer=None)

print(f"\n[STAGE 2] After Polyfit Detrend:")
print(f"  Mean: {np.mean(centered_signal):.6f} uV")
print(f"  Std: {np.std(centered_signal):.6f} uV")
print(f"  Range: [{np.min(centered_signal):.2f}, {np.max(centered_signal):.2f}]")
print(f"  First 10: {centered_signal[:10]}")

# Check for ringing in centered signal
centered_derivatives = np.diff(centered_signal)
centered_oscillations = np.sum(np.abs(centered_derivatives) > 5.0)  # 5 uV threshold
centered_osc_pct = (centered_oscillations / len(centered_signal)) * 100
print(f"  Oscillation check: {centered_osc_pct:.1f}% samples with |diff| > 5uV")

# Apply tanh normalization
cleaned_signal = tanh_normalize(centered_signal, alpha=tanh_alpha)

print(f"\n[STAGE 3] After Tanh Normalize:")
print(f"  Mean: {np.mean(cleaned_signal):.6f}")
print(f"  Std: {np.std(cleaned_signal):.6f}")
print(f"  Range: [{np.min(cleaned_signal):.6f}, {np.max(cleaned_signal):.6f}]")
print(f"  First 10: {cleaned_signal[:10]}")

# Check for ringing in cleaned signal
cleaned_derivatives = np.diff(cleaned_signal)
cleaned_oscillations = np.sum(np.abs(cleaned_derivatives) > 0.1)
cleaned_osc_pct = (cleaned_oscillations / len(cleaned_signal)) * 100
print(f"  Oscillation check: {cleaned_osc_pct:.1f}% samples with |diff| > 0.1")

print(f"\n[ANALYSIS]")
if centered_osc_pct > 50:
    print("  Problem is in POLYFIT DETRENDING stage!")
    print("  Centered signal already has excessive oscillations before tanh.")
else:
    print("  Polyfit detrending looks OK.")

if cleaned_osc_pct > 50:
    print("  Problem is in TANH NORMALIZATION or earlier!")
else:
    print("  Tanh normalization looks OK.")

# Let's check what tanh_normalize is doing internally
print(f"\n[TANH NORMALIZE INTERNAL STEPS]")
signal_mean = np.mean(centered_signal)
print(f"  Step 1 - Signal mean: {signal_mean:.6f}")
recentered = centered_signal - signal_mean
print(f"  Step 2 - After re-centering: mean={np.mean(recentered):.6f}, std={np.std(recentered):.6f}")
std_val = max(np.std(recentered), 1e-6)
print(f"  Step 3 - Std value: {std_val:.6f}")
scaled = recentered / std_val
print(f"  Step 4 - After scaling: mean={np.mean(scaled):.6f}, std={np.std(scaled):.6f}")
print(f"           Range: [{np.min(scaled):.2f}, {np.max(scaled):.2f}]")
final = np.tanh(tanh_alpha * scaled)
print(f"  Step 5 - After tanh: mean={np.mean(final):.6f}, std={np.std(final):.6f}")
print(f"           Range: [{np.min(final):.6f}, {np.max(final):.6f}]")

print("\n" + "=" * 80)
