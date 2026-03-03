"""
Test strict order of operations fix for baseline float bug.

Verifies that extreme negative drift doesn't cause baseline to float at 0.8.
"""

import numpy as np
from dsp_pipeline import process_signal, CircularBuffer

print("=" * 80)
print("STRICT ORDER OF OPERATIONS TEST")
print("Testing: Extreme negative drift baseline anchoring")
print("=" * 80)

# Generate signal with EXTREME negative drift to expose the bug
# Raw signal will be centered around -200 uV instead of 0
n_samples = 2000
time_axis = np.linspace(0, 0.05, n_samples)  # 50ms

# Create extreme negative baseline
extreme_negative_drift = -200.0  # uV (very negative baseline)

# Add small spikes on top of the extreme negative baseline
spike_amplitude = 50.0  # uV
spike_times = [0.01, 0.02, 0.03]  # seconds

raw_signal = np.full(n_samples, extreme_negative_drift, dtype=np.float32)

# Add spikes
for spike_time in spike_times:
    spike_idx = int(spike_time * 40000)  # 40kHz sampling
    if spike_idx < n_samples:
        # Sharp spike waveform
        for i in range(max(0, spike_idx-10), min(n_samples, spike_idx+20)):
            t_rel = (i - spike_idx) / 40000.0
            if 0 <= t_rel < 0.0003:  # Rising edge
                raw_signal[i] += spike_amplitude * (t_rel / 0.0003)
            elif 0.0003 <= t_rel < 0.001:  # Falling edge
                decay = t_rel - 0.0003
                raw_signal[i] += spike_amplitude * np.exp(-decay / 0.0002)

# Add noise
raw_signal += np.random.randn(n_samples) * 5.0  # 5 uV noise

print(f"\nRaw Signal Characteristics:")
print(f"  Mean: {np.mean(raw_signal):.2f} uV (extreme negative drift)")
print(f"  Min:  {np.min(raw_signal):.2f} uV")
print(f"  Max:  {np.max(raw_signal):.2f} uV")
print(f"  Range: {np.max(raw_signal) - np.min(raw_signal):.2f} uV")

# Process with FIXED pipeline
config = {
    'moving_avg_window': 400,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 0  # NO smoothing
}

cleaned, latency, metadata, buffer = process_signal(
    raw_signal,
    config,
    buffer=CircularBuffer(400)
)

print(f"\nProcessed Signal Characteristics:")
print(f"  Centered mean: {metadata['mean']:.4f} uV (should be ~0.0)")
print(f"  Centered std:  {np.std(metadata['centered_signal']):.2f} uV")

print(f"\nCleaned (Tanh-Normalized) Signal:")
print(f"  Mean: {np.mean(cleaned):.6f} (MUST be ~0.0, NOT 0.8!)")
print(f"  Min:  {np.min(cleaned):.4f}")
print(f"  Max:  {np.max(cleaned):.4f}")
print(f"  Std:  {np.std(cleaned):.4f}")

# Critical checks
baseline_mean = np.mean(cleaned)
baseline_float = abs(baseline_mean) > 0.1  # Tolerance: |mean| < 0.1

# Find spike regions (where cleaned signal is high)
spike_mask = np.abs(cleaned) > 0.5
spike_values = cleaned[spike_mask]

if len(spike_values) > 0:
    spike_flatness = np.sum(spike_values > 0.95) / len(spike_values) * 100
    spike_mean = np.mean(spike_values)
    print(f"\nSpike Characteristics:")
    print(f"  Spike samples: {len(spike_values)}")
    print(f"  Spike mean value: {spike_mean:.4f}")
    print(f"  Flat plateau (>0.95): {spike_flatness:.1f}% of spike samples")
    print(f"  Max spike value: {np.max(spike_values):.4f}")
else:
    spike_flatness = 0
    print(f"\nWARNING: No spikes detected in cleaned signal!")

print("\n" + "=" * 80)
print("VERIFICATION RESULTS")
print("=" * 80)

# Test 1: Baseline anchored at 0.0
if abs(baseline_mean) < 0.1:
    print(f"\n[PASS] Baseline anchored at 0.0 (mean = {baseline_mean:.6f})")
else:
    print(f"\n[FAIL] Baseline floating! (mean = {baseline_mean:.6f}, expected ~0.0)")
    print(f"       This indicates the order of operations bug is NOT fixed!")

# Test 2: Spikes are sharp needles, not flat blocks
if spike_flatness < 30:  # Less than 30% of spike should be flat plateau
    print(f"[PASS] Spikes are sharp needles ({spike_flatness:.1f}% flat plateau)")
else:
    print(f"[FAIL] Spikes are flat blocks! ({spike_flatness:.1f}% flat plateau)")
    print(f"       This indicates spikes are slamming into tanh ceiling!")

# Test 3: Centered signal has zero mean
if abs(metadata['mean']) < 1.0:  # Within 1 uV of zero
    print(f"[PASS] Centered signal has zero mean ({metadata['mean']:.4f} uV)")
else:
    print(f"[FAIL] Centered signal has DC offset! ({metadata['mean']:.4f} uV)")

# Overall result
all_pass = (
    abs(baseline_mean) < 0.1 and
    spike_flatness < 30 and
    abs(metadata['mean']) < 1.0
)

if all_pass:
    print("\n[SUCCESS] All tests passed! Order of operations fix verified.")
    print("\nFixed Issues:")
    print("  1. Baseline anchored at 0.0 (not floating at 0.8)")
    print("  2. Spikes remain sharp needles (not flat blocks)")
    print("  3. 1.0sigma scaling (not 1.5sigma) prevents baseline float")
else:
    print("\n[FAIL] Some tests failed - order of operations bug may still exist!")

print("=" * 80)
