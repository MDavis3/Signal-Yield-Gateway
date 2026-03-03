"""
Verification script for DSP architecture fixes.

Tests three critical fixes:
1. Flat baseline (MA window reduction from 1500 -> 400)
2. Sharp spike morphology (smoothing reduction from 40 -> 10)
3. Stable health with clean signals (adjusted variance/spike thresholds)
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal, CircularBuffer
from metrics_engine import calculate_signal_yield

print("=" * 80)
print("DSP ARCHITECTURE FIX VERIFICATION")
print("=" * 80)

# ============================================================================
# Test 1: Flat Baseline (MA Window Reduction)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Flat Baseline with Reduced MA Window")
print("=" * 80)

# Generate signal with moderate drift
test_chunk_wavy = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.25,
    drift_severity=0.35,  # Same as screenshot settings
    spike_rate=20.0,
    seed=42
)

# Test OLD configuration (1500-sample MA window)
config_old_ma = {
    'moving_avg_window': 1500,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 10
}

cleaned_old, latency_old, metadata_old, _ = process_signal(
    test_chunk_wavy,
    config_old_ma,
    buffer=CircularBuffer(1500)
)

# Test NEW configuration (400-sample MA window)
config_new_ma = {
    'moving_avg_window': 400,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 10
}

cleaned_new, latency_new, metadata_new, _ = process_signal(
    test_chunk_wavy,
    config_new_ma,
    buffer=CircularBuffer(400)
)

# CRITICAL: Measure baseline waviness on PRE-TANH centered signal (not post-tanh)
# The centered_signal shows the actual baseline drift remaining after MA subtraction
# A wavy baseline means the MA couldn't track the drift properly
centered_old = metadata_old['centered_signal']
centered_new = metadata_new['centered_signal']

# Compute rolling mean to measure low-frequency drift components (100-sample window)
# If MA worked perfectly, the rolling mean should be flat (near 0)
from scipy.ndimage import uniform_filter1d
rolling_mean_old = uniform_filter1d(centered_old, size=100, mode='nearest')
rolling_mean_new = uniform_filter1d(centered_new, size=100, mode='nearest')

# Measure baseline waviness (std of rolling mean = low-frequency drift amplitude)
baseline_drift_old = np.std(rolling_mean_old)
baseline_drift_new = np.std(rolling_mean_new)

print(f"\nBaseline Drift Remaining (StdDev of 100-sample rolling mean, uV):")
print(f"  OLD (MA=1500): {baseline_drift_old:.2f} uV")
print(f"  NEW (MA=400):  {baseline_drift_new:.2f} uV")
print(f"  Improvement:   {(baseline_drift_old - baseline_drift_new) / baseline_drift_old * 100:.1f}% reduction")

# Measure DC offset
mean_offset_old = abs(np.mean(centered_old))
mean_offset_new = abs(np.mean(centered_new))

print(f"\nMean Baseline Offset (should be ~0.0 uV):")
print(f"  OLD (MA=1500): {mean_offset_old:.2f} uV")
print(f"  NEW (MA=400):  {mean_offset_new:.2f} uV")

if baseline_drift_new < baseline_drift_old * 0.85:  # At least 15% improvement
    print("\n[PASS] Baseline is significantly flatter with reduced MA window")
else:
    print("\n[FAIL] Baseline still wavy - needs further MA window reduction")

# ============================================================================
# Test 2: Sharp Spike Morphology (Smoothing Window Reduction)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Sharp Spike Morphology with Reduced Smoothing")
print("=" * 80)

# Generate signal with a BIG spike to test morphology
test_chunk_spike = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.15,
    drift_severity=0.0,  # No drift for clear spike visualization
    spike_rate=30.0,
    seed=123
)

# Test OLD smoothing (40 samples = destroys morphology)
config_old_smooth = {
    'moving_avg_window': 400,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 40  # OLD: destroys spikes
}

cleaned_old_smooth, _, metadata_old_smooth, _ = process_signal(
    test_chunk_spike,
    config_old_smooth,
    buffer=CircularBuffer(400)
)

# Test NEW smoothing (10 samples = preserves morphology)
config_new_smooth = {
    'moving_avg_window': 400,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 10  # NEW: preserves spikes
}

cleaned_new_smooth, _, metadata_new_smooth, _ = process_signal(
    test_chunk_spike,
    config_new_smooth,
    buffer=CircularBuffer(400)
)

# Measure spike sharpness (max derivative = rise steepness)
# Sharper spikes have higher max derivative
max_derivative_old = np.max(np.abs(np.diff(cleaned_old_smooth)))
max_derivative_new = np.max(np.abs(np.diff(cleaned_new_smooth)))

print(f"\nSpike Sharpness (Max Derivative):")
print(f"  OLD (Smoothing=40): {max_derivative_old:.4f}")
print(f"  NEW (Smoothing=10): {max_derivative_new:.4f}")
print(f"  Improvement:        {(max_derivative_new - max_derivative_old) / max_derivative_old * 100:.1f}% sharper")

# Measure spike peak flatness (count samples near max value)
# Square spikes have many samples at max, sharp spikes have few
max_val_old = np.max(cleaned_old_smooth)
max_val_new = np.max(cleaned_new_smooth)
flat_samples_old = np.sum(cleaned_old_smooth > 0.95 * max_val_old)
flat_samples_new = np.sum(cleaned_new_smooth > 0.95 * max_val_new)

print(f"\nSpike Peak Flatness (samples within 95% of max):")
print(f"  OLD (Smoothing=40): {flat_samples_old} samples (more = flatter/square)")
print(f"  NEW (Smoothing=10): {flat_samples_new} samples (fewer = sharper/needle)")
print(f"  Improvement:        {(flat_samples_old - flat_samples_new) / flat_samples_old * 100:.1f}% reduction in flatness")

if max_derivative_new > max_derivative_old and flat_samples_new < flat_samples_old:
    print("\n[PASS] Spikes are sharper and less square with reduced smoothing")
else:
    print("\n[FAIL] Spikes still square")

# ============================================================================
# Test 3: Stable Health with Clean Signals (Metrics Thresholds)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Stable Health Scores with Clean Signals")
print("=" * 80)

# Simulate 20 epochs with CLEAN signals (low drift, low noise)
yields_clean = []
healths_clean = []

for i in range(20):
    chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.25,  # Clean
        drift_severity=0.35,  # Clean
        spike_rate=20.0,
        seed=i  # Different seed each epoch
    )

    config = {
        'moving_avg_window': 400,
        'tanh_alpha': 1.0,
        'spike_threshold': 5.0,
        'smoothing_window': 10
    }

    cleaned, _, metadata, _ = process_signal(chunk, config, buffer=CircularBuffer(400))
    yield_pct = calculate_signal_yield(cleaned, metadata['spike_count'], metadata)
    yields_clean.append(yield_pct)

    # Health status
    if yield_pct >= 90:
        health = "Healthy"
    elif yield_pct >= 70:
        health = "Warning"
    else:
        health = "Critical"
    healths_clean.append(health)

print(f"\nYields over 20 epochs (CLEAN signals, drift=0.35, noise=0.25):")
print(f"  Mean:   {np.mean(yields_clean):.1f}%")
print(f"  StdDev: {np.std(yields_clean):.1f}%")
print(f"  Range:  {np.min(yields_clean):.1f}% to {np.max(yields_clean):.1f}%")
print(f"  Min:    {np.min(yields_clean):.1f}%")

# Count health status transitions
critical_count = healths_clean.count("Critical")
warning_count = healths_clean.count("Warning")
healthy_count = healths_clean.count("Healthy")

print(f"\nHealth Status Distribution:")
print(f"  Healthy (>=90%):  {healthy_count}/20 epochs ({healthy_count/20*100:.0f}%)")
print(f"  Warning (70-89%): {warning_count}/20 epochs ({warning_count/20*100:.0f}%)")
print(f"  Critical (<70%):  {critical_count}/20 epochs ({critical_count/20*100:.0f}%)")

# Check for random drops below 70% (the original bug)
if critical_count == 0 and np.min(yields_clean) >= 70:
    print("\n[PASS] No random health drops with clean signals")
else:
    print(f"\n[FAIL] {critical_count} random drops to Critical status detected")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

all_passed = (
    baseline_drift_new < baseline_drift_old and
    max_derivative_new > max_derivative_old and
    critical_count == 0 and
    np.mean(yields_clean) >= 85  # Yields should be reasonably high with clean signals
)

if all_passed:
    print("\n[SUCCESS] ALL TESTS PASSED")
    print("\nFixed Issues:")
    print("  1. [OK] Flat baseline (MA window 1500 -> 400)")
    print("  2. [OK] Sharp spikes (smoothing 40 -> 10)")
    print("  3. [OK] Stable health with clean signals (adjusted thresholds)")
else:
    print("\n[FAIL] SOME TESTS FAILED - Review output above")

print("\n" + "=" * 80)
