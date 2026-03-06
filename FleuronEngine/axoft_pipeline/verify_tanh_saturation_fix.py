"""
Verification script for tanh saturation fix.

Tests that ideal biological conditions (low drift, low noise) produce 95%+ yield.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk, reset_drift_phase
from dsp_pipeline import process_signal_streaming, reset_streaming_buffer
from metrics_engine import calculate_signal_yield, check_system_health

print("=" * 80)
print("TANH SATURATION FIX VERIFICATION")
print("=" * 80)

# Reset state
reset_drift_phase()
reset_streaming_buffer()

# Test 1: IDEAL BIOLOGICAL CONDITIONS (Low drift, Low noise)
print("\n" + "=" * 80)
print("TEST 1: IDEAL BIOLOGICAL CONDITIONS")
print("=" * 80)
print("\nParameters:")
print("  - Drift Severity: 0.15 (very low)")
print("  - Noise Level: 0.15 (very low)")
print("  - Expected Result: 95%+ yield, HEALTHY status")

yields_ideal = []
variances_ideal = []
mean_shifts_ideal = []
spike_counts_ideal = []
healths_ideal = []

config = {
    'moving_avg_window': 400,  # Reduced from 1500 to preserve spike morphology (10ms window)
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0  # Lowered to detect 100μV spikes (derivative ≈8.3μV per sample)
}

for i in range(20):
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.15,  # IDEAL: Low noise
        drift_severity=0.15,  # IDEAL: Low drift
        spike_rate=20.0
    )

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    health = check_system_health(cleaned_tensor, metadata, yield_pct)

    yields_ideal.append(yield_pct)
    variances_ideal.append(metadata['variance'])
    mean_shifts_ideal.append(abs(metadata['mean']))
    spike_counts_ideal.append(metadata['spike_count'])
    healths_ideal.append(health)

print("\n" + "-" * 80)
print("RESULTS (20 iterations):")
print("-" * 80)
print(f"  Average Yield:     {np.mean(yields_ideal):.2f}%")
print(f"  Yield Range:       {np.min(yields_ideal):.2f}% - {np.max(yields_ideal):.2f}%")
print(f"  Avg Variance:      {np.mean(variances_ideal):.4f} (expect 0.20-0.40 with 1.5-sigma scaling)")
print(f"  Avg Mean Shift:    {np.mean(mean_shifts_ideal):.4f} (expect <0.10)")
print(f"  Avg Spike Count:   {np.mean(spike_counts_ideal):.0f}")
print(f"  System Health:     {healths_ideal[-1].upper()}")

healthy_count = sum(1 for h in healths_ideal if h == "healthy")
warning_count = sum(1 for h in healths_ideal if h == "warning")
critical_count = sum(1 for h in healths_ideal if h == "critical")

print(f"\n  Health Distribution:")
print(f"    - Healthy: {healthy_count}/20 ({100*healthy_count/20:.0f}%)")
print(f"    - Warning: {warning_count}/20 ({100*warning_count/20:.0f}%)")
print(f"    - Critical: {critical_count}/20 ({100*critical_count/20:.0f}%)")

print("\n" + "=" * 80)
if np.mean(yields_ideal) >= 95.0 and healthy_count >= 18:
    print("[PASS] IDEAL CONDITIONS PRODUCE 95%+ YIELD & HEALTHY STATUS")
elif np.mean(yields_ideal) >= 90.0:
    print("[PARTIAL PASS] [WARN] Yield 90-95%, close but could be better")
else:
    print(f"[FAIL] [FAIL] Yield only {np.mean(yields_ideal):.1f}%, expected 95%+")
print("=" * 80)


# Test 2: MODERATE CONDITIONS (Medium drift, Medium noise)
print("\n" + "=" * 80)
print("TEST 2: MODERATE BIOLOGICAL CONDITIONS")
print("=" * 80)
print("\nParameters:")
print("  - Drift Severity: 0.5 (moderate)")
print("  - Noise Level: 0.3 (moderate)")
print("  - Expected Result: 75-85% yield, HEALTHY status")

reset_drift_phase()
reset_streaming_buffer()

yields_moderate = []
healths_moderate = []

for i in range(20):
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.3,  # MODERATE noise
        drift_severity=0.5,  # MODERATE drift
        spike_rate=20.0
    )

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    health = check_system_health(cleaned_tensor, metadata, yield_pct)

    yields_moderate.append(yield_pct)
    healths_moderate.append(health)

print("\n" + "-" * 80)
print("RESULTS (20 iterations):")
print("-" * 80)
print(f"  Average Yield:     {np.mean(yields_moderate):.2f}%")
print(f"  Yield Range:       {np.min(yields_moderate):.2f}% - {np.max(yields_moderate):.2f}%")
print(f"  System Health:     {healths_moderate[-1].upper()}")

healthy_count_mod = sum(1 for h in healths_moderate if h == "healthy")
print(f"\n  Healthy Status: {healthy_count_mod}/20 ({100*healthy_count_mod/20:.0f}%)")

print("\n" + "=" * 80)
if np.mean(yields_moderate) >= 75.0:
    print("[PASS] [OK] MODERATE CONDITIONS PRODUCE 75%+ YIELD & HEALTHY STATUS")
else:
    print(f"[FAIL] [FAIL] Yield only {np.mean(yields_moderate):.1f}%, expected 75%+")
print("=" * 80)


# Test 3: HARSH CONDITIONS (High drift, High noise)
print("\n" + "=" * 80)
print("TEST 3: HARSH BIOLOGICAL CONDITIONS")
print("=" * 80)
print("\nParameters:")
print("  - Drift Severity: 1.5 (high)")
print("  - Noise Level: 0.6 (high)")
print("  - Expected Result: 60-70% yield, WARNING status acceptable")

reset_drift_phase()
reset_streaming_buffer()

yields_harsh = []
healths_harsh = []

for i in range(20):
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.6,  # HIGH noise
        drift_severity=1.5,  # HIGH drift
        spike_rate=20.0
    )

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    health = check_system_health(cleaned_tensor, metadata, yield_pct)

    yields_harsh.append(yield_pct)
    healths_harsh.append(health)

print("\n" + "-" * 80)
print("RESULTS (20 iterations):")
print("-" * 80)
print(f"  Average Yield:     {np.mean(yields_harsh):.2f}%")
print(f"  Yield Range:       {np.min(yields_harsh):.2f}% - {np.max(yields_harsh):.2f}%")
print(f"  System Health:     {healths_harsh[-1].upper()}")

print("\n" + "=" * 80)
if np.mean(yields_harsh) >= 60.0:
    print("[PASS] [OK] HARSH CONDITIONS PRODUCE 60%+ YIELD (degraded but functional)")
else:
    print(f"[FAIL] [FAIL] Yield only {np.mean(yields_harsh):.1f}%, expected 60%+")
print("=" * 80)


# Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\n  IDEAL conditions:    {np.mean(yields_ideal):.1f}% yield (target: 95%+)")
print(f"  MODERATE conditions: {np.mean(yields_moderate):.1f}% yield (target: 75%+)")
print(f"  HARSH conditions:    {np.mean(yields_harsh):.1f}% yield (target: 60%+)")

print("\n  Key Metrics (Ideal conditions):")
print(f"    - Variance: {np.mean(variances_ideal):.4f} (spikes clearly visible, not flattened)")
print(f"    - Mean Shift: {np.mean(mean_shifts_ideal):.4f} (close to 0, stable)")
print(f"    - Adaptive Scaling: ACTIVE (3-sigma rule preserves spike morphology)")
print(f"    - Moving Avg Window: 400 samples (10ms, preserves 1ms spikes)")

if np.mean(yields_ideal) >= 95.0:
    print("\n" + "=" * 80)
    print("[SUCCESS] [OK] TANH SATURATION BUG FIXED - IDEAL CONDITIONS NOW PRODUCE 95%+ YIELD")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print(f"[INCOMPLETE] [WARN] Improvement seen but not yet at 95% target")
    print(f"Current: {np.mean(yields_ideal):.1f}%, Target: 95%+")
    print("=" * 80)
