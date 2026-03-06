"""
Verification script to confirm the fix is working.
Tests that noise_level=0.10 produces cleaner visualization than 0.30.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal

print("=" * 80)
print("VERIFICATION: Noise Level Fix")
print("=" * 80)

# Test with OLD default (0.30)
print("\n[TEST 1] OLD Default: noise_level=0.30")
raw_chunk_old = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.30,
    drift_severity=0.40,
    spike_rate=20.0,
    seed=42
)

config_old = {
    'poly_order': 1,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 0
}

cleaned_old, latency_old, metadata_old, _ = process_signal(raw_chunk_old, config_old, buffer=None)

# Check oscillations
diff_old = np.diff(cleaned_old)
osc_old = np.sum(np.abs(diff_old) > 0.1)
osc_pct_old = (osc_old / len(cleaned_old)) * 100

print(f"  Cleaned signal std: {np.std(cleaned_old):.6f}")
print(f"  Oscillations (|diff| > 0.1): {osc_pct_old:.1f}%")
print(f"  Signal quality: {'[X] TOO NOISY for demos' if osc_pct_old > 70 else '[OK] Clean enough'}")

# Test with NEW default (0.10) - but keep drift at 0.40 for fair comparison
print("\n[TEST 2] NEW Default: noise_level=0.10 (drift=0.40)")
raw_chunk_new = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.10,
    drift_severity=0.40,
    spike_rate=20.0,
    seed=42
)

config_new = {
    'poly_order': 1,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 0
}

cleaned_new, latency_new, metadata_new, _ = process_signal(raw_chunk_new, config_new, buffer=None)

# Check oscillations
diff_new = np.diff(cleaned_new)
osc_new = np.sum(np.abs(diff_new) > 0.1)
osc_pct_new = (osc_new / len(cleaned_new)) * 100

print(f"  Cleaned signal std: {np.std(cleaned_new):.6f}")
print(f"  Oscillations (|diff| > 0.1): {osc_pct_new:.1f}%")
print(f"  Signal quality: {'[OK] CLEAN for demos' if osc_pct_new < 70 else '[X] Still too noisy'}")

# Test with DEMO preset (noise=0.10, drift=0.20)
print("\n[TEST 3] DEMO Preset: noise_level=0.10, drift=0.20")
raw_chunk_demo = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.10,
    drift_severity=0.20,
    spike_rate=20.0,
    seed=42
)

config_demo = {
    'poly_order': 1,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 0
}

cleaned_demo, latency_demo, metadata_demo, _ = process_signal(raw_chunk_demo, config_demo, buffer=None)

# Check oscillations
diff_demo = np.diff(cleaned_demo)
osc_demo = np.sum(np.abs(diff_demo) > 0.1)
osc_pct_demo = (osc_demo / len(cleaned_demo)) * 100

print(f"  Cleaned signal std: {np.std(cleaned_demo):.6f}")
print(f"  Oscillations (|diff| > 0.1): {osc_pct_demo:.1f}%")
print(f"  Signal quality: {'[OK] CLEAN for demos' if osc_pct_demo < 70 else '[X] Still too noisy'}")

# Compare improvement
print("\n" + "=" * 80)
print("COMPARISON & RESULTS")
print("=" * 80)
print(f"\nOscillation Levels:")
print(f"  Test 1 (OLD): noise=0.30, drift=0.40 -> {osc_pct_old:.1f}% oscillations")
print(f"  Test 2 (NEW): noise=0.10, drift=0.40 -> {osc_pct_new:.1f}% oscillations")
print(f"  Test 3 (DEMO): noise=0.10, drift=0.20 -> {osc_pct_demo:.1f}% oscillations")

print(f"\nImprovement from changing noise_level only:")
print(f"  {osc_pct_old:.1f}% -> {osc_pct_new:.1f}% = {(osc_pct_old - osc_pct_new):.1f} pp reduction")

print(f"\nImprovement with Demo preset (noise AND drift reduced):")
print(f"  {osc_pct_old:.1f}% -> {osc_pct_demo:.1f}% = {(osc_pct_old - osc_pct_demo):.1f} pp reduction")

if osc_pct_demo < 70:
    print("\n[SUCCESS] FIX SUCCESSFUL!")
    print("   - OLD default (noise=0.30, drift=0.40) was TOO NOISY for demos")
    print(f"   - DEMO preset (noise=0.10, drift=0.20) provides CLEAN visualization ({osc_pct_demo:.1f}% oscillations)")
    print("   - Dashboard is now ready for video recording!")
    print("\n[INFO] Dashboard available at: http://localhost:8511")
    print("   Use preset buttons in sidebar:")
    print("   - [Demo] Clean visualization (RECOMMENDED for videos)")
    print("   - [Real] Realistic biological conditions (for testing)")
    print("   - [Stress] Worst-case testing (maximum noise/drift)")
elif osc_pct_new < osc_pct_old:
    print("\n[PARTIAL SUCCESS]")
    print(f"   - Reducing noise_level helps (89.1% -> {osc_pct_new:.1f}%)")
    print("   - But Demo preset (with lower drift too) provides best results")
else:
    print("\n[WARNING] Results need review")

print("=" * 80)
