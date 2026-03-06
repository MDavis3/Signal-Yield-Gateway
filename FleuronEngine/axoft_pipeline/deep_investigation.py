"""
Deep investigation: What's really causing the "crazy" appearance?
Check for any hidden interactions or issues in the pipeline.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import polyfit_detrend, tanh_normalize

print("=" * 80)
print("DEEP INVESTIGATION: Pipeline Step-by-Step Analysis")
print("=" * 80)

# Test Real mode with different alphas
scenarios = [
    ("Real Mode, Alpha=0.40", 0.30, 0.40, 0.40),
    ("Real Mode, Alpha=1.00", 0.30, 0.40, 1.00),
    ("Demo Mode, Alpha=1.00", 0.10, 0.20, 1.00),
]

for scenario_name, noise_level, drift_severity, alpha in scenarios:
    print(f"\n{'=' * 80}")
    print(f"{scenario_name}")
    print(f"{'=' * 80}")

    # Generate raw signal
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=noise_level,
        drift_severity=drift_severity,
        spike_rate=20.0,
        seed=42
    )

    print(f"\n[STEP 1] Raw Signal:")
    print(f"  Mean: {np.mean(raw_chunk):.2f} uV")
    print(f"  Std: {np.std(raw_chunk):.2f} uV")
    print(f"  Range: [{np.min(raw_chunk):.2f}, {np.max(raw_chunk):.2f}]")

    # Apply polyfit detrending
    centered, _ = polyfit_detrend(raw_chunk, poly_order=1)

    print(f"\n[STEP 2] After Polyfit Detrending:")
    print(f"  Mean: {np.mean(centered):.6f} uV")
    print(f"  Std: {np.std(centered):.2f} uV")
    print(f"  Range: [{np.min(centered):.2f}, {np.max(centered):.2f}]")

    # Check if centered signal has issues
    centered_diff = np.diff(centered)
    centered_large_changes = np.sum(np.abs(centered_diff) > 20)
    print(f"  Large changes (>20uV): {centered_large_changes} samples")

    # Apply tanh normalization - THIS IS WHERE WE NEED TO LOOK CAREFULLY
    # Let's trace through tanh_normalize step by step

    # Step 1: Remove mean
    signal_mean = np.mean(centered)
    recentered = centered - signal_mean

    print(f"\n[STEP 3a] Tanh: After re-centering:")
    print(f"  Mean: {np.mean(recentered):.6f} uV")
    print(f"  Std: {np.std(recentered):.2f} uV")

    # Step 2: Get std
    std_val = max(np.std(recentered), 1e-6)

    print(f"\n[STEP 3b] Tanh: Std calculation:")
    print(f"  Std: {std_val:.2f} uV")

    # Step 3: Normalize by std
    scaled = recentered / std_val

    print(f"\n[STEP 3c] Tanh: After std normalization:")
    print(f"  Mean: {np.mean(scaled):.6f}")
    print(f"  Std: {np.std(scaled):.6f}")
    print(f"  Range: [{np.min(scaled):.2f}, {np.max(scaled):.2f}]")
    print(f"  Values >3 sigma: {np.sum(np.abs(scaled) > 3)} samples")
    print(f"  Values >5 sigma: {np.sum(np.abs(scaled) > 5)} samples")

    # Step 4: Apply tanh
    cleaned = np.tanh(alpha * scaled)

    print(f"\n[STEP 3d] Tanh: Final output:")
    print(f"  Mean: {np.mean(cleaned):.6f}")
    print(f"  Std: {np.std(cleaned):.6f}")
    print(f"  Range: [{np.min(cleaned):.3f}, {np.max(cleaned):.3f}]")

    # Check oscillation pattern
    cleaned_diff = np.diff(cleaned)
    oscillations = np.sum(np.abs(cleaned_diff) > 0.1)
    osc_pct = (oscillations / len(cleaned)) * 100

    print(f"\n[ANALYSIS] Oscillation Metrics:")
    print(f"  Oscillations (|diff| > 0.1): {osc_pct:.1f}%")
    print(f"  Max derivative: {np.max(np.abs(cleaned_diff)):.3f}")
    print(f"  Mean abs derivative: {np.mean(np.abs(cleaned_diff)):.3f}")

    # Check if there are any unexpected patterns
    # Are there regions of extreme oscillation?
    window_size = 200
    for i in range(0, len(cleaned), window_size):
        window = cleaned[i:i+window_size]
        window_diff = np.diff(window)
        window_osc = np.sum(np.abs(window_diff) > 0.1)
        window_osc_pct = (window_osc / len(window)) * 100
        if window_osc_pct > 95:
            print(f"  WARNING: Extreme oscillation in window {i}-{i+window_size}: {window_osc_pct:.1f}%")

print("\n" + "=" * 80)
print("KEY QUESTION: Is the pipeline working correctly?")
print("=" * 80)
print("\nThings to check:")
print("1. Does polyfit_detrend leave residual drift that varies with noise/drift level?")
print("2. Does tanh_normalize std calculation behave differently with different signal characteristics?")
print("3. Are there any numerical instabilities or edge cases?")
print("4. Is the 1σ scaling appropriate for all scenarios?")
print("=" * 80)
