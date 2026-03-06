"""
Diagnose why lower tanh_alpha makes oscillations MORE visible.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal

print("=" * 80)
print("DIAGNOSTIC: Tanh Alpha Effect on Oscillation Visibility")
print("=" * 80)

# Test different scenarios with different alpha values
scenarios = [
    ("Demo Mode", 0.10, 0.20),
    ("Real Mode", 0.30, 0.40),
    ("Stress Mode", 0.50, 1.00)
]

alpha_values = [0.40, 0.70, 1.00, 1.50, 2.00]

for scenario_name, noise_level, drift_severity in scenarios:
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {scenario_name} (noise={noise_level}, drift={drift_severity})")
    print(f"{'=' * 80}")

    # Generate chunk for this scenario
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=noise_level,
        drift_severity=drift_severity,
        spike_rate=20.0,
        seed=42
    )

    print(f"\nRaw signal stats:")
    print(f"  Mean: {np.mean(raw_chunk):.2f} uV")
    print(f"  Std: {np.std(raw_chunk):.2f} uV")
    print(f"  Range: [{np.min(raw_chunk):.2f}, {np.max(raw_chunk):.2f}]")

    for alpha in alpha_values:
        config = {
            'poly_order': 1,
            'tanh_alpha': alpha,
            'spike_threshold': 5.0,
            'smoothing_window': 0
        }

        cleaned, latency, metadata, _ = process_signal(raw_chunk, config, buffer=None)

        # Measure oscillations
        diff = np.diff(cleaned)
        oscillations = np.sum(np.abs(diff) > 0.1)
        osc_pct = (oscillations / len(cleaned)) * 100

        # Measure utilization of [-1, 1] range
        range_used = np.max(cleaned) - np.min(cleaned)
        range_util = (range_used / 2.0) * 100  # As percentage of [-1, 1] = 2.0 total

        # Measure noise relative to signal
        noise_to_signal = np.std(cleaned) / (np.max(np.abs(cleaned)) + 1e-6)

        print(f"\n  Alpha={alpha:.2f}:")
        print(f"    Cleaned range: [{np.min(cleaned):.3f}, {np.max(cleaned):.3f}]")
        print(f"    Range utilization: {range_util:.1f}% of [-1, 1]")
        print(f"    Std: {np.std(cleaned):.3f}")
        print(f"    Noise/Signal ratio: {noise_to_signal:.3f}")
        print(f"    Oscillations: {osc_pct:.1f}%")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("\nLOWER alpha (e.g., 0.40) makes oscillations MORE visible because:")
print("  1. Tanh is more LINEAR (less compression)")
print("  2. Output range is UNDERUTILIZED (spikes don't reach ±1.0)")
print("  3. Noise fills a LARGER percentage of the used range")
print("  4. Visual effect: Noise looks BIGGER relative to spikes")
print("\nHIGHER alpha (e.g., 1.00+) makes oscillations LESS visible because:")
print("  1. Tanh is more SATURATED (stronger compression)")
print("  2. Output range is BETTER UTILIZED (spikes hit near ±1.0)")
print("  3. Noise fills a SMALLER percentage of the used range")
print("  4. Visual effect: Noise looks SMALLER relative to spikes")
print("=" * 80)
