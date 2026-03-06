"""
Explain why lower alpha looks "crazier" visually even with similar oscillation counts.
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal
import matplotlib.pyplot as plt

print("=" * 80)
print("WHY LOWER ALPHA LOOKS 'CRAZY' - Visual Separation Analysis")
print("=" * 80)

# Test Real mode with two alpha values
noise_level = 0.30
drift_severity = 0.40

raw_chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=noise_level,
    drift_severity=drift_severity,
    spike_rate=20.0,
    seed=42
)

alphas_to_test = [0.40, 1.00]

for alpha in alphas_to_test:
    print(f"\n{'=' * 80}")
    print(f"ALPHA = {alpha:.2f}")
    print(f"{'=' * 80}")

    config = {
        'poly_order': 1,
        'tanh_alpha': alpha,
        'spike_threshold': 5.0,
        'smoothing_window': 0
    }

    cleaned, latency, metadata, _ = process_signal(raw_chunk, config, buffer=None)

    # Analyze distribution of values
    hist, bins = np.histogram(cleaned, bins=20, range=(-1.0, 1.0))

    # Find where spikes live (top 5% of absolute values)
    spike_threshold_val = np.percentile(np.abs(cleaned), 95)
    spike_mask = np.abs(cleaned) > spike_threshold_val
    noise_mask = ~spike_mask

    spike_values = cleaned[spike_mask]
    noise_values = cleaned[noise_mask]

    print(f"\nValue Distribution:")
    print(f"  Spike region (top 5%): mean={np.mean(spike_values):.3f}, std={np.std(spike_values):.3f}")
    print(f"  Noise region (other 95%): mean={np.mean(noise_values):.3f}, std={np.std(noise_values):.3f}")
    print(f"  Separation (spike_mean - noise_mean): {abs(np.mean(spike_values)) - abs(np.mean(noise_values)):.3f}")

    # Calculate visual "messiness" - how much do values spread across the plot
    # We want spikes to be near ±1.0 and noise to be near 0.0
    spike_concentration = np.mean(np.abs(spike_values))  # How close are spikes to ±1.0?
    noise_concentration = np.std(noise_values)  # How spread out is noise?

    visual_separation = spike_concentration / (noise_concentration + 0.01)

    print(f"\nVisual Clarity Metrics:")
    print(f"  Spike concentration (closeness to ±1.0): {spike_concentration:.3f}")
    print(f"  Noise spread (std): {noise_concentration:.3f}")
    print(f"  Visual separation ratio: {visual_separation:.2f}x")

    # What percentage of the plot is "filled" by noise vs spikes?
    noise_fill = len(noise_values) / len(cleaned) * 100
    spike_fill = len(spike_values) / len(cleaned) * 100

    print(f"\nPlot 'Real Estate' Usage:")
    print(f"  Noise occupies: {noise_fill:.1f}% of samples")
    print(f"  Spikes occupy: {spike_fill:.1f}% of samples")

print("\n" + "=" * 80)
print("CONCLUSION: Why Alpha=0.40 Looks 'Crazy'")
print("=" * 80)

print("""
With LOWER alpha (0.40):
  - Tanh acts more LINEAR (less compression)
  - Spikes DON'T reach ±1.0 (maybe only ±0.8)
  - Noise is NOT compressed much either
  - Result: Noise and spikes occupy SIMILAR amplitude ranges
  - Visual effect: Hard to distinguish what's signal vs what's noise
  - Plot looks "MESSY" or "CRAZY" - everything is at similar scale

With HIGHER alpha (1.00):
  - Tanh acts more SATURATED (strong compression)
  - Spikes GET PUSHED to ±1.0 (full range utilization)
  - Noise gets COMPRESSED to smaller middle range
  - Result: Clear visual separation between spikes (±1.0) and noise (~0)
  - Visual effect: Spikes "POP OUT" from noisy background
  - Plot looks CLEANER - clear hierarchy of amplitudes

This is NOT a bug - it's the fundamental tradeoff of the alpha parameter:
  - Lower alpha = More linear, less saturation, worse visual separation
  - Higher alpha = More saturated, better visual separation, but risks clipping

For demos with Real/Stress modes:
  - Keep alpha >= 1.0 for good visual separation
  - Lower alpha only makes sense in very clean Demo mode
""")

print("=" * 80)
