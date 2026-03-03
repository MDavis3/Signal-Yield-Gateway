"""Quick test to see which component is dragging down yields"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal, CircularBuffer

# Generate clean signal
chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.25,  # Clean
    drift_severity=0.35,  # Clean
    spike_rate=20.0,
    seed=42
)

config = {
    'moving_avg_window': 400,
    'tanh_alpha': 1.0,
    'spike_threshold': 5.0,
    'smoothing_window': 10
}

cleaned, _, metadata, _ = process_signal(chunk, config, buffer=CircularBuffer(400))

print("Signal Characteristics (CLEAN: drift=0.35, noise=0.25):")
print(f"  Variance: {metadata['variance']:.2f} uV^2")
print(f"  Spike count (crossings): {metadata['spike_count']}")
print(f"  Mean offset: {abs(metadata['mean']):.2f} uV")

# Now manually compute the component scores using the same logic as metrics_engine.py
variance = metadata['variance']
spike_count = metadata['spike_count']
mean_shift = abs(metadata['mean'])

# Component 1: Variance Score
optimal_variance = 120.0
min_variance = 30.0
max_variance = 2000.0

if variance < min_variance:
    variance_score = 0.0
elif variance > max_variance:
    variance_score = 30.0
else:
    variance_score = 100.0 * np.exp(-((variance - optimal_variance) ** 2) / (2 * 1200.0 ** 2))

print(f"\nComponent 1: Variance Score")
print(f"  Optimal variance: {optimal_variance} uV^2")
print(f"  Actual variance: {variance:.2f} uV^2")
print(f"  Deviation: {abs(variance - optimal_variance):.2f} uV^2")
print(f"  Score: {variance_score:.1f}/100")

# Component 2: Spike Score
optimal_crossing_count = 1100.0  # Updated to match metrics_engine.py fix
min_crossing_count = 50.0
max_crossing_count = 2000.0

if spike_count < min_crossing_count:
    spike_score = 40.0
elif spike_count > max_crossing_count:
    spike_score = 50.0
else:
    deviation = abs(spike_count - optimal_crossing_count) / optimal_crossing_count
    spike_score = 100.0 * np.exp(-0.5 * (deviation ** 2))

print(f"\nComponent 2: Spike Score")
print(f"  Optimal crossings: {optimal_crossing_count}")
print(f"  Actual crossings: {spike_count}")
print(f"  Deviation: {abs(spike_count - optimal_crossing_count):.0f} crossings ({abs(spike_count - optimal_crossing_count)/optimal_crossing_count*100:.1f}%)")
print(f"  Score: {spike_score:.1f}/100")

# Component 3: Stability Score
max_acceptable_shift = 80.0

if mean_shift > max_acceptable_shift:
    stability_score = 70.0
else:
    stability_score = 100.0 * np.exp(-1.0 * (mean_shift / max_acceptable_shift) ** 2)

print(f"\nComponent 3: Stability Score")
print(f"  Max acceptable shift: {max_acceptable_shift} uV")
print(f"  Actual mean shift: {mean_shift:.2f} uV")
print(f"  Score: {stability_score:.1f}/100")

# Composite Yield
yield_pct = 0.40 * variance_score + 0.30 * spike_score + 0.30 * stability_score

# Add biological jitter
biological_jitter = np.random.normal(0.0, 2.5)
yield_with_jitter = yield_pct + biological_jitter

print(f"\n{'='*60}")
print(f"Composite Yield Calculation:")
print(f"  40% * variance ({variance_score:.1f}) = {0.40 * variance_score:.1f}")
print(f"  30% * spike ({spike_score:.1f})    = {0.30 * spike_score:.1f}")
print(f"  30% * stability ({stability_score:.1f}) = {0.30 * stability_score:.1f}")
print(f"  Subtotal:                     {yield_pct:.1f}%")
print(f"  + Biological jitter:          {biological_jitter:+.1f}%")
print(f"  TOTAL YIELD:                  {yield_with_jitter:.1f}%")
print(f"{'='*60}")

if yield_pct < 85:
    print("\nDIAGNOSIS: Yields too low due to:")
    if variance_score < 90:
        print(f"  - Variance score ({variance_score:.1f}) penalizing clean signal")
    if spike_score < 90:
        print(f"  - Spike score ({spike_score:.1f}) penalizing clean signal (fewer noise crossings)")
    if stability_score < 90:
        print(f"  - Stability score ({stability_score:.1f}) penalizing drift")
