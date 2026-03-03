"""
Quick verification that dashboard metrics will be realistic after fix
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal_streaming, reset_streaming_buffer
from metrics_engine import calculate_signal_yield, calculate_active_channels, check_system_health

print("=" * 80)
print("DASHBOARD METRIC VERIFICATION (After Fix)")
print("=" * 80)

# Reset state
reset_streaming_buffer()

# Generate a realistic chunk with moderate drift and noise
print("\nGenerating synthetic chunk with realistic parameters...")
print("  - Drift Severity: 1.0 (moderate)")
print("  - Noise Level: 0.3 (moderate)")
print("  - Spike Rate: 20 Hz")

raw_chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.3,
    drift_severity=1.0,
    spike_rate=20.0
)

# Process through pipeline
config = {
    'moving_avg_window': 500,
    'tanh_alpha': 1.0,
    'spike_threshold': 20.0
}

cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)

# Calculate metrics
yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
active_channels = calculate_active_channels(yield_pct, total_channels=10000)
health_status = check_system_health(cleaned_tensor, metadata, yield_pct)

print("\n" + "=" * 80)
print("EXPECTED DASHBOARD METRICS (After Fix)")
print("=" * 80)

print(f"\n  Live Signal Yield:  {yield_pct:.1f}%")
print(f"  Active Channels:    {active_channels:,} / 10,000")
print(f"  System Health:      {health_status.upper()}")
print(f"  Pipeline Latency:   {latency_ms:.2f} ms")

print("\n" + "-" * 80)
print("DETAILED BREAKDOWN:")
print("-" * 80)

print(f"\n  Raw Metrics:")
print(f"    - Spike/Crossing Count: {metadata['spike_count']}")
print(f"    - Signal Variance:      {metadata['variance']:.4f}")
print(f"    - Mean Shift:           {metadata['mean']:.4f}")

print(f"\n  Scoring Components (estimated):")
variance = metadata['variance']
spike_count = metadata['spike_count']
mean_shift = abs(metadata['mean'])

# Estimate variance score
optimal_variance = 0.3
variance_score_est = 100.0 * np.exp(-((variance - optimal_variance) ** 2) / (2 * 0.2 ** 2))
variance_score_est = max(variance_score_est, 60.0)

# Estimate spike score
optimal_crossing_count = 200.0
if spike_count < 50:
    spike_score_est = 40.0
elif spike_count > 800:
    spike_score_est = 50.0
else:
    deviation = abs(spike_count - optimal_crossing_count) / optimal_crossing_count
    spike_score_est = 100.0 * np.exp(-2.0 * (deviation ** 2))
    spike_score_est = max(spike_score_est, 70.0)

# Estimate stability score
max_acceptable_shift = 0.3
if mean_shift > max_acceptable_shift:
    stability_score_est = 40.0
else:
    stability_score_est = 100.0 * (1.0 - mean_shift / max_acceptable_shift)
    stability_score_est = max(stability_score_est, 60.0)

print(f"    - Variance Score:   {variance_score_est:.1f}% (40% weight)")
print(f"    - Spike Score:      {spike_score_est:.1f}% (30% weight)")
print(f"    - Stability Score:  {stability_score_est:.1f}% (30% weight)")

composite_est = 0.4 * variance_score_est + 0.3 * spike_score_est + 0.3 * stability_score_est
print(f"\n  Composite Yield:    {composite_est:.1f}%")

print("\n" + "=" * 80)
if yield_pct >= 70 and health_status == "healthy":
    print("STATUS: [OK] DASHBOARD WILL SHOW HEALTHY METRICS")
elif yield_pct >= 50:
    print("STATUS: [WARNING]  DASHBOARD WILL SHOW WARNING (Acceptable)")
else:
    print("STATUS: [FAIL] DASHBOARD STILL SHOWING CRITICAL")

print("=" * 80)

# Run 5 more iterations to get average
print("\nRunning 5 more iterations to verify stability...")
yields = [yield_pct]
healths = [health_status]

for i in range(5):
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.3,
        drift_severity=1.0,
        spike_rate=20.0
    )
    cleaned_tensor, _, metadata = process_signal_streaming(raw_chunk, config)
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    health = check_system_health(cleaned_tensor, metadata, yield_pct)
    yields.append(yield_pct)
    healths.append(health)
    print(f"  Iteration {i+2}: Yield={yield_pct:.1f}%, Health={health}")

avg_yield = np.mean(yields)
healthy_count = sum(1 for h in healths if h == "healthy")
warning_count = sum(1 for h in healths if h == "warning")
critical_count = sum(1 for h in healths if h == "critical")

print(f"\n  Average Yield: {avg_yield:.1f}%")
print(f"  Health Distribution: {healthy_count} healthy, {warning_count} warning, {critical_count} critical")

print("\n" + "=" * 80)
if avg_yield >= 70:
    print("[OK] FIX SUCCESSFUL - Dashboard will show healthy/good metrics")
elif avg_yield >= 50:
    print("[WARNING]  PARTIAL FIX - Dashboard shows marginal but acceptable")
else:
    print("[FAIL] FIX INSUFFICIENT - Need further tuning")
print("=" * 80)
