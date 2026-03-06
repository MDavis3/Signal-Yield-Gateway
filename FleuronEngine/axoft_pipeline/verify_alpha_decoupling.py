"""
Verification Script - Tanh Alpha Slider Decoupling Fix
========================================================

This script verifies that the Signal Yield % is now INDEPENDENT of the Tanh Alpha
aesthetic parameter, fixing the critical UI/Logic coupling bug.

Expected Result:
----------------
Yield % should remain constant (90-95%) across all alpha values (0.2 to 3.0),
proving that clinical metrics are decoupled from visualization aesthetics.

Author: Senior AI Software Engineer & BCI Data Architect
Date: 2026-03-02
"""

import numpy as np
from data_simulator import generate_synthetic_chunk
from dsp_pipeline import process_signal_streaming
from metrics_engine import calculate_signal_yield, check_system_health

# Reset any streaming state
from dsp_pipeline import reset_streaming_buffer
from data_simulator import reset_drift_phase
reset_streaming_buffer()
reset_drift_phase()

print("=" * 80)
print("VERIFICATION: Tanh Alpha Slider Decoupling Fix")
print("=" * 80)
print()
print("Testing Signal Yield % across different Tanh Alpha values...")
print("Expected: Yield should be CONSTANT regardless of alpha (aesthetic parameter)")
print()

# Generate a single test chunk with IDEAL conditions
raw_chunk = generate_synthetic_chunk(
    duration_ms=50.0,
    sample_rate=40000,
    noise_level=0.15,      # Ideal biological conditions
    drift_severity=0.15,   # Ideal biological conditions
    spike_rate=20.0
)

# Test with multiple alpha values
alpha_values = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
results = []

print(f"{'Tanh Alpha':<15} {'Yield %':<12} {'Health Status':<15} {'POST-tanh Var':<15} {'PRE-tanh Var (uV^2)':<20}")
print("-" * 95)

for alpha in alpha_values:
    # Reset buffer for clean test
    reset_streaming_buffer()

    # Process with this alpha value
    config = {
        'moving_avg_window': 400,
        'tanh_alpha': alpha,
        'spike_threshold': 5.0
    }

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk.copy(), config)

    # Calculate metrics (now using PRE-tanh signal from metadata)
    yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
    health_status = check_system_health(cleaned_tensor, metadata, yield_pct)

    # For comparison: calculate POST-tanh variance (what we used to use)
    post_tanh_variance = float(np.var(cleaned_tensor))

    # PRE-tanh variance is now in metadata (what we NOW use)
    pre_tanh_variance = metadata['variance']

    results.append({
        'alpha': alpha,
        'yield': yield_pct,
        'health': health_status,
        'post_var': post_tanh_variance,
        'pre_var': pre_tanh_variance
    })

    print(f"{alpha:<15.1f} {yield_pct:<12.1f} {health_status:<15} {post_tanh_variance:<15.3f} {pre_tanh_variance:<20.1f}")

print()
print("=" * 80)
print("ANALYSIS:")
print("=" * 80)

# Calculate yield variance across alpha values
yields = [r['yield'] for r in results]
yield_std = np.std(yields)
yield_mean = np.mean(yields)
yield_cv = (yield_std / yield_mean) * 100 if yield_mean > 0 else 0

print(f"Yield Mean:         {yield_mean:.2f}%")
print(f"Yield Std Dev:      {yield_std:.2f}%")
print(f"Coefficient of Var: {yield_cv:.2f}% (lower is better)")
print()

# Verify PRE-tanh variance is constant (should be, since same raw signal)
pre_vars = [r['pre_var'] for r in results]
pre_var_std = np.std(pre_vars)
pre_var_mean = np.mean(pre_vars)
print(f"PRE-tanh Variance Mean: {pre_var_mean:.1f} uV^2 (should be constant)")
print(f"PRE-tanh Variance Std:  {pre_var_std:.1f} uV^2 (should be ~0)")
print()

# Compare POST-tanh variance (should vary with alpha)
post_vars = [r['post_var'] for r in results]
post_var_std = np.std(post_vars)
post_var_mean = np.mean(post_vars)
print(f"POST-tanh Variance Mean: {post_var_mean:.3f} (for reference)")
print(f"POST-tanh Variance Std:  {post_var_std:.3f} (DOES vary with alpha)")
print()

# Verdict
print("=" * 80)
print("VERDICT:")
print("=" * 80)

if yield_cv < 2.0:  # Less than 2% variation
    print("[PASS] Signal Yield is DECOUPLED from Tanh Alpha slider!")
    print(f"   Yield variance across alpha values: {yield_cv:.2f}% (excellent)")
    print()
    print("   User can now adjust Tanh Alpha from 0.2 to 3.0 for clean R&D visualization")
    print("   WITHOUT affecting the Clinical View yield percentage.")
    print()
    print("   [FIXED] Metrics now evaluate PRE-tanh biological signal,")
    print("           not POST-tanh aesthetic visualization.")
else:
    print("[FAIL] Signal Yield still coupled to Tanh Alpha slider")
    print(f"   Yield variance: {yield_cv:.2f}% (too high, should be < 2%)")
    print()
    print("   The metrics engine is still using POST-tanh signal for calculations.")
    print("   Need to verify metadata['variance'] and metadata['mean'] are PRE-tanh.")

print("=" * 80)
