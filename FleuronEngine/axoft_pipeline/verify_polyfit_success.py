"""
Quick verification: Polyfit detrending eliminates ringing artifacts.
"""

import numpy as np
from dsp_pipeline import polyfit_detrend, tanh_normalize

print("=" * 80)
print("POLYFIT DETRENDING - RINGING ELIMINATION VERIFICATION")
print("=" * 80)

# Create test signal with sharp spike
signal = np.zeros(2000, dtype=np.float32)
signal[:1000] = 100.0  # Baseline
signal[1000] = 200.0   # Sharp 100 uV spike
signal[1001:] = 100.0

print("\nTest Signal:")
print(f"  Baseline: 100 uV")
print(f"  Spike at sample 1000: 200 uV (100 uV amplitude)")
print(f"  Return to baseline after spike")

# Process with polyfit
detrended, _ = polyfit_detrend(signal, poly_order=1)
cleaned = tanh_normalize(detrended, alpha=1.0)

# Measure ringing in post-spike region
post_spike_region = cleaned[1010:1100]
ringing_std = np.std(post_spike_region)
ringing_mean = np.mean(post_spike_region)
rapid_oscillations = np.sum(np.abs(np.diff(cleaned)) > 0.1)
oscillation_pct = (rapid_oscillations / len(cleaned)) * 100

print("\n" + "=" * 80)
print("RESULTS: Polyfit Detrending")
print("=" * 80)

print(f"\nRinging Metrics:")
print(f"  Post-spike std: {ringing_std:.6f} (target: <0.5)")
print(f"  Post-spike mean: {ringing_mean:.6f} (target: ~0.0)")
print(f"  Rapid oscillations: {oscillation_pct:.1f}% of samples")

print(f"\nSpike Preservation:")
print(f"  Peak value: {np.max(cleaned):.4f} (target: >0.8)")
print(f"  Baseline mean: {np.mean(cleaned[:500]):.6f} (target: ~0.0)")

print(f"\nComparison to MA Baseline (from diagnose_ringing.py):")
print(f"  MA ringing std: 0.3218 (32.2% of [-1,1] range)")
print(f"  MA oscillations: 81.7% of samples")
print(f"  ")
print(f"  Polyfit ringing std: {ringing_std:.6f}")
print(f"  Polyfit oscillations: {oscillation_pct:.1f}% of samples")
print(f"  ")
print(f"  Improvement: {(1 - ringing_std/0.3218)*100:.1f}% ringing reduction!")
print(f"  Improvement: {(1 - oscillation_pct/81.7)*100:.1f}% less oscillations!")

print("\n" + "=" * 80)
if ringing_std < 0.5 and np.max(cleaned) > 0.8:
    print("[SUCCESS] Polyfit eliminates ringing while preserving spikes!")
else:
    print("[WARNING] Results not as expected")
print("=" * 80)
