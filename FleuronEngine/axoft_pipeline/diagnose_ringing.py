"""
Diagnostic: Why does the cyan line keep going "crazy" with every change?

This script reveals the fundamental recurring problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from dsp_pipeline import moving_average_subtract, tanh_normalize, CircularBuffer

print("=" * 80)
print("DIAGNOSTIC: Why the Cyan Line Goes Crazy")
print("=" * 80)

# Create a simple test signal: Clean spike on flat baseline
n_samples = 2000
signal = np.zeros(n_samples, dtype=np.float32)

# Add ONE clean spike at sample 1000
spike_idx = 1000
spike_amplitude = 100.0  # 100 uV
for i in range(spike_idx - 10, spike_idx + 30):
    if 0 <= i < n_samples:
        t_rel = (i - spike_idx) / 40000.0
        if 0 <= t_rel < 0.0003:  # Rising edge
            signal[i] = spike_amplitude * (t_rel / 0.0003)
        elif 0.0003 <= t_rel < 0.001:  # Falling edge
            decay = t_rel - 0.0003
            signal[i] = spike_amplitude * np.exp(-decay / 0.0002)

# Add small baseline noise
signal += np.random.randn(n_samples) * 3.0  # 3 uV noise

print("\n1. ORIGINAL SIGNAL (One clean spike + noise)")
print(f"   Mean: {np.mean(signal):.2f} uV")
print(f"   Std:  {np.std(signal):.2f} uV")
print(f"   Max spike: {np.max(signal):.2f} uV")

# Step 1: Apply moving average subtraction
centered, _ = moving_average_subtract(signal, window_size=400, buffer=CircularBuffer(400))

print("\n2. AFTER MOVING AVERAGE SUBTRACTION (centered)")
print(f"   Mean: {np.mean(centered):.4f} uV (should be ~0)")
print(f"   Std:  {np.std(centered):.2f} uV")
print(f"   Max:  {np.max(centered):.2f} uV")
print(f"   Min:  {np.min(centered):.2f} uV")

# CRITICAL: Analyze the ringing artifacts
# Look at a window around the spike
spike_window = centered[spike_idx-100:spike_idx+200]
spike_peak_idx = np.argmax(np.abs(spike_window))
spike_peak_val = spike_window[spike_peak_idx]

# Count oscillations (zero crossings in spike region)
zero_crossings = np.sum(np.diff(np.sign(spike_window)) != 0)

# Measure ringing amplitude (std deviation in post-spike region)
post_spike = centered[spike_idx+50:spike_idx+150]  # 50-150 samples after spike
ringing_amplitude = np.std(post_spike)

print("\n3. RINGING ANALYSIS (The Core Problem)")
print(f"   Spike peak: {spike_peak_val:.2f} uV")
print(f"   Post-spike ringing std: {ringing_amplitude:.2f} uV")
print(f"   Zero crossings near spike: {zero_crossings}")
print(f"   Ringing/Signal ratio: {ringing_amplitude / spike_peak_val * 100:.1f}%")

# Key insight: MA subtraction creates biphasic ringing
before_spike = centered[spike_idx-50:spike_idx-10]
during_spike = centered[spike_idx-10:spike_idx+30]
after_spike = centered[spike_idx+30:spike_idx+100]

print(f"\n   Before spike: std = {np.std(before_spike):.2f} uV")
print(f"   During spike: std = {np.std(during_spike):.2f} uV")
print(f"   After spike:  std = {np.std(after_spike):.2f} uV (RINGING!)")

# Step 2: Apply current tanh_normalize (with 1.0sigma scaling)
cleaned = tanh_normalize(centered, alpha=1.0)

print("\n4. AFTER TANH NORMALIZE (alpha=1.0, 1.0sigma scaling)")
print(f"   Mean: {np.mean(cleaned):.6f}")
print(f"   Std:  {np.std(cleaned):.4f}")
print(f"   Max:  {np.max(cleaned):.4f}")
print(f"   Min:  {np.min(cleaned):.4f}")

# The problem: Compute what fraction of output is ringing vs spike
cleaned_window = cleaned[spike_idx-100:spike_idx+200]
cleaned_post_spike = cleaned[spike_idx+50:spike_idx+150]
cleaned_ringing_std = np.std(cleaned_post_spike)

print(f"\n   Post-spike ringing std: {cleaned_ringing_std:.4f}")
print(f"   Ringing fills {cleaned_ringing_std / 1.0 * 100:.1f}% of [-1, 1] range")

# Measure high-frequency content
# Count samples oscillating rapidly (abs(diff) > threshold)
rapid_changes = np.sum(np.abs(np.diff(cleaned)) > 0.1)
print(f"   Rapid oscillations (|diff| > 0.1): {rapid_changes} samples ({rapid_changes/len(cleaned)*100:.1f}%)")

print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("\nThe Vicious Cycle:")
print("1. Moving average subtraction creates BIPHASIC RINGING at every spike edge")
print("   - This is fundamental to MA high-pass filtering")
print("   - Ringing frequency ~100-200 Hz for 400-sample window at 40kHz")
print(f"   - Measured ringing/signal ratio: {ringing_amplitude / spike_peak_val * 100:.1f}%")

print("\n2. With our fixes, we REMOVED smoothing to preserve spike morphology")
print("   - No smoothing = full ringing artifacts present")
print(f"   - Ringing std: {ringing_amplitude:.2f} uV (significant!)")

print("\n3. Tanh normalize with 1.0sigma scaling amplifies ALL variance")
print(f"   - Centered std: {np.std(centered):.2f} uV")
print(f"   - Normalization divides by this std")
print(f"   - Ringing gets scaled to: {ringing_amplitude / np.std(centered):.4f} of [-1,1] range")

print("\n4. With alpha=1.0, tanh doesn't compress the ringing")
print(f"   - tanh(1.0) = 0.76 (not saturated)")
print(f"   - Ringing oscillations visible throughout output")
print(f"   - Result: {rapid_changes/len(cleaned)*100:.1f}% of signal shows rapid oscillations")

print("\n" + "=" * 80)
print("WHY THIS KEEPS HAPPENING")
print("=" * 80)

print("\nYou're caught in an IMPOSSIBLE TRADE-OFF with moving average:")
print("")
print("Option A: Keep smoothing (40 samples)")
print("  - Pro: Cyan line looks smooth")
print("  - Con: Destroys spike morphology (square blocks)")
print("  - Verdict: Bad for ML decoders")
print("")
print("Option B: Remove smoothing (current)")
print("  - Pro: Preserves sharp spike needles")
print("  - Con: Cyan line goes CRAZY with ringing")
print("  - Verdict: Bad for visualization/demos")
print("")
print("Option C: Light smoothing (10 samples)")
print("  - Pro: Compromise between sharp and clean")
print("  - Con: Still some ringing, still some morphology loss")
print("  - Verdict: Nobody's happy")

print("\n" + "=" * 80)
print("THE FUNDAMENTAL PROBLEM")
print("=" * 80)
print("\nMoving average subtraction is a CRUDE HIGH-PASS FILTER that creates")
print("severe ringing artifacts. This is not a bug - it's the mathematical reality")
print("of using a rectangular window function.")
print("")
print("Every time you make ANY change (MA window size, smoothing, scaling),")
print("you're just moving the problem around. The ringing is ALWAYS there.")
print("")
print("With smoothing OFF (to preserve spikes):")
print(f"  - Ringing amplitude: {ringing_amplitude:.2f} uV")
print(f"  - Normalized ringing: {ringing_amplitude / np.std(centered):.2f} of signal std")
print(f"  - Visible as 'crazy' high-frequency oscillations in cyan line")
print("")
print("This is why the cyan line keeps going crazy with every change.")
print("=" * 80)
