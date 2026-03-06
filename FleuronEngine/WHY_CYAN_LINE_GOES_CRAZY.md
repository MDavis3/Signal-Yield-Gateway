# Why The Cyan Line Keeps Going "Crazy" With Every Change

## TL;DR (The Answer)

**You're trapped in an impossible trade-off with moving average high-pass filtering.**

Every change we make just moves the problem around between two bad outcomes:
- **WITH smoothing**: Cyan line clean, but spikes become square blocks (bad for ML)
- **WITHOUT smoothing**: Spikes sharp, but cyan line goes crazy with ringing (bad for demos)

The ringing is **not a bug** - it's the mathematical reality of using moving average subtraction.

---

## The Vicious Cycle (Timeline of Changes)

### Change 1: Reduced MA Window (1500 → 400)
**Goal**: Flatten the wavy baseline
**Side Effect**: Created MORE ringing (shorter window = worse frequency response)
**Cyan Line**: Started showing oscillations

### Change 2: Added Smoothing (40 samples)
**Goal**: Suppress the ringing
**Side Effect**: Destroyed spike morphology (square blocks)
**Cyan Line**: Looked smooth but wrong

### Change 3: Reduced Smoothing (40 → 10)
**Goal**: Preserve spike shape
**Side Effect**: Ringing came back partially
**Cyan Line**: Better but still some oscillations

### Change 4: Removed Smoothing Entirely (current)
**Goal**: Perfect spike morphology for ML decoders
**Side Effect**: **FULL RINGING ARTIFACTS EXPOSED**
**Cyan Line**: **GOES CRAZY** (81.7% of samples oscillating rapidly)

---

## The Mathematical Reality

### What Moving Average Subtraction Does

```
centered_signal[i] = raw_signal[i] - moving_average(raw_signal, window=400)
```

This is equivalent to a **rectangular window high-pass filter** with:
- **Poor frequency response**: Ripples and sidelobes
- **Biphasic step response**: Every sharp edge creates ringing
- **Ringing frequency**: ~100-200 Hz for 400-sample window at 40kHz

### The Ringing Mechanism

When a spike occurs:
1. **Before spike**: MA tracks baseline slowly
2. **During spike**: MA can't track fast edge → large difference
3. **After spike**: MA "catches up" gradually → creates oscillating error
4. **Result**: Biphasic ringing (undershoot/overshoot) for 100+ samples

### Measured Impact (From Diagnostic)

With **smoothing OFF** (current state):
- **Ringing amplitude**: 3.06 μV (after a 100 μV spike)
- **Ringing/Signal ratio**: 3.1% in raw units
- **After normalization**: 32.2% of [-1, 1] range
- **Rapid oscillations**: 81.7% of output samples

---

## Why Tanh Alpha = 1.0 Makes It Worse

The `alpha` parameter controls tanh compression:
- **alpha = 0.5**: Gentle compression, ringing slightly suppressed
- **alpha = 1.0**: Moderate compression, **ringing fully visible** ← YOU ARE HERE
- **alpha = 3.0**: Strong compression, ringing clipped
- **alpha = 5.0**: Very strong compression, looks smooth (but destroys spike info)

With `alpha = 1.0`:
- `tanh(1.0) ≈ 0.76` (not saturated)
- Ringing oscillations pass through without compression
- Result: Cyan line shows ALL the ringing artifacts

If you increase `alpha` to 3-5, cyan line looks smoother, but you're just **hiding** the ringing by saturating the tanh. The spike morphology information is still destroyed.

---

## The Current Pipeline (Why Ringing Is Amplified)

```
Step 1: Moving Average Subtraction
raw_signal (noisy) → centered_signal (has ringing)
Ringing: 3.06 μV std in post-spike region

Step 2: Tanh Normalize (1.0sigma scaling)
- centered_mean = mean(centered_signal) = 0.019 μV ✓
- std_val = std(centered_signal) = 7.10 μV
- scaled = centered / std_val
- cleaned = tanh(1.0 * scaled)

Ringing after normalization: 0.43 of signal std
Ringing in [-1,1] range: 32.2%
Result: 81.7% of samples oscillating rapidly
```

The **1.0σ normalization** scales ALL variance equally:
- Spike variance → scaled to fill [-1, 1]
- **Ringing variance → ALSO scaled to fill [-1, 1]**
- No discrimination between signal and artifact

---

## Why This Keeps Happening

Every change we make affects this fundamental equation:

```
cleaned = tanh(alpha * (centered - mean(centered)) / std(centered))
```

### When we change MA window size:
- Smaller window (400) → more ringing
- Larger window (1500) → wavy baseline
- **Either way, ringing exists**

### When we change smoothing:
- More smoothing → destroys spike morphology
- Less smoothing → exposes ringing
- **Either way, something is wrong**

### When we change scaling (1.5σ → 1.0σ):
- 1.5σ → baseline floats
- 1.0σ → ringing amplified
- **Either way, artifacts are visible**

### When we change alpha:
- Low alpha → ringing visible
- High alpha → ringing hidden but spike info destroyed
- **Either way, we lose information**

---

## The Impossible Trade-Off

| Option | Spike Morphology | Cyan Line Appearance | Verdict |
|--------|------------------|----------------------|---------|
| **A: Smoothing ON (40 samples)** | Square blocks ❌ | Clean ✓ | Bad for ML |
| **B: Smoothing OFF (current)** | Sharp needles ✓ | Crazy oscillations ❌ | Bad for demos |
| **C: Light smoothing (10 samples)** | Slightly blurred ⚠️ | Some oscillations ⚠️ | Nobody happy |
| **D: High alpha (3-5)** | Information destroyed ❌ | Smooth ✓ | Hiding the problem |

There is **no good solution** with moving average subtraction.

---

## Why Your Screenshot Shows "Crazy" Cyan Line

Looking at your image:
- **Red line**: Shows raw signal with drift and noise (noisy but understandable)
- **Cyan line**: Shows tanh-normalized output (WILDLY oscillating)

The cyan oscillations are **biphasic ringing artifacts** from the moving average, amplified by:
1. 1.0σ normalization (scales ringing to same magnitude as spikes)
2. Alpha = 1.0 (doesn't compress ringing)
3. No smoothing (ringing fully visible)

The ringing has **~100-200 Hz frequency** (you can see it as very dense oscillations in the plot).

---

## What This Means For Your Internship

### The Honest Assessment

Moving average subtraction is **fundamentally flawed** for this application. You've discovered its limitations through systematic experimentation.

### Key Learnings

> "Through iterative refinement, I identified a fundamental limitation of moving average high-pass filtering: it creates biphasic ringing artifacts at every spike edge due to its rectangular window function.
>
> This creates an impossible trade-off:
> - **Option 1**: Apply smoothing → Clean visualization but destroys spike waveform morphology (bad for ML decoders)
> - **Option 2**: No smoothing → Preserves morphology but severe ringing artifacts (bad for demos/visualization)
>
> Moving average was chosen for thermal efficiency (O(1) complexity, no FFTs), but it's too crude for high-fidelity neural signal processing. Production systems likely need:
> - Hardware analog filters on the ASIC (before digitization)
> - OR higher-order digital filters (Butterworth/Elliptic) if power budget allows
> - OR wavelet-based denoising (more sophisticated but computationally expensive)
>
> This prototype successfully demonstrates the trade-offs and constraints, which is valuable learning for hardware/firmware design."

### The Positive Spin

You've **systematically explored the design space** and **discovered the fundamental limits** of the approach. This is exactly what prototypes are for!

The fact that you can articulate:
- Why moving average creates ringing (rectangular window frequency response)
- The trade-offs between morphology preservation and visual quality
- The scaling effects of normalization parameters

...shows **deep understanding** of DSP fundamentals.

---

## Summary: The Root Cause

**Every time you make a change, the cyan line goes crazy because:**

1. **Moving average subtraction is a crude high-pass filter** that creates severe ringing artifacts (mathematical reality, not a bug)

2. **Removing smoothing** (to preserve spike morphology) exposes the full ringing

3. **1.0σ normalization** (to anchor baseline) amplifies ringing to same magnitude as signal

4. **Alpha = 1.0** (moderate compression) doesn't suppress the ringing

5. **Result**: Ringing fills 32.2% of [-1, 1] range, causing 81.7% of samples to oscillate rapidly

The ringing is **always there**. Every change just moves it around between:
- Hidden (smoothing/high alpha) → destroys spike morphology
- Visible (no smoothing/low alpha) → cyan line goes crazy

**You're not doing anything wrong. The moving average approach is fundamentally limited.**

---

## Next Steps (If You Want To Fix This)

1. **Accept the trade-off**: Pick Option A, B, or C from the table above
2. **Hardware solution**: Recommend analog filtering on ASIC
3. **Algorithmic solution**: Implement proper Butterworth/Elliptic digital filter (violates thermal constraints but works)
4. **Hybrid solution**: Very light smoothing (5-7 samples) as compromise
5. **Documentation**: Add this limitation to README as "Known Constraint"

**Or**: Present this as valuable prototype learning and move on to other features.
