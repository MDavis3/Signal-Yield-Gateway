# Bug Report: Axoft Signal Yield Gateway

**Date:** 2026-03-02
**Test Results:** 62/66 tests passed (93.9%), **4 failures identified**

---

## CRITICAL BUG #1: Spike Detection Massively Over-Sensitive
**File:** `dsp_pipeline.py:detect_spikes_derivative()`
**Severity:** CRITICAL - Renders spike count metric useless

### Issue:
The derivative-based spike detection is detecting **1792 spikes out of 2000 samples** (89.6% of all samples!). This is physically impossible. For a 50ms chunk at 20Hz firing rate, we should expect **1-3 spikes maximum**, not 1792.

### Root Cause:
The threshold of `3.0` is being applied to the raw derivative values, but after moving average subtraction and normalization, the derivative magnitudes are much smaller. Almost every sample's derivative exceeds 3.0 after the signal transformations.

### Test Evidence:
```
[Test 6.8] Spike count is physically plausible
  [FAIL] Spike count 1792 is plausible (<100 for 50ms)
```

### Fix Required:
Either:
1. Increase the default spike threshold from 3.0 to 30-50
2. Normalize the derivatives before thresholding
3. Use a percentile-based threshold (e.g., top 1% of derivative magnitudes)

---

## BUG #2: Drift Amplitude Scaling Incorrect
**File:** `data_simulator.py:generate_synthetic_chunk()`
**Severity:** MEDIUM - Affects simulation realism

### Issue:
When `drift_severity=1.0`, the generated drift amplitude is only **38 μV** instead of the expected **200-500 μV** range documented in the code comments.

### Test Evidence:
```
[Test 1.3] Drift-only signal (sinusoidal)
  [FAIL] Drift amplitude in expected range
         Value 37.95982360839844 outside range [200.0, 500.0]
```

### Root Cause:
Looking at `data_simulator.py` lines 182-193, the drift calculation seems correct:
```python
heartbeat_amplitude = 200.0 * drift_severity  # Should be 200
respiration_amplitude = 150.0 * drift_severity  # Should be 150
```

But the observed amplitude is ~38, suggesting the **sine waves are canceling each other out** due to phase alignment, rather than adding constructively.

### Fix Required:
Use different frequencies or phase offsets to ensure constructive addition:
```python
heartbeat_drift = heartbeat_amplitude * np.sin(2 * np.pi * heartbeat_freq * time_axis)
respiration_drift = respiration_amplitude * np.sin(2 * np.pi * respiration_freq * time_axis + np.pi/4)
```

---

## BUG #3: System Health Status Classification Error
**File:** `metrics_engine.py:check_system_health()`
**Severity:** MEDIUM - Affects clinical dashboard accuracy

### Issue:
A signal with **95% yield** (excellent quality) and **healthy variance** is being classified as something other than "healthy".

### Test Evidence:
```
[Test 7.4] System health status
  [FAIL] High yield (95%) -> healthy status
```

### Root Cause:
Need to examine the conditional logic in `check_system_health()`. Likely an issue with the variance range check on line ~489:
```python
if variance < 0.01 or variance > 0.5:
    return "warning"
```

The test signal might have variance slightly outside this range despite being healthy.

### Fix Required:
Review and potentially widen the variance thresholds, or add variance check to metadata before calling health check.

---

## BUG #4: Tanh Alpha Compression Test Logic Error
**File:** `test_suite.py:test_tanh_normalization()` OR `dsp_pipeline.py:tanh_normalize()`
**Severity:** LOW - May be test logic issue, not code bug

### Issue:
The test expects that higher alpha values should compress more aggressively, but the test is failing.

### Test Evidence:
```
[Test 5.3] Alpha parameter effect
  [FAIL] Higher alpha compresses more
```

### Root Cause:
Need to inspect actual values. The test does:
```python
signal = np.array([0.5, 1.0, 2.0])
normalized_soft = tanh_normalize(signal, alpha=0.5)  # Soft
normalized_hard = tanh_normalize(signal, alpha=2.0)  # Hard
# Expects: abs(normalized_hard[2]) < abs(normalized_soft[2])
```

For `signal[2] = 2.0`:
- `tanh(0.5 * 2.0) = tanh(1.0) = 0.762`
- `tanh(2.0 * 2.0) = tanh(4.0) = 0.999`

Wait, this is backwards! Higher alpha brings values **closer to ±1**, not closer to 0. The test logic is wrong - "compress more" should mean "closer to saturation", not "smaller magnitude".

### Fix Required:
Fix the test assertion:
```python
result.assert_true(np.abs(normalized_hard[2]) > np.abs(normalized_soft[2]),
                  "Higher alpha compresses more (closer to saturation)")
```

---

## Summary of Fixes Needed

| Bug # | File | Line(s) | Severity | Action |
|-------|------|---------|----------|--------|
| 1 | `dsp_pipeline.py` | ~95-120 | CRITICAL | Increase spike detection threshold 10x |
| 2 | `data_simulator.py` | ~190 | MEDIUM | Add phase offset to respiration sine wave |
| 3 | `metrics_engine.py` | ~489 | MEDIUM | Widen variance thresholds in health check |
| 4 | `test_suite.py` | ~311 | LOW | Fix test assertion logic (code is correct) |

---

## Test Statistics

```
Total Tests: 66
Passed: 62 (93.9%)
Failed: 4 (6.1%)

Breakdown by Category:
✓ Data Simulator: 6/7 passed (85.7%)
✓ Circular Buffer: 4/4 passed (100%)
✓ Moving Average: 3/3 passed (100%)
✓ Spike Detection: 4/4 passed (100%)
✓ Tanh Normalization: 3/4 passed (75.0%)
✓ Full DSP Pipeline: 13/14 passed (92.9%)
✓ Metrics Engine: 7/8 passed (87.5%)
✓ Storage Manager: 8/8 passed (100%)
✓ End-to-End Integration: 14/14 passed (100%)
```

---

## Impact Assessment

**For Oliver's Demo:**
- Bug #1 (spike detection) will make the "Live Signal Yield %" metric unreliable
- Bug #2 (drift amplitude) makes the simulator less realistic but doesn't affect real hardware
- Bug #3 (health status) will occasionally show "warning" when it should show "healthy"
- Bug #4 (test logic) is not a code bug, just a test issue

**Recommendation:** Fix Bugs #1, #2, #3 before the Loom presentation to Oliver.
