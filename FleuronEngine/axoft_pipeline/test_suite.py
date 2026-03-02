"""
Comprehensive Test Suite for Axoft Signal Yield Gateway
========================================================

This test suite validates the correctness of:
1. Data simulator (synthetic signal generation)
2. DSP pipeline (moving average, spike detection, tanh normalization)
3. Metrics engine (signal yield, stability tracking, channel mapping)
4. Storage manager (in-memory backend)

Author: Lead Systems Architect
Date: 2026-03-02
"""

import numpy as np
import time
import sys

# Import all modules to test
from data_simulator import generate_synthetic_chunk, generate_batch
from dsp_pipeline import (
    CircularBuffer,
    moving_average_subtract,
    detect_spikes_derivative,
    tanh_normalize,
    process_signal_streaming,
    reset_streaming_buffer
)
from metrics_engine import (
    calculate_signal_yield,
    calculate_active_channels,
    check_system_health,
    StabilityTracker
)
from storage_manager import InMemoryStorage


# ============================================================================
# Test Utilities
# ============================================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_true(self, condition, test_name, error_msg=""):
        if condition:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            error = f"{test_name}: {error_msg}"
            self.errors.append(error)
            print(f"  [FAIL] {test_name}")
            if error_msg:
                print(f"         {error_msg}")

    def assert_close(self, actual, expected, tolerance, test_name):
        if abs(actual - expected) <= tolerance:
            self.passed += 1
            print(f"  [PASS] {test_name} (actual={actual:.4f}, expected={expected:.4f})")
        else:
            self.failed += 1
            error = f"{test_name}: Expected {expected:.4f}, got {actual:.4f} (tolerance={tolerance})"
            self.errors.append(error)
            print(f"  [FAIL] {test_name}")
            print(f"         Expected {expected:.4f}, got {actual:.4f} (tolerance={tolerance})")

    def assert_range(self, value, min_val, max_val, test_name):
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"  [PASS] {test_name} (value={value}, range=[{min_val}, {max_val}])")
        else:
            self.failed += 1
            error = f"{test_name}: Value {value} outside range [{min_val}, {max_val}]"
            self.errors.append(error)
            print(f"  [FAIL] {test_name}")
            print(f"         Value {value} outside range [{min_val}, {max_val}]")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 80)
        print(f"TEST SUMMARY: {self.passed}/{total} passed, {self.failed}/{total} failed")
        if self.failed > 0:
            print("\nFAILURES:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        print("=" * 80)
        return self.failed == 0


# ============================================================================
# Test 1: Data Simulator
# ============================================================================

def test_data_simulator():
    print("\n" + "=" * 80)
    print("TEST 1: DATA SIMULATOR")
    print("=" * 80)

    result = TestResult()

    # Test 1.1: Basic chunk generation
    print("\n[Test 1.1] Basic chunk generation")
    chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.0,
        drift_severity=0.0,
        spike_rate=0.0,
        seed=42
    )
    result.assert_true(len(chunk) == 2000, "Chunk has correct length (2000 samples)")
    result.assert_true(chunk.dtype == np.float32, "Chunk is float32 type")

    # Test 1.2: Noise-only signal
    print("\n[Test 1.2] Noise-only signal")
    noise_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.5,
        drift_severity=0.0,
        spike_rate=0.0,
        seed=42
    )
    noise_std = np.std(noise_chunk)
    result.assert_range(noise_std, 5.0, 25.0, "Noise standard deviation in reasonable range")

    # Test 1.3: Drift-only signal
    print("\n[Test 1.3] Drift-only signal (sinusoidal)")
    drift_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.0,
        drift_severity=1.0,
        spike_rate=0.0,
        seed=42
    )
    drift_amplitude = (drift_chunk.max() - drift_chunk.min()) / 2
    # For 50ms chunks with 1Hz heartbeat (0.05 cycles) and 0.3Hz respiration (0.015 cycles),
    # we only capture a small portion of the drift. Expect 20-400 μV depending on phase.
    result.assert_range(drift_amplitude, 20.0, 400.0, "Drift amplitude in expected range for 50ms chunk")

    # Test 1.4: Spike-only signal
    print("\n[Test 1.4] Spike-only signal")
    spike_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.0,
        drift_severity=0.0,
        spike_rate=20.0,
        seed=42
    )
    # For 50ms at 20Hz, expect ~1 spike (20 spikes/second * 0.05 seconds = 1 spike)
    # Spike amplitude should be ~100 uV
    result.assert_true(spike_chunk.max() > 50.0, "Spike amplitude is significant (>50 uV)")

    # Test 1.5: Full signal (noise + drift + spikes)
    print("\n[Test 1.5] Full signal (noise + drift + spikes)")
    full_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.5,
        drift_severity=1.5,
        spike_rate=25.0,
        seed=42
    )
    result.assert_true(len(full_chunk) == 2000, "Full signal has correct length")
    result.assert_range(full_chunk.max(), 50.0, 800.0, "Full signal max in reasonable range")

    return result


# ============================================================================
# Test 2: Circular Buffer (Moving Average)
# ============================================================================

def test_circular_buffer():
    print("\n" + "=" * 80)
    print("TEST 2: CIRCULAR BUFFER")
    print("=" * 80)

    result = TestResult()

    # Test 2.1: Basic buffer operations
    print("\n[Test 2.1] Basic buffer operations")
    buffer = CircularBuffer(size=5)

    # Add first value
    avg = buffer.add(10.0)
    result.assert_close(avg, 10.0, 0.01, "First value average is itself")

    # Add more values
    buffer.add(20.0)
    buffer.add(30.0)
    avg = buffer.add(40.0)
    result.assert_close(avg, 25.0, 0.01, "Average of [10, 20, 30, 40] is 25.0")

    # Test 2.2: Buffer overflow (exceeds size limit)
    print("\n[Test 2.2] Buffer overflow handling")
    buffer.add(50.0)  # [10, 20, 30, 40, 50]
    avg = buffer.add(60.0)  # Should drop 10, now [20, 30, 40, 50, 60]
    result.assert_close(avg, 40.0, 0.01, "Buffer correctly drops oldest value")

    # Test 2.3: Buffer reset
    print("\n[Test 2.3] Buffer reset")
    buffer.reset()
    avg = buffer.add(100.0)
    result.assert_close(avg, 100.0, 0.01, "Buffer resets correctly")

    return result


# ============================================================================
# Test 3: Moving Average Subtraction
# ============================================================================

def test_moving_average_subtraction():
    print("\n" + "=" * 80)
    print("TEST 3: MOVING AVERAGE SUBTRACTION")
    print("=" * 80)

    result = TestResult()

    # Test 3.1: Constant signal (should center to ~0)
    print("\n[Test 3.1] Constant signal")
    constant_signal = np.ones(100) * 50.0
    centered, buffer = moving_average_subtract(constant_signal, window_size=10)
    # After stabilization, centered signal should be near 0
    result.assert_close(np.mean(centered[-50:]), 0.0, 5.0, "Constant signal centers to ~0")

    # Test 3.2: Linear drift removal
    print("\n[Test 3.2] Linear drift removal")
    linear_drift = np.linspace(0, 100, 100)  # Linear ramp
    centered, buffer = moving_average_subtract(linear_drift, window_size=10)
    # The drift should be significantly reduced
    original_std = np.std(linear_drift)
    centered_std = np.std(centered)
    result.assert_true(centered_std < original_std, "Drift removal reduces variance")

    # Test 3.3: Spike preservation
    print("\n[Test 3.3] Spike preservation")
    signal_with_spike = np.zeros(100)
    signal_with_spike[50] = 100.0  # Add a spike
    centered, buffer = moving_average_subtract(signal_with_spike, window_size=10)
    # The spike should still be visible in the centered signal
    result.assert_true(centered[50] > 50.0, "Moving average preserves spikes")

    return result


# ============================================================================
# Test 4: Spike Detection
# ============================================================================

def test_spike_detection():
    print("\n" + "=" * 80)
    print("TEST 4: SPIKE DETECTION")
    print("=" * 80)

    result = TestResult()

    # Test 4.1: No spikes in flat signal
    print("\n[Test 4.1] No spikes in flat signal")
    flat_signal = np.zeros(100)
    spike_count = detect_spikes_derivative(flat_signal, threshold=3.0)
    result.assert_true(spike_count == 0, "Flat signal has no spikes")

    # Test 4.2: Single clear spike
    print("\n[Test 4.2] Single clear spike")
    signal_with_spike = np.zeros(100)
    signal_with_spike[50:55] = [0, 50, 100, 50, 0]  # Sharp spike
    spike_count = detect_spikes_derivative(signal_with_spike, threshold=10.0)
    result.assert_range(spike_count, 1, 5, "Single spike detected (1-5 threshold crossings)")

    # Test 4.3: Multiple spikes
    print("\n[Test 4.3] Multiple spikes")
    signal_with_spikes = np.zeros(200)
    signal_with_spikes[50:55] = [0, 50, 100, 50, 0]
    signal_with_spikes[100:105] = [0, 50, 100, 50, 0]
    signal_with_spikes[150:155] = [0, 50, 100, 50, 0]
    spike_count = detect_spikes_derivative(signal_with_spikes, threshold=10.0)
    result.assert_range(spike_count, 3, 15, "Multiple spikes detected")

    # Test 4.4: Noise rejection
    print("\n[Test 4.4] Noise rejection (high threshold)")
    noisy_signal = np.random.randn(100) * 2.0  # Small noise
    spike_count = detect_spikes_derivative(noisy_signal, threshold=10.0)
    result.assert_true(spike_count < 10, "High threshold rejects noise")

    return result


# ============================================================================
# Test 5: Tanh Normalization
# ============================================================================

def test_tanh_normalization():
    print("\n" + "=" * 80)
    print("TEST 5: TANH NORMALIZATION")
    print("=" * 80)

    result = TestResult()

    # Test 5.1: Bounds guarantee [-1, 1]
    print("\n[Test 5.1] Bounds guarantee [-1, 1]")
    large_signal = np.linspace(-1000, 1000, 100)
    normalized = tanh_normalize(large_signal, alpha=1.0)
    result.assert_true(np.all(normalized >= -1.0) and np.all(normalized <= 1.0),
                      "All values within [-1, 1] bounds")
    result.assert_close(normalized.min(), -1.0, 0.01, "Minimum approaches -1")
    result.assert_close(normalized.max(), 1.0, 0.01, "Maximum approaches 1")

    # Test 5.2: Linear passthrough for small values
    print("\n[Test 5.2] Linear passthrough for small values")
    small_signal = np.array([0.1, 0.2, 0.3])
    normalized = tanh_normalize(small_signal, alpha=1.0)
    # tanh(x) ≈ x for small x
    result.assert_close(normalized[0], 0.1, 0.01, "Small values pass through linearly")

    # Test 5.3: Alpha parameter effect
    print("\n[Test 5.3] Alpha parameter effect")
    signal = np.array([0.5, 1.0, 2.0])
    normalized_soft = tanh_normalize(signal, alpha=0.5)  # Soft compression
    normalized_hard = tanh_normalize(signal, alpha=2.0)  # Hard compression
    # Higher alpha should compress more aggressively (bring closer to saturation at ±1)
    result.assert_true(np.abs(normalized_hard[2]) > np.abs(normalized_soft[2]),
                      "Higher alpha compresses more (closer to saturation)")

    # Test 5.4: Float32 output type
    print("\n[Test 5.4] Float32 output type")
    signal = np.array([1.0, 2.0, 3.0])
    normalized = tanh_normalize(signal, alpha=1.0)
    result.assert_true(normalized.dtype == np.float32, "Output is float32")

    return result


# ============================================================================
# Test 6: Full DSP Pipeline
# ============================================================================

def test_full_pipeline():
    print("\n" + "=" * 80)
    print("TEST 6: FULL DSP PIPELINE")
    print("=" * 80)

    result = TestResult()

    # Reset streaming buffer before tests
    reset_streaming_buffer()

    # Test 6.1: Latency budget compliance
    print("\n[Test 6.1] Latency budget compliance (<20ms)")
    raw_chunk = generate_synthetic_chunk(
        duration_ms=50.0,
        sample_rate=40000,
        noise_level=0.5,
        drift_severity=1.5,
        spike_rate=20.0,
        seed=42
    )

    config = {
        'moving_avg_window': 500,
        'tanh_alpha': 1.0,
        'spike_threshold': 30.0
    }

    cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)
    result.assert_true(latency_ms < 20.0, f"Pipeline latency {latency_ms:.2f}ms < 20ms")

    # Test 6.2: Output bounds
    print("\n[Test 6.2] Output bounds [-1, 1]")
    result.assert_true(np.all(cleaned_tensor >= -1.0) and np.all(cleaned_tensor <= 1.0),
                      "Cleaned tensor within [-1, 1] bounds")

    # Test 6.3: Output shape
    print("\n[Test 6.3] Output shape preservation")
    result.assert_true(len(cleaned_tensor) == len(raw_chunk), "Output length matches input")

    # Test 6.4: Output type
    print("\n[Test 6.4] Output type (float32)")
    result.assert_true(cleaned_tensor.dtype == np.float32, "Output is float32")

    # Test 6.5: Metadata completeness
    print("\n[Test 6.5] Metadata completeness")
    required_keys = ['spike_count', 'variance', 'mean', 'has_nan', 'has_inf']
    for key in required_keys:
        result.assert_true(key in metadata, f"Metadata contains '{key}'")

    # Test 6.6: NaN/Inf detection
    print("\n[Test 6.6] NaN/Inf detection")
    result.assert_true(metadata['has_nan'] == False, "No NaN values in clean data")
    result.assert_true(metadata['has_inf'] == False, "No Inf values in clean data")

    # Test 6.7: Variance is reasonable
    print("\n[Test 6.7] Variance is reasonable")
    result.assert_range(metadata['variance'], 0.0, 1.0, "Variance within [0, 1] for normalized signal")

    # Test 6.8: Spike count is physically plausible
    print("\n[Test 6.8] Spike count is physically plausible")
    # For 50ms at ~20Hz firing rate, expect 0-5 actual spikes (Poisson distribution)
    # With derivative detection (threshold crossings), we count multiple crossings per spike
    # plus some noise. Expect < 500 crossings (vs 1000s with wrong threshold).
    result.assert_true(metadata['spike_count'] < 500,
                      f"Spike count {metadata['spike_count']} is plausible (<500 for 50ms)")

    return result


# ============================================================================
# Test 7: Metrics Engine
# ============================================================================

def test_metrics_engine():
    print("\n" + "=" * 80)
    print("TEST 7: METRICS ENGINE")
    print("=" * 80)

    result = TestResult()

    # Test 7.1: Signal yield calculation
    print("\n[Test 7.1] Signal yield calculation")
    # Create ideal conditions: good variance, reasonable spike count, low mean
    ideal_signal = np.random.randn(2000) * 0.4  # Variance ~0.16
    ideal_metadata = {
        'variance': 0.15,
        'mean': 0.02,
        'spike_count': 2,
    }
    yield_pct = calculate_signal_yield(ideal_signal, ideal_metadata['spike_count'], ideal_metadata)
    result.assert_range(yield_pct, 70.0, 100.0, "Ideal conditions give high yield (70-100%)")

    # Test 7.2: Poor quality detection
    print("\n[Test 7.2] Poor quality detection")
    poor_signal = np.random.randn(2000) * 2.0  # Very high variance
    poor_metadata = {
        'variance': 0.9,
        'mean': 0.5,
        'spike_count': 0,
    }
    yield_pct = calculate_signal_yield(poor_signal, poor_metadata['spike_count'], poor_metadata)
    result.assert_range(yield_pct, 0.0, 50.0, "Poor conditions give low yield (0-50%)")

    # Test 7.3: Channel mapping
    print("\n[Test 7.3] Active channel mapping")
    active_100 = calculate_active_channels(100.0, total_channels=10000)
    active_50 = calculate_active_channels(50.0, total_channels=10000)
    active_0 = calculate_active_channels(0.0, total_channels=10000)

    result.assert_range(active_100, 9700, 10000, "100% yield -> ~9850 active channels")
    result.assert_range(active_50, 4500, 5500, "50% yield -> ~5000 active channels")
    result.assert_range(active_0, 0, 500, "0% yield -> ~0 active channels")

    # Test 7.4: System health status
    print("\n[Test 7.4] System health status")
    # Create a tanh-normalized signal (all values within [-1, 1])
    healthy_signal = np.tanh(np.random.randn(2000) * 0.5).astype(np.float32)
    healthy_metadata = {'variance': 0.15, 'mean': 0.02, 'has_nan': False, 'has_inf': False}
    status = check_system_health(healthy_signal, healthy_metadata, yield_pct=95.0)
    result.assert_true(status == "healthy", "High yield (95%) -> healthy status")

    critical_metadata = {'variance': 0.9, 'mean': 0.5, 'has_nan': False, 'has_inf': False}
    status = check_system_health(healthy_signal, critical_metadata, yield_pct=30.0)
    result.assert_true(status == "critical", "Low yield (30%) -> critical status")

    # Test 7.5: Stability tracker
    print("\n[Test 7.5] Stability tracker")
    tracker = StabilityTracker(max_history=100)

    # Add some yields
    for i in range(50):
        tracker.add_yield(90.0 + np.random.randn() * 2.0)  # ~90% with small variation

    stability_index, stability_variance = tracker.calculate_stability_index(window_size=50)
    result.assert_range(stability_index, 85.0, 95.0, "Stability index reflects input (~90%)")
    result.assert_true(stability_variance < 5.0, "Low variance indicates stable performance")

    return result


# ============================================================================
# Test 8: Storage Manager
# ============================================================================

def test_storage_manager():
    print("\n" + "=" * 80)
    print("TEST 8: STORAGE MANAGER")
    print("=" * 80)

    result = TestResult()

    # Test 8.1: Basic storage operations
    print("\n[Test 8.1] Basic storage operations")
    storage = InMemoryStorage()

    tensor = np.random.randn(2000).astype(np.float32)
    metadata = {'spike_count': 5, 'variance': 0.2}

    storage.save_tensor(tensor, yield_pct=85.0, metadata=metadata)
    result.assert_true(storage.get_epoch_count() == 1, "Epoch count increments")

    # Test 8.2: History retrieval
    print("\n[Test 8.2] History retrieval")
    for i in range(10):
        storage.save_tensor(tensor, yield_pct=80.0 + i, metadata=metadata)

    yield_history = storage.get_yield_history(max_count=5)
    result.assert_true(len(yield_history) == 5, "Retrieved last 5 yields")
    result.assert_close(yield_history[-1], 89.0, 0.01, "Most recent yield is 89.0")

    # Test 8.3: Latest tensor retrieval
    print("\n[Test 8.3] Latest tensor retrieval")
    latest = storage.get_latest_tensor()
    result.assert_true(latest is not None, "Latest tensor retrieved")
    result.assert_true(len(latest) == 2000, "Latest tensor has correct length")

    # Test 8.4: Reset functionality
    print("\n[Test 8.4] Reset functionality")
    storage.reset()
    result.assert_true(storage.get_epoch_count() == 0, "Epoch count resets to 0")
    result.assert_true(storage.get_latest_tensor() is None, "No tensor after reset")

    return result


# ============================================================================
# Test 9: End-to-End Integration
# ============================================================================

def test_end_to_end():
    print("\n" + "=" * 80)
    print("TEST 9: END-TO-END INTEGRATION")
    print("=" * 80)

    result = TestResult()

    reset_streaming_buffer()
    storage = InMemoryStorage()
    tracker = StabilityTracker(max_history=100)

    print("\n[Test 9.1] Process 10 consecutive chunks")

    config = {
        'moving_avg_window': 500,
        'tanh_alpha': 1.0,
        'spike_threshold': 30.0  # Properly calibrated threshold for realistic spike counts
    }

    for i in range(10):
        # Generate chunk
        raw_chunk = generate_synthetic_chunk(
            duration_ms=50.0,
            sample_rate=40000,
            noise_level=0.3,
            drift_severity=1.0,
            spike_rate=20.0,
            seed=42 + i
        )

        # Process through pipeline
        cleaned_tensor, latency_ms, metadata = process_signal_streaming(raw_chunk, config)

        # Calculate metrics
        yield_pct = calculate_signal_yield(cleaned_tensor, metadata['spike_count'], metadata)
        active_channels = calculate_active_channels(yield_pct, total_channels=10000)
        health_status = check_system_health(cleaned_tensor, metadata, yield_pct)

        # Store results
        storage.save_tensor(cleaned_tensor, yield_pct, metadata)
        tracker.add_yield(yield_pct)

        # Verify each iteration
        result.assert_true(latency_ms < 20.0, f"Chunk {i+1}: Latency {latency_ms:.2f}ms < 20ms")

    # Test 9.2: Verify complete processing
    print("\n[Test 9.2] Verify complete processing")
    result.assert_true(storage.get_epoch_count() == 10, "All 10 chunks processed")

    yield_history = storage.get_yield_history(max_count=10)
    result.assert_true(len(yield_history) == 10, "Complete yield history stored")

    stability_index, stability_variance = tracker.calculate_stability_index(window_size=10)
    result.assert_range(stability_index, 0.0, 100.0, "Stability index in valid range")

    return result


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    print("\n" + "=" * 80)
    print("AXOFT SIGNAL YIELD GATEWAY - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing all components: Simulator, DSP, Metrics, Storage, Integration")
    print("=" * 80)

    all_results = []

    # Run all test suites
    all_results.append(test_data_simulator())
    all_results.append(test_circular_buffer())
    all_results.append(test_moving_average_subtraction())
    all_results.append(test_spike_detection())
    all_results.append(test_tanh_normalization())
    all_results.append(test_full_pipeline())
    all_results.append(test_metrics_engine())
    all_results.append(test_storage_manager())
    all_results.append(test_end_to_end())

    # Aggregate results
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed} ({100*total_passed/total_tests:.1f}%)")
    print(f"Failed: {total_failed} ({100*total_failed/total_tests:.1f}%)")

    if total_failed > 0:
        print("\n" + "!" * 80)
        print("SOME TESTS FAILED - REVIEW FAILURES ABOVE")
        print("!" * 80)
        return False
    else:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED - SYSTEM VERIFIED [SUCCESS]")
        print("=" * 80)
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
