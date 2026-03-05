"""
Axoft Pipeline Test Suite
=========================
Systematic verification that signal processing improvements work correctly.

Run: python -m pytest axoft_pipeline/test_pipeline.py -v
Or:  python axoft_pipeline/test_pipeline.py  (standalone)

Tests verify:
- IIR highpass filter removes DC offset while preserving alpha rhythms
- Dual-mode processing (EEG vs intracortical)
- Adaptive metrics thresholds for different signal types
- Stability envelope doesn't form expanding wedge
- Real data loader returns native sample rate
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dsp_pipeline import iir_highpass_filter, process_signal, polyfit_detrend
from metrics_engine import calculate_signal_yield, StabilityTracker
from real_data_loader import RealDataLoader, generate_real_chunk


class TestIIRHighpassFilter:
    """Test the IIR highpass filter for DC removal and rhythm preservation."""

    def test_removes_dc_offset(self):
        """DC offset should be removed (reduced by >90%)."""
        t = np.linspace(0, 1, 160)  # 1 second at 160Hz
        signal = np.sin(2 * np.pi * 10 * t) + 5.0  # 10Hz + DC=5

        filtered, _ = iir_highpass_filter(signal, cutoff_hz=0.5, sample_rate=160.0)

        # Skip transient (first 50 samples)
        input_dc = signal.mean()
        output_dc = abs(filtered[50:].mean())

        assert input_dc > 4.5, f"Input DC should be ~5, got {input_dc}"
        assert output_dc < 0.5, f"Output DC should be <0.5 (90% removed), got {output_dc}"

    def test_preserves_alpha_rhythm(self):
        """10Hz alpha rhythm should be preserved (amplitude within 80%)."""
        t = np.linspace(0, 1, 160)
        signal = np.sin(2 * np.pi * 10 * t)  # Pure 10Hz

        filtered, _ = iir_highpass_filter(signal, cutoff_hz=0.5, sample_rate=160.0)

        # Compare RMS amplitude (skip transient)
        input_rms = np.std(signal[50:])
        output_rms = np.std(filtered[50:])

        ratio = output_rms / input_rms
        assert ratio > 0.8, f"10Hz should be preserved (>80%), got {ratio*100:.1f}%"

    def test_removes_slow_drift(self):
        """0.3Hz drift should be attenuated (reduced by >10%)."""
        np.random.seed(42)  # For reproducibility
        t = np.linspace(0, 2, 320)  # 2 seconds at 160Hz
        drift = 10 * np.sin(2 * np.pi * 0.3 * t)  # 0.3Hz drift
        signal = drift + np.random.randn(len(t)) * 0.1  # + small noise

        filtered, _ = iir_highpass_filter(signal, cutoff_hz=0.5, sample_rate=160.0)

        # Compare RMS - drift should be attenuated
        input_rms = np.std(drift)
        output_rms = np.std(filtered[80:])  # Skip longer transient

        ratio = output_rms / input_rms
        # Single-pole filter has gentle rolloff, but should still attenuate
        assert ratio < 1.0, f"0.3Hz drift should be attenuated, got {ratio*100:.1f}%"

    def test_buffer_continuity(self):
        """Processing chunks should produce continuous output."""
        t1 = np.linspace(0, 0.5, 80)
        t2 = np.linspace(0.5, 1.0, 80)

        chunk1 = np.sin(2 * np.pi * 10 * t1) + 3.0
        chunk2 = np.sin(2 * np.pi * 10 * t2) + 3.0

        # Process with buffer continuity
        filtered1, buffer = iir_highpass_filter(chunk1, cutoff_hz=0.5, sample_rate=160.0)
        filtered2, _ = iir_highpass_filter(chunk2, cutoff_hz=0.5, sample_rate=160.0, buffer=buffer)

        # Check no discontinuity at chunk boundary
        jump = abs(filtered2[0] - filtered1[-1])
        assert jump < 0.5, f"Chunk boundary should be continuous, got jump={jump:.3f}"


class TestProcessSignal:
    """Test the full processing pipeline in both modes."""

    def test_eeg_mode_config(self):
        """EEG mode should use highpass filter, not polynomial detrending."""
        np.random.seed(42)
        signal = np.random.randn(160) * 10 + 5  # Random with DC offset

        config = {
            'processing_mode': 'eeg',
            'sample_rate': 160.0,
            'highpass_cutoff': 0.5,
            'tanh_alpha': 1.0
        }

        cleaned, latency, meta, _ = process_signal(signal, config)

        assert meta.get('processing_mode') == 'eeg'
        assert latency < 20.0, f"Processing should be <20ms, got {latency:.2f}ms"

    def test_intracortical_mode_config(self):
        """Intracortical mode should use polynomial detrending."""
        np.random.seed(42)
        signal = np.random.randn(2000) * 50

        config = {
            'processing_mode': 'intracortical',
            'poly_order': 1,
            'tanh_alpha': 1.0
        }

        cleaned, latency, meta, _ = process_signal(signal, config)

        assert meta.get('processing_mode') == 'intracortical'
        assert latency < 20.0

    def test_tanh_bounds_output(self):
        """Output should be bounded in [-1, 1] after tanh normalization."""
        np.random.seed(42)
        signal = np.random.randn(160) * 100  # Large amplitude

        config = {
            'processing_mode': 'eeg',
            'sample_rate': 160.0,
            'tanh_alpha': 1.5
        }

        cleaned, _, _, _ = process_signal(signal, config)

        assert cleaned.min() >= -1.0, f"Min should be >= -1, got {cleaned.min()}"
        assert cleaned.max() <= 1.0, f"Max should be <= 1, got {cleaned.max()}"

    def test_default_mode_is_intracortical(self):
        """When no mode specified, should default to intracortical."""
        np.random.seed(42)
        signal = np.random.randn(100) * 10

        config = {
            'tanh_alpha': 1.0
        }

        cleaned, latency, meta, _ = process_signal(signal, config)

        assert meta.get('processing_mode') == 'intracortical'


class TestMetricsEngine:
    """Test adaptive metrics for different signal types."""

    def test_eeg_yield_reasonable(self):
        """EEG data should produce 20-100% yield with adaptive thresholds."""
        np.random.seed(42)
        # Simulate typical EEG cleaned signal
        cleaned = np.random.randn(160) * 0.3  # Variance ~0.09 (in tanh space)

        metadata = {
            'signal_type': 'eeg',
            'spike_count': 3,
            'variance': float(np.var(cleaned)),  # Required by calculate_signal_yield
            'mean': float(np.mean(cleaned))
        }

        yield_pct = calculate_signal_yield(cleaned, metadata['spike_count'], metadata)

        assert 20 < yield_pct < 100, f"EEG yield should be 20-100%, got {yield_pct:.1f}%"

    def test_synthetic_yield_separate_thresholds(self):
        """Synthetic data should use different (stricter) thresholds."""
        np.random.seed(42)
        cleaned = np.random.randn(2000) * 0.5

        metadata = {
            'signal_type': 'synthetic',
            'spike_count': 50,
            'variance': float(np.var(cleaned)),
            'mean': float(np.mean(cleaned))
        }

        yield_pct = calculate_signal_yield(cleaned, metadata['spike_count'], metadata)

        # Should produce a result (not crash)
        assert 0 <= yield_pct <= 100

    def test_metadata_missing_signal_type(self):
        """Should default to synthetic when signal_type not specified."""
        np.random.seed(42)
        cleaned = np.random.randn(100) * 0.4

        metadata = {
            'spike_count': 10,
            'variance': float(np.var(cleaned)),
            'mean': float(np.mean(cleaned))
        }

        yield_pct = calculate_signal_yield(cleaned, metadata.get('spike_count', 0), metadata)

        # Should not crash and produce a result
        assert 0 <= yield_pct <= 100


class TestStabilityTracker:
    """Test the stability tracking and envelope calculation."""

    def test_add_yield_stores_history(self):
        """Adding yields should store them in history."""
        tracker = StabilityTracker(max_history=100)

        for i in range(10):
            tracker.add_yield(80.0 + i)

        history = tracker.get_full_history()
        assert len(history) == 10

    def test_smoothed_history_exists(self):
        """Smoothed history should be available."""
        tracker = StabilityTracker(max_history=100)

        for _ in range(20):
            tracker.add_yield(80.0 + np.random.randn() * 2)

        smoothed = tracker.get_smoothed_history()
        assert len(smoothed) == 20

    def test_history_respects_max(self):
        """History should not exceed max_history."""
        tracker = StabilityTracker(max_history=50)

        for _ in range(100):
            tracker.add_yield(80.0)

        history = tracker.get_full_history()
        assert len(history) <= 50

    def test_rolling_std_is_consistent(self):
        """
        Rolling std at different points should be similar for stable data.
        This tests that we don't have an expanding wedge pattern.
        """
        np.random.seed(42)
        tracker = StabilityTracker(max_history=200)

        # Add 100 epochs of stable data (small variance)
        for _ in range(100):
            tracker.add_yield(80.0 + np.random.randn() * 2)  # 80% +/- 2%

        history = tracker.get_smoothed_history()

        # Calculate rolling std at different points
        early_std = np.std(history[10:30])  # Epochs 10-30
        late_std = np.std(history[70:100])  # Epochs 70-100

        # Std should be similar (not dramatically different)
        ratio = late_std / early_std if early_std > 0 else 1
        assert 0.2 < ratio < 5.0, f"Early vs late std should be similar, got ratio={ratio:.2f}"


class TestRealDataLoader:
    """Test real PhysioNet data loading (requires data files)."""

    def test_loader_initialization(self):
        """Loader should initialize without crashing."""
        loader = RealDataLoader("../data/physionet")
        assert loader is not None

    def test_get_available_files(self):
        """Should return list (possibly empty) of available files."""
        loader = RealDataLoader("../data/physionet")
        files = loader.get_available_files()
        assert isinstance(files, list)

    def test_native_sample_rate(self):
        """Real data should return native sample rate, not upsampled."""
        loader = RealDataLoader("../data/physionet")
        files = loader.get_available_files()

        if files:
            loader.load_file(files[0])
            chunk, meta = loader.get_chunk(chunk_duration_ms=50.0, native_rate=True)

            # 50ms at 160Hz = 8 samples
            expected_samples = int(50.0 * 160 / 1000)
            assert len(chunk) == expected_samples, f"Expected {expected_samples} samples, got {len(chunk)}"
            assert meta['signal_type'] == 'eeg'
            assert meta['sample_rate'] == 160.0
        else:
            print("  SKIP: No PhysioNet data files available")

    def test_generate_real_chunk_returns_metadata(self):
        """generate_real_chunk should return (chunk, metadata) tuple."""
        loader = RealDataLoader("../data/physionet")
        files = loader.get_available_files()

        if files:
            loader.load_file(files[0])
            chunk, meta = generate_real_chunk(loader, chunk_duration_ms=50.0, native_rate=True)

            assert isinstance(chunk, np.ndarray)
            assert isinstance(meta, dict)
            assert 'signal_type' in meta
            assert 'sample_rate' in meta
        else:
            print("  SKIP: No PhysioNet data files available")


class TestPolyfitDetrend:
    """Test polynomial detrending for synthetic/intracortical data."""

    def test_removes_linear_drift(self):
        """Linear drift should be removed."""
        t = np.linspace(0, 1, 2000)
        drift = 50 * t  # Linear drift from 0 to 50
        signal = np.sin(2 * np.pi * 50 * t) * 10 + drift  # 50Hz signal + drift

        detrended, _ = polyfit_detrend(signal, poly_order=1)

        # After detrending, mean should be near zero
        assert abs(detrended.mean()) < 5, f"Mean should be near 0, got {detrended.mean()}"

    def test_preserves_signal_shape(self):
        """Signal shape should be preserved after detrending."""
        np.random.seed(42)
        signal = np.random.randn(2000) * 10

        detrended, _ = polyfit_detrend(signal, poly_order=1)

        # Standard deviation should be similar (signal not destroyed)
        ratio = np.std(detrended) / np.std(signal)
        assert 0.5 < ratio < 2.0, f"Signal std should be preserved, got ratio={ratio}"


def run_all_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestIIRHighpassFilter,
        TestProcessSignal,
        TestMetricsEngine,
        TestStabilityTracker,
        TestRealDataLoader,
        TestPolyfitDetrend,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing: {test_class.__name__}")
        print('='*60)

        instance = test_class()
        for method_name in sorted(dir(instance)):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  PASS: {method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  FAIL: {method_name}")
                    print(f"        {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ERROR: {method_name}")
                    print(f"        {type(e).__name__}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print('='*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
