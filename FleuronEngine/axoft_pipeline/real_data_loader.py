"""
Real Neural Data Loader for Axoft Signal Processing Pipeline

Loads PhysioNet EEG Motor Movement/Imagery Dataset (or other EDF files)
and provides chunks in the same format as the synthetic data generator.

This demonstrates the pipeline working on REAL neural recordings,
not just simulated data.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
Format: EDF (European Data Format)
Channels: 64 EEG electrodes
Sampling Rate: 160 Hz (will be resampled to match pipeline expectations)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import warnings


class RealDataLoader:
    """
    Loads and streams real neural data from EDF files.

    Provides the same interface as the synthetic data generator
    so it can be dropped into the existing pipeline.
    """

    def __init__(self, data_dir: str = "../data/physionet"):
        """
        Initialize the real data loader.

        Args:
            data_dir: Path to directory containing EDF files
        """
        self.data_dir = Path(data_dir)
        self.data = None
        self.sampling_rate = None
        self.channel_names = None
        self.current_position = 0
        self.current_channel = 0
        self._loaded = False

    def load_file(self, filename: str, channel: int = 0) -> bool:
        """
        Load a specific EDF file.

        Args:
            filename: Name of the EDF file (e.g., "S001R03.edf")
            channel: Which channel to use (0-63 for PhysioNet data)

        Returns:
            True if successful, False otherwise
        """
        try:
            import pyedflib

            filepath = self.data_dir / filename
            if not filepath.exists():
                print(f"File not found: {filepath}")
                return False

            # Load EDF file using pyedflib
            edf = pyedflib.EdfReader(str(filepath))

            # Get file info
            n_channels = edf.signals_in_file
            self.channel_names = edf.getSignalLabels()
            self.sampling_rate = edf.getSampleFrequency(0)

            # Get data for specified channel
            if channel >= n_channels:
                channel = 0

            self.current_channel = channel
            self.data = edf.readSignal(channel)

            # Close file
            edf.close()

            # Convert from Volts to microvolts for consistency
            # PhysioNet EEG data is typically in microvolts already
            # but some files may be in volts - check magnitude
            if np.abs(self.data).max() < 0.01:  # Likely in volts
                self.data = self.data * 1e6

            self.current_position = 0
            self._loaded = True

            return True

        except ImportError:
            print("pyedflib library not installed. Run: pip install pyedflib")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def get_available_files(self) -> List[str]:
        """Get list of available EDF files in data directory."""
        if not self.data_dir.exists():
            return []
        return sorted([f.name for f in self.data_dir.glob("*.edf")])

    def get_channel_names(self) -> List[str]:
        """Get list of channel names from loaded file."""
        return self.channel_names if self.channel_names else []

    def get_chunk(self, chunk_duration_ms: float = 50.0, native_rate: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Get next chunk of real neural data.

        **UPDATED**: Now returns data at NATIVE sample rate by default.
        This is critical for proper EEG processing - the IIR highpass filter
        needs to know the actual sample rate, not an artificial upsampled rate.

        Args:
            chunk_duration_ms: Chunk duration in milliseconds (default 50ms)
            native_rate: If True (default), return at native 160Hz sample rate.
                        If False, upsample to 2000 samples (legacy behavior).

        Returns:
            Tuple of (signal_chunk, metadata_dict)
        """
        if not self._loaded or self.data is None:
            # Return zeros if no data loaded
            fallback_samples = int(chunk_duration_ms * 160 / 1000) if native_rate else 2000
            return np.zeros(fallback_samples), {"source": "no_data", "signal_type": "unknown"}

        # Calculate how many samples for this duration at native rate
        native_samples = int(chunk_duration_ms * self.sampling_rate / 1000.0)
        native_samples = max(1, native_samples)

        # Handle wraparound
        if self.current_position + native_samples > len(self.data):
            self.current_position = 0

        # Extract chunk from real data at native sample rate
        raw_chunk = self.data[self.current_position:self.current_position + native_samples]
        self.current_position += native_samples

        if native_rate:
            # Return at native sample rate (recommended for EEG mode)
            chunk = raw_chunk.astype(np.float32)
        else:
            # Legacy: Upsample to 2000 samples for compatibility
            target_samples = 2000
            x_old = np.linspace(0, 1, len(raw_chunk))
            x_new = np.linspace(0, 1, target_samples)
            chunk = np.interp(x_new, x_old, raw_chunk).astype(np.float32)

        metadata = {
            "source": "real_physionet",
            "signal_type": "eeg",  # Critical: tells pipeline to use EEG mode
            "channel": self.channel_names[self.current_channel] if self.channel_names else "unknown",
            "sample_rate": self.sampling_rate,  # Native sample rate
            "original_sample_rate": self.sampling_rate,  # Backward compat
            "position_seconds": self.current_position / self.sampling_rate,
            "chunk_duration_ms": chunk_duration_ms,
            "native_rate": native_rate
        }

        return chunk, metadata

    def reset(self):
        """Reset to beginning of data."""
        self.current_position = 0

    def get_info(self) -> dict:
        """Get information about loaded data."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "sampling_rate": self.sampling_rate,
            "num_channels": len(self.channel_names) if self.channel_names else 0,
            "duration_seconds": len(self.data) / self.sampling_rate if self.data is not None else 0,
            "current_channel": self.channel_names[self.current_channel] if self.channel_names else "none"
        }


def generate_real_chunk(
    loader: RealDataLoader,
    chunk_duration_ms: float = 50.0,
    add_synthetic_drift: bool = False,
    drift_amplitude: float = 0.0,
    native_rate: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Generate a chunk from real data, optionally adding synthetic drift.

    **UPDATED**: Now returns full metadata dict (not just spike count).
    This is critical for the dual-mode pipeline which needs sample_rate
    and signal_type from the metadata.

    Args:
        loader: Initialized RealDataLoader with file loaded
        chunk_duration_ms: Duration of chunk in milliseconds (default 50ms)
        add_synthetic_drift: Whether to add artificial baseline drift
        drift_amplitude: Amplitude of synthetic drift to add (microvolts)
        native_rate: If True, return at native 160Hz. If False, upsample.

    Returns:
        Tuple of (signal_chunk, metadata_dict)
        metadata_dict includes: signal_type, sample_rate, channel, etc.
    """
    chunk, metadata = loader.get_chunk(chunk_duration_ms, native_rate=native_rate)

    # Optionally add synthetic drift to demonstrate drift removal
    if add_synthetic_drift and drift_amplitude > 0:
        t = np.linspace(0, 1, len(chunk))
        drift = drift_amplitude * np.sin(2 * np.pi * t)  # 1 Hz drift
        chunk = chunk + drift

    # Estimate spike count using derivative method (same as synthetic)
    derivative = np.abs(np.diff(chunk))
    threshold = np.mean(derivative) + 2 * np.std(derivative)
    spike_count = int(np.sum(derivative > threshold) / 10)  # Rough estimate

    # Add spike count to metadata
    metadata['spike_count_estimate'] = spike_count

    return chunk, metadata


# Convenience function for quick testing
def test_loader():
    """Quick test of the data loader."""
    loader = RealDataLoader("../data/physionet")

    files = loader.get_available_files()
    print(f"Available files: {files}")

    if files:
        success = loader.load_file(files[0])
        print(f"Load success: {success}")

        if success:
            info = loader.get_info()
            print(f"Data info: {info}")

            # Test native rate mode (new default)
            chunk, meta = loader.get_chunk(chunk_duration_ms=50.0, native_rate=True)
            print(f"\nNative rate mode:")
            print(f"  Chunk shape: {chunk.shape} (should be ~8 samples for 50ms @ 160Hz)")
            print(f"  Sample rate: {meta['sample_rate']} Hz")
            print(f"  Signal type: {meta['signal_type']}")
            print(f"  Chunk stats: min={chunk.min():.2f}, max={chunk.max():.2f}, std={chunk.std():.2f}")

            # Test generate_real_chunk
            chunk2, meta2 = generate_real_chunk(loader, chunk_duration_ms=50.0, native_rate=True)
            print(f"\ngenerate_real_chunk test:")
            print(f"  Chunk shape: {chunk2.shape}")
            print(f"  Metadata keys: {list(meta2.keys())}")


if __name__ == "__main__":
    test_loader()
