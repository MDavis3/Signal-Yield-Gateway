# Axoft Signal Yield & Clinical Translation Gateway - Design Document

**Date:** 2026-03-02
**Author:** Lead Systems Architect
**Reviewer:** Senior AI Software Engineer & BCI Data Architect
**Status:** Approved

## Executive Summary

This document specifies the architecture for a Signal Yield & Clinical Translation Gateway prototype for Axoft's flexible polymer BCI electrodes. The system addresses the micromotion-induced baseline drift problem through thermally-constrained O(1) DSP operations while providing dual personas (R&D and Clinical/FDA) for stakeholder communication.

## Problem Statement

Axoft's ultra-soft polymer electrodes solve brain scarring (gliosis) but suffer from physical micromotion due to heartbeat and respiration. This creates:
- Severe low-frequency baseline drift in raw neural data
- Variable spike amplitudes as electrodes move relative to neurons
- Incompatible data format for TN-VAE latent-space decoders

**Critical Constraints:**
- Thermal budget: <1°C heat increase (tissue necrosis prevention)
- Latency budget: <20ms per chunk processing
- Computational complexity: O(1) or highly efficient linear time only

## Architecture Overview

### Module Structure

```
axoft_pipeline/
├── __init__.py                 # Package initialization
├── dsp_pipeline.py            # O(1) signal processing functions
├── data_simulator.py          # Synthetic hardware data generation
├── metrics_engine.py          # FDA/clinical business logic
├── storage_manager.py         # Backend abstraction layer
├── app.py                     # Streamlit UI (< 200 lines)
└── requirements.txt           # Dependencies
```

### Module Responsibilities

**dsp_pipeline.py** - Core Signal Processing (~150 lines)
- `moving_average_subtract()`: O(1) amortized baseline drift removal via circular buffer
- `detect_spikes_derivative()`: O(n) derivative-based action potential detection
- `tanh_normalize()`: O(n) soft-clipping normalization to [-1, 1]
- `process_signal()`: Main pipeline orchestrator with latency tracking

**data_simulator.py** - Mock Hardware (~100 lines)
- `generate_synthetic_chunk()`: Generates 40kHz mock data with Gaussian noise, injected spikes, and sinusoidal micromotion drift
- Returns numpy array shaped (n_samples,) as float32

**metrics_engine.py** - Clinical Translation (~200 lines)
- `calculate_signal_yield()`: Multi-factor 0-100% score (variance health + spike rate + amplitude stability)
- `calculate_chronic_stability_index()`: Rolling average of yield over configurable epochs
- `calculate_active_channels()`: Maps yield % to simulated channel dropout
- `check_system_health()`: Returns health status based on anomaly detection

**storage_manager.py** - Abstraction Layer (~80 lines)
- `StorageManager` abstract base class
- `InMemoryStorage` implementation (default for prototype)
- `RedisStorage` stub (production integration path)
- Swappable backend without touching other modules

**app.py** - UI Layer (~180 lines)
- Streamlit dashboard with dual persona views
- R&D View: raw vs cleaned waveforms, latency metrics
- Clinical View: KPI cards, chronic stability trend, system health
- Play/Pause/Step controls for presentation

## Data Flow

### Processing Sequence (Single 50ms Chunk)

1. **Data Generation**: `data_simulator.generate_synthetic_chunk()` → 2000 samples @ 40kHz
2. **DSP Pipeline**: `dsp_pipeline.process_signal()`
   - **Step 2a**: Moving average subtraction (circular buffer, O(1) amortized)
   - **Step 2b**: Spike detection via derivatives (O(n) linear pass)
   - **Step 2c**: Tanh normalization (O(n) vectorized, soft-clip to [-1, 1])
   - **Step 2d**: Latency measurement
3. **Metrics Calculation**: `metrics_engine.calculate_signal_yield()` → composite score
4. **Channel Mapping**: `metrics_engine.calculate_active_channels()` → yield to dropout
5. **Persistence**: `storage_manager.save_tensor()` → backend storage
6. **Stability Tracking**: `metrics_engine.calculate_chronic_stability_index()` → rolling average
7. **Health Check**: `metrics_engine.check_system_health()` → status flag
8. **Rendering**: `app.py` displays persona-specific UI

## Key Design Decisions

### Why Circular Buffer for Moving Average?
- Maintains fixed-size window without array shifts
- O(1) amortized: add new sample, subtract oldest, update sum
- Avoids scipy.signal.butter (requires FFT, thermal budget killer)

### Why Tanh Instead of Hard Clipping?
- Differentiable (critical for future gradient-based ML decoders)
- Smooth transition prevents high-frequency artifacts
- `np.tanh()` is hardware-accelerated on most CPUs
- Bounds guaranteed [-1, 1] without conditional logic

### Why Derivative-Based Spike Detection?
- Action potentials have characteristic sharp rising edge (~1ms)
- `np.diff()` is O(n) vectorized, thermally cheap
- More robust than amplitude threshold alone (fails as electrode drifts)

### Why Modular Architecture Over Monolith?
- Demonstrates production-grade separation of concerns to Axoft
- DSP functions can be integrated directly into firmware/ML pipelines
- Easy to unit test each component independently
- Clean abstraction allows Redis swap for production deployment

## UI/UX Design

### Sidebar Controls
- Signal parameters: micromotion drift, noise level, tanh alpha
- DSP parameters: moving avg window (100-2000 samples), stability window (10-200 epochs)
- View persona toggle: R&D Engineer vs Clinical/FDA
- Playback controls: Play/Pause/Step for presentation

### R&D Engineer View
- Pipeline latency metric card
- Dual-trace Plotly chart: raw (red) vs cleaned (cyan) waveforms
- Visible proof of baseline drift removal and spike preservation

### Clinical/FDA View
- KPI metric cards: Live Signal Yield %, Active Channels (out of 10,000), System Uptime
- Chronic Stability Index chart with ±2σ variance envelope
- System health indicator (✅ Healthy | ⚠️ Warning | 🔴 Critical)
- Business value proposition: No manual recalibration needed

### Interaction Behaviors
- **Play mode**: Auto-generates chunk every 50ms, real-time updates
- **Pause mode**: Freezes state for parameter tweaking
- **Step mode**: Single chunk generation for frame-by-frame explanation
- **Slider changes**: Immediate effect on next chunk, no reload
- **View toggle**: Instant persona switch, no data loss

## Multi-Factor Signal Yield Scoring

Composite score combining three factors:

1. **Variance Health (40% weight)**: Checks if cleaned signal variance is within healthy bounds
2. **Spike Rate (30% weight)**: Validates detected spike count is within physiological range
3. **Amplitude Stability (30% weight)**: Ensures amplitude variance is low (electrode position stable)

Formula: `yield_pct = 0.4 * variance_score + 0.3 * spike_score + 0.3 * stability_score`

## Channel Dropout Mapping

Links Signal Yield % to Active Channel Count to demonstrate clinical impact:
- 100% yield → 9,850 ± 50 active channels (out of 10,000)
- 50% yield → ~5,000 active channels
- 0% yield → 0 active channels

This visually demonstrates the real-world consequences of micromotion on clinical viability.

## Error Handling Strategy

### System Health Indicator
- **Healthy**: All metrics nominal, processing successful
- **Warning**: Detected NaN/inf values, recovered with fallback, yield 50-80%
- **Critical**: Processing failure, yield <50%, potential hardware fault

### Medical-Grade Continuous Operation
- Never crash during data anomalies
- Flag errors to clinician via health indicator
- Maintain pipeline uptime for continuous monitoring

## Deployment Configuration

### Storage Backend Selection
```python
# storage_manager.py configuration
STORAGE_BACKEND = "in_memory"  # Options: "in_memory" | "redis"
```

For production deployment, swap to `"redis"` without modifying other modules.

## Success Criteria

1. **Thermal Compliance**: Pipeline latency <20ms consistently
2. **Drift Removal**: Raw baseline swing ±500μV → Cleaned baseline flat at 0μV
3. **Spike Preservation**: Neural action potentials visible in cleaned trace
4. **Normalization**: All cleaned values strictly within [-1, 1] bounds
5. **Clinical Translation**: Stability index remains >90% over 200 epochs
6. **Presentation Value**: Play/Pause/Step controls enable frame-by-frame explanation

## Future Extensions

1. **Redis Integration**: Uncomment `RedisStorage` class, deploy backend
2. **Multi-Channel Processing**: Extend to 10,000 parallel channels
3. **TN-VAE Integration**: Pipe cleaned tensors to latent-space decoder
4. **Real Hardware Interface**: Replace simulator with actual Axoft electrode data stream
5. **Adaptive Parameters**: Auto-tune tanh alpha and moving avg window based on drift detection

## References

- Axoft Hardware Specifications: Flexible Polymer BCI Electrodes
- Thermal Constraints: <1°C tissue temperature increase limit
- FDA Statistical Requirements: ±2σ variance envelope for stability demonstration
- TN-VAE Decoder Requirements: Float32 tensors normalized to [-1, 1]

---

**Design Approved:** 2026-03-02
**Ready for Implementation:** ✅
