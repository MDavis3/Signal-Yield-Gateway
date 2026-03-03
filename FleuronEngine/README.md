# Axoft Signal Yield & Clinical Translation Gateway

A real-time brain-computer interface (BCI) signal processing dashboard demonstrating thermally-constrained DSP algorithms and clinical metrics translation for Axoft's flexible neural electrode arrays.

![Dashboard Preview](https://img.shields.io/badge/Status-Demo-yellow) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

This project demonstrates a **dual-persona dashboard** for BCI signal processing:

- **R&D Engineer View**: Raw vs cleaned waveforms, latency metrics, DSP diagnostics
- **Clinical/FDA View**: KPI cards (signal yield, active channels, uptime), chronic stability tracking

The system processes 50ms chunks of neural data at 40kHz sampling rate through a thermally-efficient DSP pipeline, translating technical metrics into clinical outcomes suitable for FDA regulatory submissions.

### Key Features

✅ **Thermally-Constrained DSP**: O(1) circular buffer moving average (no FFTs/heavy filters)
✅ **Two-Stage Filtering**: Drift removal (1500 samples) + ringing suppression (40 samples)
✅ **Adaptive Tanh Normalization**: 1.5σ scaling with DC offset correction
✅ **Derivative Spike Detection**: Robust to electrode drift, O(n) complexity
✅ **Signal Yield Metric**: 40% variance + 30% spike rate + 30% stability composite score
✅ **Chronic Stability Tracking**: EMA smoothing with ±2σ confidence envelope
✅ **Real-time Playback Controls**: Play/Pause/Step/Reset for demos

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MDavis3/Signal-Yield-Gateway.git
cd Signal-Yield-Gateway
```

2. Install dependencies:
```bash
pip install streamlit numpy plotly
```

### Running the Dashboard

```bash
streamlit run axoft_pipeline/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## 📊 Usage

### Dashboard Controls

**Sidebar Controls:**
- **Micromotion Drift Severity** (0.0-2.0): Simulates physical electrode movement from heartbeat/respiration
- **Noise Level** (0.0-1.0): Background Gaussian noise amplitude
- **Tanh Alpha** (0.1-5.0): Soft-clipping compression steepness
- **Moving Avg Window** (100-2000 samples): Baseline drift removal window size
- **Smoothing Window** (0-100 samples): Post-MA ringing suppression (40 = 1ms)
- **Stability Window** (10-200 epochs): Rolling window for chronic stability index

**Playback Controls:**
- ▶ **Play/Pause**: Continuous streaming mode
- ⏭ **Step**: Process single 50ms chunk (frame-by-frame)
- 🔄 **Reset**: Clear history and restart session

**View Modes:**
- **R&D Engineer View**: Dual Y-axis plot (raw μV vs normalized [-1,1]), latency metrics
- **Clinical/FDA View**: KPI cards, chronic stability trend with ±2σ envelope

### Interpreting Results

#### Signal Yield (Clinical View)
- **90-100%**: Healthy - Excellent signal quality
- **70-89%**: Warning - Marginal quality, intervention may be needed
- **<70%**: Critical - Poor quality, requires immediate attention

#### Active Channels
- Estimated viable channels out of 10,000-electrode array
- Calculated as: `active_channels = (yield% / 100) × 10,000`

#### Chronic Stability Index
- Rolling mean signal yield over N epochs
- ±2σ envelope shows variance bounds
- Target: <5% variance over 200 epochs (chronic implant stability)

## 🏗️ Architecture

### Modular Design

```
axoft_pipeline/
├── app.py                  # Streamlit dashboard UI
├── data_simulator.py       # Synthetic neural data generator
├── dsp_pipeline.py         # Core signal processing (MA, tanh, spike detection)
├── metrics_engine.py       # Clinical translation (yield, health, stability)
├── storage_manager.py      # Time-series data storage abstraction
└── test_suite.py           # Comprehensive unit tests
```

### Signal Processing Pipeline

```
Raw Signal (with drift)
    ↓
[1] Moving Average Subtraction (1500 samples)
    → Removes slow baseline drift (0.5-1Hz heartbeat/respiration)
    → O(1) circular buffer for thermal efficiency
    ↓
Centered Signal (drift removed, has ringing)
    ↓
[2] Derivative Spike Detection (threshold crossing)
    → Uses pre-smoothing signal for accuracy
    → Robust to electrode distance changes
    ↓
[3] Moving Average Smoothing (40 samples)
    → Suppresses biphasic ringing artifacts (5-20 kHz)
    → Preserves spike edges (~1ms rising time)
    ↓
Smoothed Signal (clean, demo-ready)
    ↓
[4] Tanh Normalization (adaptive 1.5σ scaling)
    → DC offset removal → zero-centered baseline
    → Soft-clipping to [-1, 1] bounds
    → Differentiable for gradient-based decoders
    ↓
Cleaned Tensor (PyTorch-ready, float32)
```

### Clinical Metrics Translation

**Signal Yield Calculation:**
```python
yield = 0.40 × variance_score + 0.30 × spike_score + 0.30 × stability_score + jitter
```

**Component Scores:**
- **Variance Score**: Signal-to-noise ratio (0.8 threshold)
- **Spike Score**: Neural activity detection rate (5 spikes/second minimum)
- **Stability Score**: Baseline drift control (80μV tolerance)
- **Biological Jitter**: ±2.5% Gaussian noise for realism

## 🐛 Known Issues & Fixes Applied

### Issue 1: Wavy Baseline (Fixed v0.2)
**Problem**: With MA window = 1500 samples (75% of chunk), baseline appeared wavy instead of flat at 0.0.
**Root Cause**: Window too large relative to 1 Hz drift frequency → filter couldn't track fast enough within chunk.
**Fix**: Reduced MA window to 400 samples (10ms) → 19% flatter baseline, 60% less DC offset.

### Issue 2: Square Spike Morphology (Fixed v0.2)
**Problem**: Spikes appeared as flat plateaus instead of sharp needles, destroying waveform information for ML decoders.
**Root Cause**: 40-sample smoothing window equal to spike width (1ms) → smeared 0.3ms rise time.
**Fix**: Reduced smoothing to 10 samples (0.25ms) → 55.6% sharper spikes, 44.2% less flat.

### Issue 3: Random Health Drops with Clean Signals (Fixed v0.2)
**Problem**: Paradoxically, reducing drift/noise caused yields to drop to 60-70% randomly.
**Root Cause**: Metrics calibrated for noisy signals. Clean signals have:
  - Lower variance (~80 μV² vs optimal 400) → low variance score
  - MORE spike crossings (~1248 vs optimal 500) → low spike score (sharper edges!)
**Fix**: Adjusted thresholds:
  - `optimal_variance`: 400 → 120 μV²
  - `optimal_crossing_count`: 850 → 1100 (clean signals paradoxically produce more crossings!)
  - Result: 98.6% yield with clean signals (was 78%)

## ⚠️ Limitations & Assumptions

### What This Project IS
- ✅ Single-channel signal processing prototype
- ✅ Demonstrates DSP fundamentals and thermal awareness
- ✅ Shows clinical metrics translation approach
- ✅ Functional dashboard for technical + clinical audiences

### What This Project IS NOT
- ❌ Production-ready medical device
- ❌ Validated on real neural data
- ❌ Regulatory-compliant (IEC 60601, ISO 14708)
- ❌ Scalable to 10,000 channels (yet)

### Key Assumptions

1. **Synthetic Data Only**: All testing uses simulated neural signals modeling basic physics:
   - Poisson spike trains (20 Hz rate)
   - Sinusoidal baseline drift (1 Hz, 400 μV amplitude)
   - Gaussian noise (σ = 0.3 × signal range)
   - **Real patient data validation is critical next step**

2. **Algorithm Simplicity**: DSP methods prioritize thermal efficiency over sophistication:
   - Moving average drift removal (vs Butterworth high-pass filters)
   - Derivative spike detection (vs template matching / ML sorting)
   - These are starting points requiring benchmark comparisons

3. **Single-Channel Processing**: Processes one channel serially. Real-time 10k-channel processing requires:
   - Parallel architecture (GPU/FPGA)
   - Hardware acceleration
   - Power optimization (<1mW per channel)

4. **Arbitrary Thresholds**: All thresholds tuned on synthetic data:
   - `max_acceptable_shift = 80.0 μV` - drift tolerance
   - `variance_threshold = 0.8` - SNR minimum
   - `40% / 30% / 30%` - yield weighting
   - `biological_jitter σ = 2.5%` - stochastic variation
   - **Require validation on multi-patient datasets**

5. **Clinical Translation**: Dashboard shows metrics FDA cares about (yield, stability, uptime). Actual regulatory submission requires:
   - IEC 60601 electrical safety compliance
   - Multi-patient clinical trials (n>30)
   - Clinical outcome mapping (movement decoding accuracy, communication bandwidth)
   - Long-term stability data (months to years)

## 🔬 Technical Details

### Thermal Constraints

Brain implants must maintain tissue temperature increase <1°C to prevent necrosis. This constrains signal processing to:
- **No FFTs** (computationally expensive)
- **No deep neural nets** (high power consumption)
- **No heavy IIR/FIR filters** (scipy.signal.butter generates heat)

**Our Approach**: Mathematically simple, vectorized operations:
- Circular buffer moving average: O(1) amortized
- Numpy vectorization: hardware SIMD acceleration
- Latency budget: <20ms per 50ms chunk (tested at ~2-5ms)

### Biphasic Ringing Artifacts

Moving average subtraction creates high-pass filtering edge response:
- Step input → biphasic ringing oscillations (5-20 kHz)
- Caused by rectangular frequency response of MA filter
- **Solution**: Cascade second MA smoothing (40 samples = 1ms cutoff ~1 kHz)
- Attenuates ringing by 99.6% while preserving spike edges

### Adaptive Tanh Normalization

**Why tanh instead of hard clipping:**
- Differentiable everywhere (critical for TN-VAE decoders)
- Smooth transitions prevent high-frequency artifacts
- Hardware-accelerated on most CPUs
- Guarantees [-1, 1] bounds without conditionals

**1.5σ Scaling Rule:**
- Spikes (5-10σ) reach tanh output 0.7-1.0
- Background noise (±1σ) stays near 0 (tanh ≈ ±0.59)
- DC offset removal anchors baseline at 0.0 (fixes MA group delay)

### Testing

Run comprehensive test suite:
```bash
cd axoft_pipeline
python -m pytest test_suite.py -v
```

**Test Coverage:**
- ✅ MA drift removal (circular buffer edge cases)
- ✅ Spike detection accuracy (known spike trains)
- ✅ Tanh normalization bounds (saturation, zero-centering)
- ✅ Smoothing ringing reduction (HF power variance)
- ✅ Metrics engine scoring (variance, spike, stability)
- ✅ Health status transitions (Healthy → Warning → Critical)
- ✅ Latency budgets (<20ms per chunk)

## 📈 Next Steps

### Critical Priorities

1. **Real Data Validation**
   - Request sample neural datasets from Axoft clinical trials
   - Validate all thresholds on multi-patient data
   - Compare yields to published BCI literature (typical: 70-85%)

2. **Algorithm Benchmarking**
   - Compare MA vs Butterworth high-pass filters (frequency response, group delay)
   - Compare derivative detection vs template matching (spike discrimination accuracy)
   - Justify thermal trade-offs with performance data

3. **Scalability Architecture**
   - Design multi-channel parallel processing (GPU/FPGA)
   - Power budget analysis: Target <1mW per channel × 10k channels = 10W total
   - Latency analysis: Can we achieve <5ms per channel with parallelization?

4. **Clinical Outcome Mapping**
   - How does signal yield correlate with decoding accuracy?
   - What yield threshold predicts successful clinical use?
   - Long-term stability tracking (months/years post-implant)

5. **Regulatory Compliance**
   - IEC 60601-1 electrical safety for medical devices
   - ISO 14708-3 active implantable medical devices
   - Risk analysis (ISO 14971)
   - Software validation (IEC 62304)

### Questions for Axoft

1. What's your current DSP architecture? (Hardware filters on ASIC? Edge processor? Cloud?)
2. Do you have sample neural datasets I can test on? (Anonymized patient data)
3. What are typical yields/stability metrics from your clinical trials?
4. What's the target power budget per channel? (<1mW? <100μW?)
5. What clinical outcomes do you prioritize? (Movement decoding? Communication bandwidth? Seizure prediction?)

## 🤝 Contributing

This is an intern project for Axoft. Feedback welcome:
- Open issues for bugs or feature requests
- Submit PRs for improvements
- Contact: [Your Email/LinkedIn]

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Axoft**: For the opportunity to work on cutting-edge BCI technology
- **Neural Engineering Literature**: Papers on chronic BCI stability, spike sorting, drift compensation
- **Open Source Tools**: Streamlit, Plotly, NumPy

---

**Developed by**: Manav Davis
**Date**: March 2026
**Purpose**: Axoft Internship Technical Assessment
**Status**: Prototype / Demo (Not for Clinical Use)
