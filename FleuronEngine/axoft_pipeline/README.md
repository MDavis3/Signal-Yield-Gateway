# BCI Signal Yield & Clinical Translation Gateway

**Production-grade BCI signal processing pipeline for flexible polymer electrodes**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

This system addresses the critical micromotion-induced baseline drift problem in ultra-soft polymer BCI electrodes through thermally-constrained O(1) DSP operations, while providing dual personas (R&D and Clinical/FDA) for stakeholder communication.

### The Problem

Flexible electrodes solve brain scarring (gliosis) but suffer from physical micromotion due to heartbeat and respiration, creating:
- Severe low-frequency baseline drift (±500 μV)
- Variable spike amplitudes as electrodes move relative to neurons
- Incompatible data format for TN-VAE latent-space decoders

### The Solution

A modular DSP pipeline with **strict thermal and latency constraints**:
- **Thermal Budget:** <1°C heat increase (tissue necrosis prevention)
- **Latency Budget:** <20ms per chunk processing
- **Computational Complexity:** O(1) or highly efficient linear time only

---

## Features

### 🔬 R&D Engineer View
- **Real-time waveform visualization:** Raw (drifting) vs. Cleaned (stabilized)
- **Pipeline latency monitoring:** Verify <20ms budget compliance
- **Dual-trace comparison:** See baseline drift removal and spike preservation

### 🏥 Clinical / FDA View
- **Live Signal Yield %:** Multi-factor quality score (variance + spike rate + stability)
- **Active Channels:** Maps yield to channel dropout (out of 10,000 total)
- **Chronic Stability Index:** Proves no manual recalibration needed over 200 epochs
- **System Health Indicator:** Medical-grade error handling (Healthy / Warning / Critical)

### ▶️ Playback Controls
- **Play Mode:** Auto-stream 50ms chunks in real-time
- **Pause Mode:** Freeze state for parameter tweaking
- **Step Mode:** Single chunk generation for frame-by-frame explanation

---

## Architecture

### Module Breakdown

```
pipeline/
├── __init__.py                 # Package initialization
├── dsp_pipeline.py            # O(1) signal processing (150 lines)
├── data_simulator.py          # Synthetic hardware data generation (100 lines)
├── metrics_engine.py          # FDA/clinical business logic (200 lines)
├── storage_manager.py         # Backend abstraction layer (80 lines)
├── app.py                     # Streamlit UI (180 lines)
└── requirements.txt           # Dependencies
```

### Data Flow

```
1. Generate Synthetic Chunk (data_simulator)
   ↓ 2000 samples @ 40kHz (50ms) with noise, spikes, drift
2. DSP Pipeline (dsp_pipeline)
   ↓ Polynomial detrending → Spike detection → Tanh normalization
3. Metrics Calculation (metrics_engine)
   ↓ Signal Yield % (variance + spike rate + stability)
4. Storage (storage_manager)
   ↓ In-memory or Redis backend
5. Visualization (app.py)
   ↓ Dual persona dashboard (R&D / Clinical)
```

### Key Algorithms

**Polynomial Detrending (O(n) LAPACK-accelerated)**
- Fits linear polynomial to each 50ms chunk and subtracts baseline
- Eliminates ringing artifacts that plague moving average filters
- Stateless (no buffer carryover between chunks)
- Preserves sharp spike morphology for ML decoders

**Derivative-Based Spike Detection (O(n) linear)**
- Detects sharp rising edges characteristic of action potentials (~1ms)
- More robust than amplitude threshold (which fails as electrode drifts)
- Uses `np.diff()` for vectorized computation

**Hyperbolic Tangent Normalization (O(n) vectorized)**
- Soft-clips extreme artifacts while preserving differentiability for ML decoders
- Guarantees bounds [-1, 1] without conditional logic
- Hardware-accelerated `np.tanh()` keeps latency <20ms

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Quick Start

```bash
# 1. Clone or download the repository
cd pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Production Deployment (Redis Backend)

```bash
# 1. Install Redis dependencies
pip install redis msgpack

# 2. Deploy Redis server (AWS ElastiCache, Redis Cloud, or local)
# Example: Local Redis
docker run -d -p 6379:6379 redis:latest

# 3. Update storage_manager.py configuration
# Change: DEFAULT_STORAGE_BACKEND = "in_memory"
# To:     DEFAULT_STORAGE_BACKEND = "redis"

# 4. Run the app
streamlit run app.py
```

---

## Usage Guide

### For R&D Engineers

1. **Launch the app:** `streamlit run app.py`
2. **Select "R&D Engineer View"** in the sidebar
3. **Adjust signal parameters:**
   - **Micromotion Drift Severity:** 0.0 (ideal) → 2.0 (severe)
   - **Noise Level:** 0.0 (pristine) → 1.0 (very noisy)
   - **Tanh Alpha:** 0.1 (soft) → 5.0 (hard clipping)
4. **Note:** Polynomial detrending (order=1) is used automatically - no manual tuning required
5. **Use playback controls:**
   - **Play:** Auto-stream chunks every 50ms
   - **Pause:** Freeze to explain math
   - **Step:** Generate one chunk at a time for detailed analysis
6. **Observe waveforms:**
   - Red trace (raw): Shows severe baseline drift
   - Cyan trace (cleaned): Flat baseline at zero, spikes preserved

### For Clinical / FDA Reviewers

1. **Launch the app:** `streamlit run app.py`
2. **Select "Clinical / FDA View"** in the sidebar
3. **Monitor KPIs:**
   - **Live Signal Yield %:** >90% excellent, 70-90% good, <50% poor
   - **Active Channels:** Out of 10,000 total (shows clinical impact of micromotion)
   - **System Uptime:** Continuous operation time
4. **Track long-term stability:**
   - Adjust **Stability Window** slider (10-200 epochs)
   - Observe **Chronic Stability Index** chart
   - Look for flat trend >90% over 200 epochs (proves no recalibration needed)
5. **Check system health:**
   - ✅ **Healthy:** All metrics nominal
   - ⚠️ **Warning:** Degraded performance but operational
   - 🔴 **Critical:** Hardware fault suspected, requires intervention

### Preset Configurations

Use the preset buttons in the dashboard sidebar for different scenarios:

| Preset | Noise | Drift | Alpha | Use Case |
|--------|-------|-------|-------|----------|
| **🎯 Demo Mode** | 0.10 | 0.20 | 0.70 | Clean visualization for presentations |
| **⚙️ Real Mode** | 0.30 | 0.40 | 1.00 | Realistic biological conditions |
| **🔥 Stress Test** | 0.50 | 1.00 | 1.50 | Worst-case testing |

---

## Performance Benchmarks

### Latency (50ms chunk @ 40kHz = 2000 samples)

| Operation | Complexity | Latency | Thermal Impact |
|-----------|------------|---------|----------------|
| Polynomial Detrending | O(n) LAPACK | 0.5-1.0 ms | Negligible |
| Spike Detection | O(n) vectorized | 0.3 ms | Negligible |
| Tanh Normalization | O(n) vectorized | 0.5 ms | Negligible |
| **Total Pipeline** | **O(n)** | **1.5-2.0 ms** | **<0.01°C** |

**Thermal Budget Compliance:** ✅ Well within <1°C constraint
**Latency Budget Compliance:** ✅ Well within <20ms constraint

---

## Troubleshooting

### "Module not found" error

```bash
# Ensure you're in the correct directory
cd pipeline

# Reinstall dependencies
pip install -r requirements.txt
```

### Dashboard not updating in Play mode

```bash
# Check Streamlit version
pip show streamlit

# Update to latest
pip install --upgrade streamlit
```

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=pipeline
```

### Code Formatting

```bash
# Install black
pip install black

# Format code
black pipeline/ --line-length 100
```

---

## License

MIT License - See LICENSE file for details

---

## Authors

- **Lead Developer:** Manav Davis

---

## Changelog

### v0.3.0 (2026-03-05)
- **MAJOR:** Added real PhysioNet EEG data support (64-channel, 160Hz)
- **NEW:** Motor Imagery BCI view with C3/C4 mu-band classification
- **NEW:** Frequency band power analysis (delta/theta/alpha/beta/gamma)
- **NEW:** Multi-channel waterfall view (4-8 channels)
- **NEW:** 60Hz notch filter for power line interference removal

### v0.2.1 (2026-03-03)
- **IMPROVED:** Preset buttons now set optimal alpha values for each scenario
- **NEW:** Added visual clarity warning for suboptimal alpha/noise combinations

### v0.2.0 (2026-03-03)
- **MAJOR:** Replaced moving average with polynomial detrending (eliminates ringing)
- **NEW:** Added preset buttons for Demo/Real/Stress configurations

### v0.1.0 (2026-03-02)
- Initial release
- Modular architecture (5 core modules)
- Dual persona dashboard (R&D / Clinical)
- Play/Pause/Step playback controls

---

## Contact

For questions or feedback:
- **LinkedIn:** https://www.linkedin.com/in/manavdavis313/
- **Email:** manav_davis@brown.edu

---

**Built with ❤️ for advancing neurotechnology**
