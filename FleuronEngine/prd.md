Product Requirements Document (PRD)
Project: BCI Signal Ingestion Gateway & Clinical Translation Engine
Document Owner: Manav Davis
Status: Prototype / V1
1. Executive Summary
Flexible polymer electrodes provide a massive biocompatibility advantage by eliminating gliosis (scarring). However, this soft hardware introduces a unique data challenge: physical micromotion caused by cerebrospinal fluid pulsations and heartbeats. This PRD outlines a streaming data pipeline designed to mathematically stabilize this micromotion artifact in real-time, format the data for TN-VAE decoding models, and translate the pipeline's efficacy into an FDA-facing "Chronic Stability" dashboard.
2. Problem Statement & Biological Constraints
Traditional rigid arrays (like Utah Arrays) are physically static, ensuring a consistent distance to the neuron. Flexible arrays "float." As the brain pulsates, the distance between the electrode and the neuron shifts, causing two critical signal degradations:
Baseline Drift: A low-frequency rolling artifact (the "ocean wave").
Amplitude Variance: The neural spikes change in voltage as the physical distance fluctuates.
If this raw, drifting data is fed directly into a latent-space AI decoder, the model will interpret the physical hardware movement as a change in the user's neural intent, causing the BCI to fail.
3. Engineering & Thermal Non-Negotiables
Because this processing must eventually happen at the edge (on or near the implant), we are bound by strict thermal limits. Brain tissue cannot be heated by more than 1 Celsius without causing necrosis.
Constraint 1: No Heavy Compute. We cannot use computationally expensive recursive filters (like Fast Fourier Transforms or heavy CNNs) to clean the signal. Heavy compute generates heat and drains battery.
Constraint 2: O(1) Time Complexity Per Sample. The processing layer must use mathematically "cheap" operations to ensure sub-20ms system latency.
4. Signal Processing Architecture
To satisfy the thermal and latency constraints, the pipeline utilizes two ultra-lightweight mathematical operations applied to 50ms data chunks:
Operation 1 (Baseline Correction): A dynamic moving-average subtraction.
Logic: x{clean} = x{t} - μ{window}
Purpose: Instantly flattens the low-frequency heartbeat drift, snapping the signal baseline back to absolute zero.
Operation 2 (Amplitude Normalization): Hyperbolic Tangent (tanh) Soft-Clipping.
Logic: y{t} = tanh(α * x{clean})
Purpose: tanh is an elegant, non-linear function that squashes extreme artifact noise into a strict [-1, 1] bound while preserving the relative morphology of the neural spikes. It normalizes the data for TN-VAE models using a single, low-cost operation.
5. Functional Product Requirements
The system must bridge the gap between R&D data processing and Go-To-Market clinical utility.
Requirement 1: AI-Ready Hand-off
The pipeline must automatically chunk the stabilized data and output a structured, float32 PyTorch tensor. This ensures that the engineering team can ingest the data instantly without writing secondary transformation scripts.
Requirement 2: Regulatory Translation (The "Signal Yield" Metric)
The FDA requires proof that the BCI is chronically stable without weekly clinical recalibration. The pipeline must calculate a live Signal Yield Percentage (e.g., the percentage of the 10,000 channels maintaining a signal-to-noise ratio above a viable threshold).
Requirement 3: Multi-View Dashboard
The UI (Streamlit) must serve two distinct user personas:
Engineer View: Displays real-time Plotly waveforms comparing the raw micromotion drift against the stabilized tanh output, tracking sub-millisecond pipeline latency.
Clinician/FDA View: Abstracts the waveforms entirely. Displays macro-level KPI cards: Live Signal Yield %, System Uptime, and a Chronic Stability Index line chart to prove long-term viability.

