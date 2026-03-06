"""
BCI Signal Yield & Clinical Translation Gateway
==================================================

A production-grade BCI signal processing pipeline designed for flexible
polymer electrodes. Addresses micromotion-induced baseline drift through
thermally-constrained O(1) DSP operations.

Modules:
--------
- dsp_pipeline: Core signal processing with O(1) complexity operations
- data_simulator: Synthetic hardware data generation for testing
- metrics_engine: FDA/clinical translation and business logic
- storage_manager: Backend abstraction layer (in-memory / Redis)
- app: Streamlit dashboard with dual persona views (R&D / Clinical)

Constraints:
------------
- Thermal Budget: <1°C temperature increase (tissue necrosis prevention)
- Latency Budget: <20ms per chunk processing
- Computational Complexity: O(1) or highly efficient linear time only

Author: Senior AI Software Engineer & BCI Data Architect
Date: 2026-03-02
"""

__version__ = "0.1.0"
__author__ = "Manav Davis"

# Package metadata
__all__ = [
    "dsp_pipeline",
    "data_simulator",
    "metrics_engine",
    "storage_manager",
]
