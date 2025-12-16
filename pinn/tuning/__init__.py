"""Loss weight tuning framework.

This module contains automated grid/random search for optimal physics-informed
loss function weights (w_data, w_pde, w_bc).
"""
from pinn.tuning.weight_tuning import (
    TuningConfig,
    TuningResult,
    WeightTuningFrameworkService,
)

__all__ = [
    "TuningConfig",
    "TuningResult",
    "WeightTuningFrameworkService",
]
