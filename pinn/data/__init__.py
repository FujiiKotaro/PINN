"""Data loading utilities for FDTD .npz files.

This module contains FDTD data loaders, tensor converters, and data validation
utilities for Phase 2 integration.
"""

from pinn.data.fdtd_loader import FDTDData, FDTDDataLoaderService
from pinn.data.tensor_converter import TensorConverterService

__all__ = [
    "FDTDData",
    "FDTDDataLoaderService",
    "TensorConverterService",
]
