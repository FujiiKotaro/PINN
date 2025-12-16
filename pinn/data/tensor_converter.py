"""Tensor conversion utilities for FDTD data.

This module provides utilities to convert NumPy arrays to PyTorch tensors
with configurable dtype and device placement (CPU/CUDA).
"""

from typing import Literal

import numpy as np
import torch

from pinn.data.fdtd_loader import FDTDData


class TensorConverterService:
    """Service for converting NumPy arrays to PyTorch tensors."""

    @staticmethod
    def to_tensor(
        array: np.ndarray,
        dtype: Literal["float32", "float64"] = "float32",
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor.

        Args:
            array: Input NumPy array
            dtype: Target dtype (float32 for AMP, float64 for precision)
            device: Target device (cpu or cuda)

        Returns:
            PyTorch tensor with specified dtype and device

        Examples:
            >>> array = np.array([1.0, 2.0, 3.0])
            >>> tensor = TensorConverterService.to_tensor(array, dtype="float32")
            >>> tensor.dtype
            torch.float32
        """
        # Map string dtype to torch dtype
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64

        # Convert NumPy array to PyTorch tensor
        tensor = torch.from_numpy(array).to(dtype=torch_dtype, device=device)

        return tensor

    @staticmethod
    def batch_convert(
        data: FDTDData,
        dtype: Literal["float32", "float64"] = "float32",
        device: torch.device = torch.device("cpu")
    ) -> dict[str, torch.Tensor]:
        """Convert all FDTDData arrays to tensors.

        Args:
            data: FDTDData object with NumPy arrays
            dtype: Target dtype for all tensors
            device: Target device for all tensors

        Returns:
            Dictionary mapping field names to PyTorch tensors
            Keys: x, y, t, T1, T3, Ux, Uy

        Examples:
            >>> loader = FDTDDataLoaderService()
            >>> data = loader.load_file(Path("data.npz"))
            >>> tensors = TensorConverterService.batch_convert(data)
            >>> tensors['x'].shape
            torch.Size([80000])
        """
        return {
            "x": TensorConverterService.to_tensor(data.x, dtype, device),
            "y": TensorConverterService.to_tensor(data.y, dtype, device),
            "t": TensorConverterService.to_tensor(data.t, dtype, device),
            "T1": TensorConverterService.to_tensor(data.T1, dtype, device),
            "T3": TensorConverterService.to_tensor(data.T3, dtype, device),
            "Ux": TensorConverterService.to_tensor(data.Ux, dtype, device),
            "Uy": TensorConverterService.to_tensor(data.Uy, dtype, device),
        }
