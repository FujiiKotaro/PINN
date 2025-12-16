"""Unit tests for tensor conversion utilities.

Tests cover NumPy to PyTorch tensor conversion with different dtypes and devices.
"""


import numpy as np
import pytest
import torch

from pinn.data.fdtd_loader import FDTDDataLoaderService
from pinn.data.tensor_converter import TensorConverterService


class TestTensorConverter:
    """Test suite for TensorConverterService."""

    def test_to_tensor_float32_cpu(self):
        """Test conversion to float32 tensor on CPU."""
        array = np.array([1.0, 2.0, 3.0])
        converter = TensorConverterService()

        tensor = converter.to_tensor(array, dtype="float32", device=torch.device("cpu"))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=torch.device("cpu"))
        assert torch.allclose(tensor, expected)

    def test_to_tensor_float64_cpu(self):
        """Test conversion to float64 tensor on CPU."""
        array = np.array([1.0, 2.0, 3.0])
        converter = TensorConverterService()

        tensor = converter.to_tensor(array, dtype="float64", device=torch.device("cpu"))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float64
        assert tensor.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_tensor_cuda(self):
        """Test conversion to tensor on CUDA device."""
        array = np.array([1.0, 2.0, 3.0])
        converter = TensorConverterService()

        tensor = converter.to_tensor(array, dtype="float32", device=torch.device("cuda"))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cuda"

    def test_to_tensor_default_parameters(self):
        """Test conversion with default parameters (float32, CPU)."""
        array = np.array([1.0, 2.0, 3.0])
        converter = TensorConverterService()

        tensor = converter.to_tensor(array)

        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"

    def test_to_tensor_2d_array(self):
        """Test conversion of 2D array."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        converter = TensorConverterService()

        tensor = converter.to_tensor(array)

        assert tensor.shape == (2, 2)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=torch.device("cpu"))
        assert torch.allclose(tensor, expected)

    def test_batch_convert_fdtd_data(self, tmp_path):
        """Test batch conversion of FDTDData to tensors."""
        # Create mock FDTD data
        fdtd_data = self._create_mock_fdtd_data(tmp_path)

        converter = TensorConverterService()
        tensor_dict = converter.batch_convert(fdtd_data, dtype="float32", device=torch.device("cpu"))

        # Check all fields are converted
        expected_keys = ['x', 'y', 't', 'T1', 'T3', 'Ux', 'Uy']
        assert set(tensor_dict.keys()) == set(expected_keys)

        # Check data types
        for key in expected_keys:
            assert isinstance(tensor_dict[key], torch.Tensor)
            assert tensor_dict[key].dtype == torch.float32
            assert tensor_dict[key].device.type == "cpu"

        # Check shapes match original
        assert tensor_dict['x'].shape == fdtd_data.x.shape
        assert tensor_dict['T1'].shape == fdtd_data.T1.shape

    def test_batch_convert_float64(self, tmp_path):
        """Test batch conversion with float64 precision."""
        fdtd_data = self._create_mock_fdtd_data(tmp_path)

        converter = TensorConverterService()
        tensor_dict = converter.batch_convert(fdtd_data, dtype="float64")

        # Check precision
        for key in ['x', 'y', 't', 'T1', 'T3', 'Ux', 'Uy']:
            assert tensor_dict[key].dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_convert_cuda(self, tmp_path):
        """Test batch conversion to CUDA device."""
        fdtd_data = self._create_mock_fdtd_data(tmp_path)

        converter = TensorConverterService()
        tensor_dict = converter.batch_convert(
            fdtd_data,
            dtype="float32",
            device=torch.device("cuda")
        )

        # Check device placement
        for key in ['x', 'y', 't', 'T1', 'T3', 'Ux', 'Uy']:
            assert tensor_dict[key].device.type == "cuda"

    def test_tensor_values_preserved(self):
        """Test that tensor conversion preserves numerical values."""
        array = np.array([1.23456789, 2.34567890, 3.45678901])
        converter = TensorConverterService()

        tensor = converter.to_tensor(array, dtype="float64")

        # Convert back to numpy for comparison
        numpy_from_tensor = tensor.cpu().numpy()
        np.testing.assert_array_almost_equal(array, numpy_from_tensor, decimal=7)

    def _create_mock_fdtd_data(self, tmp_path):
        """Helper to create mock FDTDData object."""
        nx_sample, ny_sample, nt_sample = 10, 10, 5
        total_size = nx_sample * ny_sample * nt_sample

        npz_path = tmp_path / "test_data.npz"
        np.savez(
            npz_path,
            x=np.linspace(0, 1, total_size),
            y=np.linspace(0, 1, total_size),
            t=np.linspace(0, 1, total_size),
            T1=np.sin(np.linspace(0, 1, total_size)),
            T3=np.cos(np.linspace(0, 1, total_size)),
            Ux=np.sin(np.linspace(0, 1, total_size)),
            Uy=np.cos(np.linspace(0, 1, total_size)),
            p=1.5e-3,
            d=0.2e-3,
            w=0.1e-3,
            seed=42,
            nx_sample=nx_sample,
            ny_sample=ny_sample,
            nt_sample=nt_sample,
        )

        loader = FDTDDataLoaderService()
        return loader.load_file(npz_path)
