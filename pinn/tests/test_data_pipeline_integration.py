"""Tests for data loading pipeline integration with dimensionless scaling.

Test-Driven Development: Tests written before implementation.
Tests cover integration of DimensionlessScalerService with FDTDDataLoaderService.
"""

from pathlib import Path

import numpy as np
import pytest

from pinn.data.dimensionless_scaler import CharacteristicScales, DimensionlessScalerService
from pinn.data.fdtd_loader import FDTDDataLoaderService


class TestDataPipelineIntegration:
    """Test integration of data loading with dimensionless scaling."""

    @pytest.fixture
    def loader(self):
        """Create FDTD loader with data directory."""
        data_dir = Path("/home/manat/project2/PINN_data")
        return FDTDDataLoaderService(data_dir=data_dir)

    @pytest.fixture
    def scaler(self):
        """Create dimensionless scaler with physical parameters."""
        # Aluminum 6061 parameters
        elastic_lambda = 51.2e9  # Pa
        elastic_mu = 26.1e9  # Pa
        density = 2700.0  # kg/m^3
        domain_length = 0.04  # 40mm

        # Estimate U_ref from data
        file_path = Path("/home/manat/project2/PINN_data/p1250_d100.npz")
        loader = FDTDDataLoaderService()
        data = loader.load_file(file_path)
        U_ref = np.std(np.concatenate([data.Ux, data.Uy]))

        # Create characteristic scales
        scales = CharacteristicScales.from_physics(
            domain_length=domain_length,
            elastic_lambda=elastic_lambda,
            elastic_mu=elastic_mu,
            density=density,
            displacement_amplitude=U_ref
        )

        return DimensionlessScalerService(scales)

    def test_load_multiple_files_without_dimensionless_unchanged(self, loader):
        """Test backward compatibility: apply_dimensionless=False works as before."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        # Load without dimensionless scaling (default)
        dataset = loader.load_multiple_files(file_paths, apply_dimensionless=False)

        # Coordinates should be in original units (meters, seconds)
        assert np.max(np.abs(dataset.x)) > 1e-4  # Meters (e.g., 0.04m = 40mm)
        assert np.max(np.abs(dataset.t)) > 1e-7  # Seconds (e.g., 3.5e-6s)

        # Data should be in original scale (FDTD data is already normalized in .npz)
        # T1 in raw data is around O(1), not raw Pa
        assert np.max(np.abs(dataset.T1)) > 0.1  # Original scale from .npz
        assert np.max(np.abs(dataset.T1)) < 10   # Should not be large values

    def test_load_multiple_files_with_dimensionless_applies_scaling(self, loader, scaler):
        """Test that apply_dimensionless=True applies dimensionless scaling."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        # Load with dimensionless scaling
        dataset = loader.load_multiple_files(
            file_paths,
            apply_dimensionless=True,
            scaler=scaler
        )

        # Coordinates should be dimensionless (O(1) scale)
        assert np.max(np.abs(dataset.x)) < 10  # Dimensionless
        assert np.max(np.abs(dataset.t)) < 10  # Dimensionless

        # Stress should be O(1) scale
        assert np.max(np.abs(dataset.T1)) < 100  # Dimensionless O(1)

    def test_load_multiple_files_dimensionless_produces_o1_scale(self, loader, scaler):
        """Test that dimensionless scaling produces O(1) scale variables."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
            Path("/home/manat/project2/PINN_data/p1500_d200.npz"),
        ]

        dataset = loader.load_multiple_files(
            file_paths,
            apply_dimensionless=True,
            scaler=scaler
        )

        # All dimensionless variables should be O(1)
        # x, y normalized to [0, 1] approximately
        assert 0 <= np.min(dataset.x) < 1
        assert 0 < np.max(dataset.x) <= 2

        assert 0 <= np.min(dataset.y) < 1
        assert 0 < np.max(dataset.y) <= 2

        # t normalized to O(1)
        assert np.min(dataset.t) >= 0
        assert np.max(dataset.t) < 10

        # Stress and displacement O(1)
        assert np.max(np.abs(dataset.T1)) < 100
        assert np.max(np.abs(dataset.T3)) < 100
        assert np.max(np.abs(dataset.Ux)) < 100
        assert np.max(np.abs(dataset.Uy)) < 100

    def test_load_multiple_files_dimensionless_preserves_all_fields(self, loader, scaler):
        """Test that all fields are preserved after dimensionless scaling."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        # Load single file to get expected size
        single_data = loader.load_file(file_paths[0])
        expected_size = len(single_data.x)

        # Load with dimensionless
        dataset = loader.load_multiple_files(
            file_paths,
            apply_dimensionless=True,
            scaler=scaler
        )

        # All fields should have same size
        assert len(dataset.x) == expected_size
        assert len(dataset.y) == expected_size
        assert len(dataset.t) == expected_size
        assert len(dataset.pitch_norm) == expected_size
        assert len(dataset.depth_norm) == expected_size
        assert len(dataset.T1) == expected_size
        assert len(dataset.T3) == expected_size
        assert len(dataset.Ux) == expected_size
        assert len(dataset.Uy) == expected_size

    def test_load_multiple_files_dimensionless_with_multiple_files(self, loader, scaler):
        """Test dimensionless scaling with multiple files."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
            Path("/home/manat/project2/PINN_data/p1500_d200.npz"),
        ]

        # Load individual files
        data1 = loader.load_file(file_paths[0])
        data2 = loader.load_file(file_paths[1])
        expected_total = len(data1.x) + len(data2.x)

        # Load with dimensionless
        dataset = loader.load_multiple_files(
            file_paths,
            apply_dimensionless=True,
            scaler=scaler
        )

        # Total size should match
        assert len(dataset.x) == expected_total

        # All should be O(1) scale
        assert np.max(np.abs(dataset.T1)) < 100

    def test_load_multiple_files_raises_error_if_scaler_missing(self, loader):
        """Test that ValueError is raised if apply_dimensionless=True but scaler=None."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        with pytest.raises(ValueError, match="scaler must be provided"):
            loader.load_multiple_files(
                file_paths,
                apply_dimensionless=True,
                scaler=None
            )

    def test_load_multiple_files_parameter_normalization_unaffected(self, loader, scaler):
        """Test that parameter normalization still works with dimensionless scaling."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        dataset = loader.load_multiple_files(
            file_paths,
            apply_dimensionless=True,
            scaler=scaler
        )

        # Parameters should still be normalized to [0, 1]
        assert 0 <= np.min(dataset.pitch_norm) <= 1
        assert 0 <= np.max(dataset.pitch_norm) <= 1
        assert 0 <= np.min(dataset.depth_norm) <= 1
        assert 0 <= np.max(dataset.depth_norm) <= 1

    def test_load_multiple_files_metadata_preserved_with_dimensionless(self, loader, scaler):
        """Test that metadata is preserved with dimensionless scaling."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        dataset = loader.load_multiple_files(
            file_paths,
            apply_dimensionless=True,
            scaler=scaler
        )

        # Metadata should be present
        assert 'files' in dataset.metadata
        assert 'params' in dataset.metadata
        assert len(dataset.metadata['files']) == 1
        assert len(dataset.metadata['params']) == 1
