"""Tests for 2D FDTD data loader with multiple file loading.

Test-Driven Development: Tests written before implementation.
Tests cover multiple .npz file loading, parameter filtering, and data concatenation.
"""

from pathlib import Path

import numpy as np
import pytest

from pinn.data.fdtd_loader import FDTDDataLoaderService, FDTDDataset2D


class TestFDTDDataset2D:
    """Test FDTDDataset2D dataclass structure."""

    def test_fdtd_dataset_2d_structure(self):
        """Test that FDTDDataset2D has correct attributes."""
        # Create dummy dataset
        N = 100
        dataset = FDTDDataset2D(
            x=np.random.rand(N),
            y=np.random.rand(N),
            t=np.random.rand(N),
            pitch_norm=np.random.rand(N),
            depth_norm=np.random.rand(N),
            T1=np.random.rand(N),
            T3=np.random.rand(N),
            Ux=np.random.rand(N),
            Uy=np.random.rand(N),
            metadata={'files': [], 'params': []}
        )

        # Verify all attributes exist
        assert hasattr(dataset, 'x')
        assert hasattr(dataset, 'y')
        assert hasattr(dataset, 't')
        assert hasattr(dataset, 'pitch_norm')
        assert hasattr(dataset, 'depth_norm')
        assert hasattr(dataset, 'T1')
        assert hasattr(dataset, 'T3')
        assert hasattr(dataset, 'Ux')
        assert hasattr(dataset, 'Uy')
        assert hasattr(dataset, 'metadata')

    def test_fdtd_dataset_2d_array_shapes_match(self):
        """Test that all arrays in dataset have same length."""
        N = 50
        dataset = FDTDDataset2D(
            x=np.random.rand(N),
            y=np.random.rand(N),
            t=np.random.rand(N),
            pitch_norm=np.random.rand(N),
            depth_norm=np.random.rand(N),
            T1=np.random.rand(N),
            T3=np.random.rand(N),
            Ux=np.random.rand(N),
            Uy=np.random.rand(N),
            metadata={}
        )

        # All arrays should have same length
        assert len(dataset.x) == N
        assert len(dataset.y) == N
        assert len(dataset.t) == N
        assert len(dataset.pitch_norm) == N
        assert len(dataset.depth_norm) == N
        assert len(dataset.T1) == N
        assert len(dataset.T3) == N
        assert len(dataset.Ux) == N
        assert len(dataset.Uy) == N


class TestLoadMultipleFiles:
    """Test loading multiple .npz files and combining them."""

    @pytest.fixture
    def loader(self):
        """Create FDTD loader with data directory."""
        data_dir = Path("/home/manat/project2/PINN_data")
        return FDTDDataLoaderService(data_dir=data_dir)

    def test_load_multiple_files_returns_dataset(self, loader):
        """Test that load_multiple_files returns FDTDDataset2D."""
        # Load 2 files for testing
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
            Path("/home/manat/project2/PINN_data/p1500_d200.npz")
        ]

        dataset = loader.load_multiple_files(file_paths)

        assert isinstance(dataset, FDTDDataset2D)

    def test_load_multiple_files_concatenates_data(self, loader):
        """Test that data from multiple files is concatenated correctly."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
            Path("/home/manat/project2/PINN_data/p1500_d200.npz")
        ]

        # Load individual files to get expected total size
        data1 = loader.load_file(file_paths[0])
        data2 = loader.load_file(file_paths[1])
        expected_total = len(data1.x) + len(data2.x)

        # Load multiple files
        dataset = loader.load_multiple_files(file_paths)

        # Check total size matches sum of individual files
        assert len(dataset.x) == expected_total
        assert len(dataset.T1) == expected_total

    def test_load_multiple_files_normalizes_parameters(self, loader):
        """Test that pitch and depth are normalized to [0, 1]."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        dataset = loader.load_multiple_files(file_paths)

        # Normalized parameters should be in [0, 1]
        assert np.all(dataset.pitch_norm >= 0.0)
        assert np.all(dataset.pitch_norm <= 1.0)
        assert np.all(dataset.depth_norm >= 0.0)
        assert np.all(dataset.depth_norm <= 1.0)

    def test_load_multiple_files_preserves_metadata(self, loader):
        """Test that metadata contains file and parameter info."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
            Path("/home/manat/project2/PINN_data/p1500_d200.npz")
        ]

        dataset = loader.load_multiple_files(file_paths)

        # Metadata should contain files list
        assert 'files' in dataset.metadata
        assert len(dataset.metadata['files']) == 2

        # Metadata should contain parameters
        assert 'params' in dataset.metadata
        assert len(dataset.metadata['params']) == 2

    def test_load_multiple_files_filters_by_pitch_range(self, loader):
        """Test that files outside pitch range are filtered."""
        # Create list with all files
        all_files = list(Path("/home/manat/project2/PINN_data").glob("p*.npz"))

        # Load with pitch filter [1.25mm, 1.75mm]
        dataset = loader.load_multiple_files(
            all_files,
            pitch_range=(1.25e-3, 1.75e-3)
        )

        # Check metadata for filtered parameters
        for param in dataset.metadata['params']:
            pitch = param['pitch']
            assert 1.25e-3 <= pitch <= 1.75e-3, \
                f"Pitch {pitch} outside range [1.25e-3, 1.75e-3]"

    def test_load_multiple_files_filters_by_depth_range(self, loader):
        """Test that files outside depth range are filtered."""
        all_files = list(Path("/home/manat/project2/PINN_data").glob("p*.npz"))

        # Load with depth filter [0.1mm, 0.2mm]
        dataset = loader.load_multiple_files(
            all_files,
            depth_range=(0.1e-3, 0.2e-3)
        )

        # Check metadata for filtered parameters
        for param in dataset.metadata['params']:
            depth = param['depth']
            assert 0.1e-3 <= depth <= 0.2e-3, \
                f"Depth {depth} outside range [0.1e-3, 0.2e-3]"

    def test_load_multiple_files_replicates_params_per_sample(self, loader):
        """Test that normalized params are replicated for each spatiotemporal point."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/p1250_d100.npz"),
        ]

        # Load single file
        single_data = loader.load_file(file_paths[0])
        n_samples = len(single_data.x)

        # Load via load_multiple_files
        dataset = loader.load_multiple_files(file_paths)

        # pitch_norm and depth_norm should have same value for all samples from same file
        unique_pitch = np.unique(dataset.pitch_norm)
        unique_depth = np.unique(dataset.depth_norm)

        assert len(unique_pitch) == 1, "Single file should have one unique pitch_norm"
        assert len(unique_depth) == 1, "Single file should have one unique depth_norm"
        assert len(dataset.pitch_norm) == n_samples

    def test_load_multiple_files_raises_error_if_file_not_found(self, loader):
        """Test that FileNotFoundError is raised for missing files."""
        file_paths = [
            Path("/home/manat/project2/PINN_data/nonexistent.npz")
        ]

        with pytest.raises(FileNotFoundError):
            loader.load_multiple_files(file_paths)

    def test_load_multiple_files_with_empty_list(self, loader):
        """Test behavior with empty file list."""
        file_paths = []

        with pytest.raises(ValueError, match="No files provided"):
            loader.load_multiple_files(file_paths)
