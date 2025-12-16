"""Unit tests for FDTD data loader.

Tests cover .npz file loading, metadata extraction, and data validation.
"""

from pathlib import Path

import numpy as np
import pytest

from pinn.data.fdtd_loader import FDTDData, FDTDDataLoaderService


class TestFDTDDataLoader:
    """Test suite for FDTDDataLoaderService."""

    @pytest.fixture
    def mock_npz_file(self, tmp_path):
        """Create a mock .npz file with FDTD data structure."""
        # Simulate FDTD data structure
        nx_sample, ny_sample, nt_sample = 10, 10, 5
        total_size = nx_sample * ny_sample * nt_sample

        # Create spatiotemporal coordinates
        x = np.linspace(0, 1, total_size)
        y = np.linspace(0, 1, total_size)
        t = np.linspace(0, 1, total_size)

        # Create wave field data
        T1 = np.sin(x) * np.cos(t)
        T3 = np.cos(x) * np.sin(t)
        Ux = np.sin(x + y)
        Uy = np.cos(x + y)

        # Metadata
        pitch = 1.5e-3  # 1.5 mm in meters
        depth = 0.2e-3  # 0.2 mm in meters
        width = 0.1e-3  # 0.1 mm in meters
        seed = 42

        # Save to .npz file
        npz_path = tmp_path / "p1500_d200.npz"
        np.savez(
            npz_path,
            x=x,
            y=y,
            t=t,
            T1=T1,
            T3=T3,
            Ux=Ux,
            Uy=Uy,
            p=pitch,
            d=depth,
            w=width,
            seed=seed,
            nx_sample=nx_sample,
            ny_sample=ny_sample,
            nt_sample=nt_sample,
        )

        return npz_path

    def test_load_file_success(self, mock_npz_file):
        """Test successful loading of .npz file with all fields."""
        loader = FDTDDataLoaderService()
        data = loader.load_file(mock_npz_file)

        # Check that FDTDData object is returned
        assert isinstance(data, FDTDData)

        # Check spatiotemporal coordinates
        assert data.x is not None
        assert data.y is not None
        assert data.t is not None
        assert data.x.shape == (500,)  # 10 * 10 * 5

        # Check wave field data
        assert data.T1 is not None
        assert data.T3 is not None
        assert data.Ux is not None
        assert data.Uy is not None

        # Check metadata
        assert data.pitch == pytest.approx(1.5e-3)
        assert data.depth == pytest.approx(0.2e-3)
        assert data.width == pytest.approx(0.1e-3)
        assert data.seed == 42

        # Check sampling info
        assert data.nx_sample == 10
        assert data.ny_sample == 10
        assert data.nt_sample == 5

    def test_load_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        loader = FDTDDataLoaderService()
        non_existent_path = Path("/tmp/nonexistent_file.npz")

        with pytest.raises(FileNotFoundError):
            loader.load_file(non_existent_path)

    def test_load_file_missing_keys(self, tmp_path):
        """Test that KeyError is raised for malformed .npz files."""
        # Create .npz file with missing required keys
        incomplete_path = tmp_path / "incomplete.npz"
        np.savez(incomplete_path, x=np.array([1, 2, 3]))

        loader = FDTDDataLoaderService()

        with pytest.raises(KeyError):
            loader.load_file(incomplete_path)

    def test_validate_data_success(self, mock_npz_file):
        """Test successful validation of properly formatted data."""
        loader = FDTDDataLoaderService()
        data = loader.load_file(mock_npz_file)

        # Should not raise any exception
        loader.validate_data(data)

    def test_validate_data_shape_mismatch(self, mock_npz_file):
        """Test that ValueError is raised for shape mismatches."""
        loader = FDTDDataLoaderService()
        data = loader.load_file(mock_npz_file)

        # Corrupt the data shape
        data.x = np.array([1, 2, 3])  # Wrong shape

        with pytest.raises(ValueError, match="Invalid x shape"):
            loader.validate_data(data)

    def test_metadata_extraction(self, mock_npz_file):
        """Test that metadata is correctly extracted from filename and content."""
        loader = FDTDDataLoaderService()
        data = loader.load_file(mock_npz_file)

        # Verify metadata matches expected values
        assert data.pitch > 0
        assert data.depth > 0
        assert data.width > 0
        assert isinstance(data.seed, (int, np.integer))

    def test_load_multiple_no_filter(self, tmp_path):
        """Test loading multiple .npz files without filtering."""
        # Create multiple mock files
        self._create_mock_files(tmp_path)

        loader = FDTDDataLoaderService(data_dir=tmp_path)
        data_list = loader.load_multiple()

        # Should load all 3 files
        assert len(data_list) == 3

        # Check that data is sorted by filename
        pitches = [d.pitch for d in data_list]
        assert pitches == sorted(pitches)

    def test_load_multiple_pitch_filter(self, tmp_path):
        """Test loading multiple files with pitch range filter."""
        self._create_mock_files(tmp_path)

        loader = FDTDDataLoaderService(data_dir=tmp_path)
        # Filter for pitch between 1.4mm and 1.6mm (only p1500 should match)
        data_list = loader.load_multiple(pitch_range=(1.4e-3, 1.6e-3))

        assert len(data_list) == 1
        assert data_list[0].pitch == pytest.approx(1.5e-3)

    def test_load_multiple_depth_filter(self, tmp_path):
        """Test loading multiple files with depth range filter."""
        self._create_mock_files(tmp_path)

        loader = FDTDDataLoaderService(data_dir=tmp_path)
        # Filter for depth between 0.15mm and 0.25mm (p1500 and p2000 have 0.2mm depth)
        data_list = loader.load_multiple(depth_range=(0.15e-3, 0.25e-3))

        assert len(data_list) == 2
        depths = [d.depth for d in data_list]
        assert all(pytest.approx(d) == 0.2e-3 for d in depths)

    def test_load_multiple_combined_filter(self, tmp_path):
        """Test loading with both pitch and depth filters."""
        self._create_mock_files(tmp_path)

        loader = FDTDDataLoaderService(data_dir=tmp_path)
        # Filter for pitch 1.4-1.6mm AND depth 0.15-0.25mm (only p1500_d200)
        data_list = loader.load_multiple(
            pitch_range=(1.4e-3, 1.6e-3),
            depth_range=(0.15e-3, 0.25e-3)
        )

        assert len(data_list) == 1
        assert data_list[0].pitch == pytest.approx(1.5e-3)
        assert data_list[0].depth == pytest.approx(0.2e-3)

    def test_load_multiple_no_files_found(self, tmp_path):
        """Test error when no .npz files found in directory."""
        loader = FDTDDataLoaderService(data_dir=tmp_path)

        with pytest.raises(ValueError, match="No .npz files found"):
            loader.load_multiple()

    def test_load_multiple_invalid_directory(self):
        """Test error when data directory doesn't exist."""
        loader = FDTDDataLoaderService(data_dir=Path("/nonexistent/path"))

        with pytest.raises(ValueError, match="Data directory not found"):
            loader.load_multiple()

    def _create_mock_files(self, tmp_path):
        """Helper to create multiple mock .npz files with different parameters."""
        configs = [
            ("p1250_d100.npz", 1.25e-3, 0.1e-3),
            ("p1500_d200.npz", 1.5e-3, 0.2e-3),
            ("p2000_d200.npz", 2.0e-3, 0.2e-3),
        ]

        for filename, pitch, depth in configs:
            nx_sample, ny_sample, nt_sample = 10, 10, 5
            total_size = nx_sample * ny_sample * nt_sample

            np.savez(
                tmp_path / filename,
                x=np.linspace(0, 1, total_size),
                y=np.linspace(0, 1, total_size),
                t=np.linspace(0, 1, total_size),
                T1=np.sin(np.linspace(0, 1, total_size)),
                T3=np.cos(np.linspace(0, 1, total_size)),
                Ux=np.sin(np.linspace(0, 1, total_size)),
                Uy=np.cos(np.linspace(0, 1, total_size)),
                p=pitch,
                d=depth,
                w=0.1e-3,
                seed=42,
                nx_sample=nx_sample,
                ny_sample=ny_sample,
                nt_sample=nt_sample,
            )


class TestFDTDData:
    """Test suite for FDTDData dataclass."""

    def test_fdtd_data_creation(self):
        """Test FDTDData dataclass instantiation."""
        data = FDTDData(
            x=np.array([1, 2, 3]),
            y=np.array([1, 2, 3]),
            t=np.array([0, 0.1, 0.2]),
            T1=np.array([0.1, 0.2, 0.3]),
            T3=np.array([0.1, 0.2, 0.3]),
            Ux=np.array([0.01, 0.02, 0.03]),
            Uy=np.array([0.01, 0.02, 0.03]),
            pitch=1.5e-3,
            depth=0.2e-3,
            width=0.1e-3,
            seed=42,
            nx_sample=10,
            ny_sample=10,
            nt_sample=5,
        )

        assert data.x.shape == (3,)
        assert data.pitch == 1.5e-3
        assert data.seed == 42
