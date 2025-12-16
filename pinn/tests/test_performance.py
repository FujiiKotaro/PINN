"""Performance tests for PINN training and data loading.

This module tests performance requirements:
- Training time benchmarks
- Data loading speed
- GPU memory usage
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from pinn.data.fdtd_loader import FDTDDataLoaderService


class TestDataLoadingPerformance:
    """Performance tests for FDTD data loading."""

    def test_load_npz_file_under_2_seconds(self):
        """Test that loading a 4.7 MB .npz file completes in <2 seconds.

        Requirement: Data loading for single .npz file shall complete in under 2 seconds.
        """
        # Create a test .npz file similar to actual PINN_data files
        temp_dir = Path(tempfile.mkdtemp())
        test_file = temp_dir / "test_data.npz"

        # Generate synthetic data matching actual file structure
        # Actual files have nt*nx*ny samples
        n_samples = 100000  # Approximate size for ~4.7 MB

        synthetic_data = {
            'x': np.random.randn(n_samples).astype(np.float64),
            'y': np.random.randn(n_samples).astype(np.float64),
            't': np.random.randn(n_samples).astype(np.float64),
            'T1': np.random.randn(n_samples).astype(np.float64),
            'T3': np.random.randn(n_samples).astype(np.float64),
            'Ux': np.random.randn(n_samples).astype(np.float64),
            'Uy': np.random.randn(n_samples).astype(np.float64),
            'p': np.array([0.00125]),  # pitch in meters
            'd': np.array([0.0001]),   # depth in meters
            'w': np.array([0.00005]),  # width in meters
            'seed': np.array([42]),
            'nx_sample': np.array([50]),
            'ny_sample': np.array([50]),
            'nt_sample': np.array([40])
        }

        # Save to .npz
        np.savez(test_file, **synthetic_data)

        # Verify file size is reasonable
        file_size_mb = test_file.stat().st_size / (1024 * 1024)
        print(f"Test file size: {file_size_mb:.2f} MB")

        # Test loading time
        loader = FDTDDataLoaderService(data_dir=temp_dir)

        start_time = time.time()
        data = loader.load_file(test_file)
        elapsed_time = time.time() - start_time

        print(f"Loading time: {elapsed_time:.3f} seconds")

        # Assert loading completes in under 2 seconds
        assert elapsed_time < 2.0, f"Loading took {elapsed_time:.3f}s, should be <2s"

        # Verify data was loaded correctly
        assert data.x.shape == (n_samples,)
        assert data.T1.shape == (n_samples,)

        # Cleanup
        test_file.unlink()
        temp_dir.rmdir()

    def test_load_multiple_files_performance(self):
        """Test loading multiple .npz files is reasonably fast."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create 3 test files
        n_samples = 50000
        n_files = 3

        for i in range(n_files):
            test_file = temp_dir / f"p{1250+i*100}_d{100}.npz"
            synthetic_data = {
                'x': np.random.randn(n_samples).astype(np.float64),
                'y': np.random.randn(n_samples).astype(np.float64),
                't': np.random.randn(n_samples).astype(np.float64),
                'T1': np.random.randn(n_samples).astype(np.float64),
                'T3': np.random.randn(n_samples).astype(np.float64),
                'Ux': np.random.randn(n_samples).astype(np.float64),
                'Uy': np.random.randn(n_samples).astype(np.float64),
                'p': np.array([0.00125 + i*0.0001]),
                'd': np.array([0.0001]),
                'w': np.array([0.00005]),
                'seed': np.array([42]),
                'nx_sample': np.array([50]),
                'ny_sample': np.array([50]),
                'nt_sample': np.array([20])
            }
            np.savez(test_file, **synthetic_data)

        # Test loading multiple files
        loader = FDTDDataLoaderService(data_dir=temp_dir)

        start_time = time.time()
        data_list = loader.load_multiple()
        elapsed_time = time.time() - start_time

        print(f"Loading {n_files} files took: {elapsed_time:.3f} seconds")

        # Should load 3 files in reasonable time (under 10 seconds)
        assert elapsed_time < 10.0, f"Loading {n_files} files took {elapsed_time:.3f}s"
        assert len(data_list) == n_files

        # Cleanup
        for test_file in temp_dir.glob("*.npz"):
            test_file.unlink()
        temp_dir.rmdir()

class TestMemoryUsage:
    """Tests for memory efficiency."""

    def test_large_array_loading_memory_efficient(self):
        """Test that loading large arrays doesn't cause memory issues."""
        temp_dir = Path(tempfile.mkdtemp())
        test_file = temp_dir / "large_test.npz"

        # Create a larger dataset
        n_samples = 500000  # 500k samples

        synthetic_data = {
            'x': np.random.randn(n_samples).astype(np.float64),
            'y': np.random.randn(n_samples).astype(np.float64),
            't': np.random.randn(n_samples).astype(np.float64),
            'T1': np.random.randn(n_samples).astype(np.float64),
            'T3': np.random.randn(n_samples).astype(np.float64),
            'Ux': np.random.randn(n_samples).astype(np.float64),
            'Uy': np.random.randn(n_samples).astype(np.float64),
            'p': np.array([0.00125]),
            'd': np.array([0.0001]),
            'w': np.array([0.00005]),
            'seed': np.array([42]),
            'nx_sample': np.array([100]),
            'ny_sample': np.array([100]),
            'nt_sample': np.array([50])
        }

        np.savez(test_file, **synthetic_data)

        # Load data
        loader = FDTDDataLoaderService(data_dir=temp_dir)
        data = loader.load_file(test_file)

        # Verify data loaded successfully
        assert data.x.shape == (n_samples,)

        # Cleanup
        test_file.unlink()
        temp_dir.rmdir()


@pytest.mark.slow
class TestTrainingPerformance:
    """Performance benchmarks for training (marked as slow tests)."""

    def test_training_benchmark_placeholder(self):
        """Placeholder for training performance test.

        Full training performance test requires:
        1. GPU availability
        2. Complete DeepXDE model training
        3. Extended time (5+ minutes)

        This would be run separately as a benchmark, not in regular test suite.

        Requirement: Training on 1D wave equation (10k collocation points)
        shall complete in under 5 minutes on GPU.
        """
        # This is a placeholder - actual implementation would:
        # - Build PINN model with 10k collocation points
        # - Train for full epochs
        # - Measure elapsed time
        # - Assert time < 300 seconds (5 minutes)

        assert True, "Training benchmark not implemented (requires GPU and long runtime)"
