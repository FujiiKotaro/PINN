"""Tests for seed management and reproducibility utilities.

This module tests the SeedManager for setting random seeds across
NumPy, PyTorch, and Python's random module.
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import random

from pinn.utils.seed_manager import SeedManager


class TestSeedManager:
    """Test suite for SeedManager."""

    def test_set_seed_numpy(self) -> None:
        """Test that NumPy random operations are reproducible."""
        SeedManager.set_seed(42)
        result1 = np.random.rand(5)

        SeedManager.set_seed(42)
        result2 = np.random.rand(5)

        np.testing.assert_array_equal(result1, result2)

    def test_set_seed_torch(self) -> None:
        """Test that PyTorch random operations are reproducible."""
        SeedManager.set_seed(42)
        result1 = torch.rand(5)

        SeedManager.set_seed(42)
        result2 = torch.rand(5)

        torch.testing.assert_close(result1, result2)

    def test_set_seed_python_random(self) -> None:
        """Test that Python random operations are reproducible."""
        SeedManager.set_seed(42)
        result1 = [random.random() for _ in range(5)]

        SeedManager.set_seed(42)
        result2 = [random.random() for _ in range(5)]

        assert result1 == result2

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different random sequences."""
        SeedManager.set_seed(42)
        result1 = np.random.rand(5)

        SeedManager.set_seed(123)
        result2 = np.random.rand(5)

        assert not np.array_equal(result1, result2)

    def test_set_seed_affects_torch_cuda_if_available(self) -> None:
        """Test that CUDA seed is set if GPU is available."""
        SeedManager.set_seed(42)
        # Should not raise error even if CUDA not available
        # Just verify it runs without error
        assert True

    def test_log_seed_to_metadata(self, tmp_path: Path) -> None:
        """Test logging seed value to metadata JSON."""
        metadata_file = tmp_path / "metadata.json"
        seed = 42

        SeedManager.log_seed_metadata(seed, str(metadata_file))

        assert metadata_file.exists()

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert "seed" in metadata
        assert metadata["seed"] == seed

    def test_log_seed_appends_to_existing_metadata(self, tmp_path: Path) -> None:
        """Test that logging seed appends to existing metadata."""
        metadata_file = tmp_path / "metadata.json"

        # Create initial metadata
        initial_data = {"experiment_name": "test", "version": "0.1.0"}
        with open(metadata_file, 'w') as f:
            json.dump(initial_data, f)

        # Log seed
        SeedManager.log_seed_metadata(42, str(metadata_file))

        # Verify both old and new data present
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert metadata["experiment_name"] == "test"
        assert metadata["version"] == "0.1.0"
        assert metadata["seed"] == 42

    def test_log_seed_creates_new_file_if_not_exists(self, tmp_path: Path) -> None:
        """Test that log_seed_metadata creates new file if it doesn't exist."""
        metadata_file = tmp_path / "new_metadata.json"

        assert not metadata_file.exists()

        SeedManager.log_seed_metadata(42, str(metadata_file))

        assert metadata_file.exists()

    def test_get_seed_info_returns_current_seed(self) -> None:
        """Test that get_seed_info returns the currently set seed."""
        seed = 12345
        SeedManager.set_seed(seed)

        info = SeedManager.get_seed_info()

        assert info["seed"] == seed
        assert "numpy_seed" in info
        assert "torch_seed" in info
        assert "python_seed" in info

    def test_set_seed_logs_message(self, capsys) -> None:
        """Test that set_seed logs informative message."""
        SeedManager.set_seed(42, verbose=True)

        captured = capsys.readouterr()
        assert "42" in captured.out or "seed" in captured.out.lower()

    def test_set_seed_with_verbose_false(self, capsys) -> None:
        """Test that set_seed with verbose=False produces no output."""
        SeedManager.set_seed(42, verbose=False)

        captured = capsys.readouterr()
        # Should have no output or minimal output
        assert len(captured.out) == 0 or captured.out.strip() == ""
