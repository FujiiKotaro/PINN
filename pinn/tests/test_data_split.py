"""Tests for train/validation data splitting.

Test-Driven Development: Tests written before implementation.
Tests cover train/val split strategies for PINN training.
"""

import numpy as np
import pytest

from pinn.data.fdtd_loader import FDTDDataset2D, FDTDDataLoaderService


class TestTrainValSplit:
    """Test train/validation data splitting functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        N = 1000
        return FDTDDataset2D(
            x=np.random.rand(N),
            y=np.random.rand(N),
            t=np.random.rand(N),
            pitch_norm=np.random.rand(N),
            depth_norm=np.random.rand(N),
            T1=np.random.rand(N),
            T3=np.random.rand(N),
            Ux=np.random.rand(N),
            Uy=np.random.rand(N),
            metadata={'files': ['test.npz'], 'params': [{'pitch': 1.5e-3, 'depth': 0.2e-3}]}
        )

    def test_train_val_split_returns_two_datasets(self, sample_dataset):
        """Test that train_val_split returns train and validation datasets."""
        loader = FDTDDataLoaderService()

        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)

        assert isinstance(train_data, FDTDDataset2D)
        assert isinstance(val_data, FDTDDataset2D)

    def test_train_val_split_respects_ratio(self, sample_dataset):
        """Test that split ratio is respected."""
        loader = FDTDDataLoaderService()
        N_total = len(sample_dataset.x)

        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)

        expected_train = int(N_total * 0.8)
        expected_val = N_total - expected_train

        assert len(train_data.x) == expected_train
        assert len(val_data.x) == expected_val

    def test_train_val_split_is_reproducible(self, sample_dataset):
        """Test that split with same seed produces same result."""
        loader = FDTDDataLoaderService()

        train1, val1 = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)
        train2, val2 = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)

        # Same indices should be selected
        np.testing.assert_array_equal(train1.x, train2.x)
        np.testing.assert_array_equal(val1.x, val2.x)

    def test_train_val_split_different_seeds_differ(self, sample_dataset):
        """Test that different seeds produce different splits."""
        loader = FDTDDataLoaderService()

        train1, val1 = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)
        train2, val2 = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=123)

        # Different seeds should produce different splits
        assert not np.array_equal(train1.x, train2.x)

    def test_train_val_split_no_overlap(self, sample_dataset):
        """Test that train and val sets have no overlap."""
        loader = FDTDDataLoaderService()

        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)

        # Concatenate train and val, should equal original size
        total_size = len(train_data.x) + len(val_data.x)
        assert total_size == len(sample_dataset.x)

    def test_train_val_split_preserves_all_fields(self, sample_dataset):
        """Test that all fields are present in split datasets."""
        loader = FDTDDataLoaderService()

        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)

        # Check train dataset has all fields
        assert len(train_data.x) == len(train_data.y) == len(train_data.t)
        assert len(train_data.pitch_norm) == len(train_data.x)
        assert len(train_data.T1) == len(train_data.T3) == len(train_data.Ux) == len(train_data.Uy)

        # Check val dataset has all fields
        assert len(val_data.x) == len(val_data.y) == len(val_data.t)
        assert len(val_data.pitch_norm) == len(val_data.x)
        assert len(val_data.T1) == len(val_data.T3) == len(val_data.Ux) == len(val_data.Uy)

    def test_train_val_split_with_validation_equals_train(self, sample_dataset):
        """Test special case: validation = training (train_ratio=1.0)."""
        loader = FDTDDataLoaderService()

        # Use train_ratio=1.0 to use all data for training
        train_data, val_data = loader.train_val_split(
            sample_dataset,
            train_ratio=1.0,
            validation_equals_train=True,
            seed=42
        )

        # Both should have same size as original
        assert len(train_data.x) == len(sample_dataset.x)
        assert len(val_data.x) == len(sample_dataset.x)

        # Validation should be identical to training
        np.testing.assert_array_equal(train_data.x, val_data.x)
        np.testing.assert_array_equal(train_data.T1, val_data.T1)

    def test_train_val_split_different_ratios(self, sample_dataset):
        """Test split with different train ratios."""
        loader = FDTDDataLoaderService()
        N_total = len(sample_dataset.x)

        # Test 70/30 split
        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.7, seed=42)
        expected_train = int(N_total * 0.7)
        assert len(train_data.x) == expected_train
        assert len(val_data.x) == N_total - expected_train

        # Test 90/10 split
        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.9, seed=42)
        expected_train = int(N_total * 0.9)
        assert len(train_data.x) == expected_train
        assert len(val_data.x) == N_total - expected_train

    def test_train_val_split_preserves_metadata(self, sample_dataset):
        """Test that metadata is preserved in both splits."""
        loader = FDTDDataLoaderService()

        train_data, val_data = loader.train_val_split(sample_dataset, train_ratio=0.8, seed=42)

        # Metadata should be present
        assert 'files' in train_data.metadata
        assert 'files' in val_data.metadata
        assert 'split_info' in train_data.metadata
        assert 'split_info' in val_data.metadata

        # Split info should contain train_ratio and seed
        assert train_data.metadata['split_info']['train_ratio'] == 0.8
        assert train_data.metadata['split_info']['seed'] == 42

    def test_train_val_split_raises_error_for_invalid_ratio(self, sample_dataset):
        """Test that invalid train_ratio raises ValueError."""
        loader = FDTDDataLoaderService()

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            loader.train_val_split(sample_dataset, train_ratio=1.5, seed=42)

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            loader.train_val_split(sample_dataset, train_ratio=-0.1, seed=42)
