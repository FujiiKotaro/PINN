"""Unit tests for MetadataLogger."""
import json
import sys
from pathlib import Path

import deepxde as dde
import numpy as np
import torch

from pinn.utils.metadata_logger import MetadataLogger


class TestMetadataLogger:
    """Test metadata logging functionality."""

    def test_capture_software_versions(self) -> None:
        """MetadataLogger should capture all required software versions."""
        # Arrange
        logger = MetadataLogger()

        # Act
        metadata = logger.capture_versions()

        # Assert
        assert "python_version" in metadata
        assert "torch_version" in metadata
        assert "deepxde_version" in metadata
        assert "numpy_version" in metadata

        # Verify versions are non-empty strings
        assert isinstance(metadata["python_version"], str)
        assert len(metadata["python_version"]) > 0
        assert metadata["torch_version"] == torch.__version__
        assert metadata["deepxde_version"] == dde.__version__
        assert metadata["numpy_version"] == np.__version__

    def test_capture_python_version_format(self) -> None:
        """Python version should be in readable format."""
        # Arrange
        logger = MetadataLogger()

        # Act
        metadata = logger.capture_versions()

        # Assert
        python_version = metadata["python_version"]
        # Should contain version number (e.g., "3.11.5")
        assert "3." in python_version or "2." in python_version

    def test_save_metadata_to_json(self, tmp_path: Path) -> None:
        """MetadataLogger should save metadata to JSON file."""
        # Arrange
        logger = MetadataLogger()
        output_path = tmp_path / "metadata.json"

        metadata = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "deepxde_version": dde.__version__,
            "numpy_version": np.__version__,
            "seed": 42,
            "timestamp": "2025-12-15T12:00:00Z",
        }

        # Act
        logger.save_metadata(metadata, output_path)

        # Assert
        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            saved_metadata = json.load(f)

        assert saved_metadata == metadata

    def test_capture_full_metadata_with_config(self, tmp_path: Path) -> None:
        """capture_full_metadata should include versions, seed, timestamp, and config hash."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        logger = MetadataLogger()

        config = ExperimentConfig(
            experiment_name="test_experiment",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(epochs=1000),
        )

        # Act
        metadata = logger.capture_full_metadata(config)

        # Assert
        assert "python_version" in metadata
        assert "torch_version" in metadata
        assert "deepxde_version" in metadata
        assert "numpy_version" in metadata
        assert "seed" in metadata
        assert "timestamp" in metadata
        assert "config_hash" in metadata
        assert "experiment_name" in metadata

        # Verify values
        assert metadata["seed"] == 42
        assert metadata["experiment_name"] == "test_experiment"
        assert isinstance(metadata["timestamp"], str)
        assert isinstance(metadata["config_hash"], str)

    def test_config_hash_deterministic(self) -> None:
        """Config hash should be deterministic for same config."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        logger = MetadataLogger()

        config = ExperimentConfig(
            experiment_name="test",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(),
        )

        # Act
        hash1 = logger._compute_config_hash(config)
        hash2 = logger._compute_config_hash(config)

        # Assert
        assert hash1 == hash2

    def test_config_hash_different_for_different_configs(self) -> None:
        """Config hash should differ for different configurations."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        logger = MetadataLogger()

        config1 = ExperimentConfig(
            experiment_name="test1",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(epochs=1000),
        )

        config2 = ExperimentConfig(
            experiment_name="test2",
            seed=43,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="neumann"),
            network=NetworkConfig(),
            training=TrainingConfig(epochs=2000),
        )

        # Act
        hash1 = logger._compute_config_hash(config1)
        hash2 = logger._compute_config_hash(config2)

        # Assert
        assert hash1 != hash2

    def test_save_full_metadata_integration(self, tmp_path: Path) -> None:
        """Integration test: capture and save full metadata."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        logger = MetadataLogger()

        config = ExperimentConfig(
            experiment_name="integration_test",
            seed=123,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(),
        )

        output_path = tmp_path / "full_metadata.json"

        # Act
        metadata = logger.capture_full_metadata(config)
        logger.save_metadata(metadata, output_path)

        # Assert
        assert output_path.exists()

        # Verify content can be reloaded
        with open(output_path) as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata["seed"] == 123
        assert loaded_metadata["experiment_name"] == "integration_test"
        assert "python_version" in loaded_metadata
