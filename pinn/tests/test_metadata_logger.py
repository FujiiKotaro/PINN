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


class TestMetadataLoggerExtensions:
    """Test Task 8.2: FDTD metadata extensions."""

    def test_capture_fdtd_files(self) -> None:
        """MetadataLogger should capture FDTD file list."""
        from pathlib import Path
        logger = MetadataLogger()

        fdtd_files = [Path("p1250_d100.npz"), Path("p1500_d150.npz")]
        metadata = logger.capture_fdtd_metadata(fdtd_files)

        assert "fdtd_files" in metadata
        assert len(metadata["fdtd_files"]) == 2
        assert "p1250_d100.npz" in metadata["fdtd_files"]
        assert "p1500_d150.npz" in metadata["fdtd_files"]

    def test_capture_parameter_ranges(self) -> None:
        """MetadataLogger should capture parameter ranges (pitch, depth)."""
        logger = MetadataLogger()

        pitch_values = [1.25e-3, 1.5e-3, 1.75e-3, 2.0e-3]
        depth_values = [0.1e-3, 0.15e-3, 0.2e-3]

        metadata = logger.capture_fdtd_metadata(
            fdtd_files=[],
            pitch_range=(min(pitch_values), max(pitch_values)),
            depth_range=(min(depth_values), max(depth_values))
        )

        assert "parameter_ranges" in metadata
        assert "pitch" in metadata["parameter_ranges"]
        assert "depth" in metadata["parameter_ranges"]

        assert metadata["parameter_ranges"]["pitch"]["min"] == 1.25e-3
        assert metadata["parameter_ranges"]["pitch"]["max"] == 2.0e-3
        assert metadata["parameter_ranges"]["depth"]["min"] == 0.1e-3
        assert metadata["parameter_ranges"]["depth"]["max"] == 0.2e-3

    def test_capture_characteristic_scales(self) -> None:
        """MetadataLogger should capture characteristic scales (L_ref, T_ref, U_ref, σ_ref)."""
        logger = MetadataLogger()

        from pinn.data.dimensionless_scaler import CharacteristicScales

        scales = CharacteristicScales(
            L_ref=0.04,
            T_ref=6.78e-6,
            U_ref=1.5e-9,
            sigma_ref=1.0e9,
            velocity_ref=5900.0
        )

        metadata = logger.capture_fdtd_metadata(
            fdtd_files=[],
            characteristic_scales=scales
        )

        assert "characteristic_scales" in metadata
        assert metadata["characteristic_scales"]["L_ref"] == 0.04
        assert metadata["characteristic_scales"]["T_ref"] == 6.78e-6
        assert metadata["characteristic_scales"]["U_ref"] == 1.5e-9
        assert metadata["characteristic_scales"]["sigma_ref"] == 1.0e9
        assert metadata["characteristic_scales"]["velocity_ref"] == 5900.0

    def test_capture_complete_fdtd_metadata(self) -> None:
        """Integration test: capture all FDTD metadata together."""
        from pathlib import Path
        from pinn.data.dimensionless_scaler import CharacteristicScales

        logger = MetadataLogger()

        fdtd_files = [Path("p1250_d100.npz"), Path("p1500_d150.npz")]
        scales = CharacteristicScales(
            L_ref=0.04,
            T_ref=6.78e-6,
            U_ref=1.5e-9,
            sigma_ref=1.0e9,
            velocity_ref=5900.0
        )

        metadata = logger.capture_fdtd_metadata(
            fdtd_files=fdtd_files,
            pitch_range=(1.25e-3, 2.0e-3),
            depth_range=(0.1e-3, 0.3e-3),
            characteristic_scales=scales
        )

        # Verify all components present
        assert "fdtd_files" in metadata
        assert "parameter_ranges" in metadata
        assert "characteristic_scales" in metadata

        assert len(metadata["fdtd_files"]) == 2
        assert metadata["parameter_ranges"]["pitch"]["min"] == 1.25e-3
        assert metadata["characteristic_scales"]["L_ref"] == 0.04
