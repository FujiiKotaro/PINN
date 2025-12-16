"""Unit tests for ExperimentManager."""
import json
from pathlib import Path

from pinn.utils.experiment_manager import ExperimentManager


class TestExperimentManager:
    """Test experiment directory organization functionality."""

    def test_create_experiment_directory_with_timestamp(self, tmp_path: Path) -> None:
        """ExperimentManager should create timestamped experiment directory."""
        # Arrange
        manager = ExperimentManager(base_dir=tmp_path)

        # Act
        exp_dir = manager.create_experiment_directory(experiment_name="test_exp")

        # Assert
        assert exp_dir.exists()
        assert exp_dir.is_dir()
        # Directory name should contain experiment name and timestamp
        assert "test_exp" in exp_dir.name
        # Should be under base_dir
        assert exp_dir.parent == tmp_path

    def test_create_subdirectories(self, tmp_path: Path) -> None:
        """ExperimentManager should create required subdirectories."""
        # Arrange
        manager = ExperimentManager(base_dir=tmp_path)

        # Act
        exp_dir = manager.create_experiment_directory(experiment_name="test_exp")

        # Assert
        assert (exp_dir / "checkpoints").exists()
        assert (exp_dir / "checkpoints").is_dir()
        assert (exp_dir / "logs").exists()
        assert (exp_dir / "logs").is_dir()
        assert (exp_dir / "plots").exists()
        assert (exp_dir / "plots").is_dir()

    def test_save_config_to_experiment_directory(self, tmp_path: Path) -> None:
        """ExperimentManager should save config YAML to experiment directory."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        manager = ExperimentManager(base_dir=tmp_path)

        config = ExperimentConfig(
            experiment_name="test_config",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(),
        )

        exp_dir = manager.create_experiment_directory(experiment_name="test_config")

        # Act
        manager.save_config(config, exp_dir)

        # Assert
        config_path = exp_dir / "config.yaml"
        assert config_path.exists()

        # Verify content can be reloaded
        from pinn.utils.config_loader import ConfigLoaderService

        reloaded_config = ConfigLoaderService.load_config(str(config_path))
        assert reloaded_config.experiment_name == "test_config"
        assert reloaded_config.seed == 42

    def test_auto_generate_experiment_name(self, tmp_path: Path) -> None:
        """ExperimentManager should auto-generate experiment name if not provided."""
        # Arrange
        manager = ExperimentManager(base_dir=tmp_path)

        # Act
        exp_dir = manager.create_experiment_directory()

        # Assert
        assert exp_dir.exists()
        # Name should be auto-generated with timestamp
        assert exp_dir.name.startswith("exp_")

    def test_experiment_directory_timestamp_format(self, tmp_path: Path) -> None:
        """Experiment directory timestamp should be in readable format."""
        # Arrange
        manager = ExperimentManager(base_dir=tmp_path)

        # Act
        exp_dir = manager.create_experiment_directory(experiment_name="timestamp_test")

        # Assert
        # Directory name should contain timestamp in YYYY-MM-DD_HH-MM-SS format
        dir_name = exp_dir.name
        # Should contain date separator
        assert "_" in dir_name
        # Should contain experiment name
        assert "timestamp_test" in dir_name

    def test_get_experiment_paths(self, tmp_path: Path) -> None:
        """ExperimentManager should provide helper to get subdirectory paths."""
        # Arrange
        manager = ExperimentManager(base_dir=tmp_path)
        exp_dir = manager.create_experiment_directory(experiment_name="path_test")

        # Act
        paths = manager.get_experiment_paths(exp_dir)

        # Assert
        assert "checkpoints" in paths
        assert "logs" in paths
        assert "plots" in paths
        assert paths["checkpoints"] == exp_dir / "checkpoints"
        assert paths["logs"] == exp_dir / "logs"
        assert paths["plots"] == exp_dir / "plots"

    def test_save_metadata_to_experiment_directory(self, tmp_path: Path) -> None:
        """ExperimentManager should save metadata.json to experiment directory."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )
        from pinn.utils.metadata_logger import MetadataLogger

        manager = ExperimentManager(base_dir=tmp_path)
        logger = MetadataLogger()

        config = ExperimentConfig(
            experiment_name="metadata_test",
            seed=99,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(),
        )

        exp_dir = manager.create_experiment_directory(experiment_name="metadata_test")

        # Act
        metadata = logger.capture_full_metadata(config)
        manager.save_metadata(metadata, exp_dir)

        # Assert
        metadata_path = exp_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify content
        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        assert saved_metadata["seed"] == 99
        assert saved_metadata["experiment_name"] == "metadata_test"
