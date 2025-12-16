"""Experiment directory organization and management.

This module provides utilities for creating timestamped experiment directories,
organizing output files, and saving configuration and metadata.
"""
from datetime import datetime
from pathlib import Path
from typing import Any

from pinn.utils.config_loader import ConfigLoaderService, ExperimentConfig


class ExperimentManager:
    """Manager for experiment directory creation and organization.

    This class handles:
    - Creating timestamped experiment directories
    - Creating subdirectories (checkpoints/, logs/, plots/)
    - Saving configuration YAML files
    - Saving metadata JSON files

    Attributes:
        base_dir: Base directory for all experiments (default: ./experiments/)
    """

    def __init__(self, base_dir: Path | str = Path("./experiments")):
        """Initialize ExperimentManager.

        Args:
            base_dir: Base directory for experiments. Defaults to ./experiments/
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_directory(
        self,
        experiment_name: str | None = None,
    ) -> Path:
        """Create timestamped experiment directory with subdirectories.

        Args:
            experiment_name: Optional experiment name. If not provided,
                auto-generates name as "exp_{timestamp}".

        Returns:
            Path to created experiment directory.

        Example:
            >>> manager = ExperimentManager()
            >>> exp_dir = manager.create_experiment_directory("my_experiment")
            >>> print(exp_dir)
            ./experiments/my_experiment_2025-12-15_12-30-45/
        """
        # Generate directory name with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if experiment_name:
            dir_name = f"{experiment_name}_{timestamp}"
        else:
            dir_name = f"exp_{timestamp}"

        # Create experiment directory
        exp_dir = self.base_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)

        return exp_dir

    def get_experiment_paths(self, exp_dir: Path) -> dict[str, Path]:
        """Get paths to experiment subdirectories.

        Args:
            exp_dir: Experiment directory path

        Returns:
            Dictionary mapping subdirectory names to paths:
                - checkpoints: Path to checkpoints/
                - logs: Path to logs/
                - plots: Path to plots/

        Example:
            >>> manager = ExperimentManager()
            >>> exp_dir = manager.create_experiment_directory("test")
            >>> paths = manager.get_experiment_paths(exp_dir)
            >>> checkpoint_dir = paths["checkpoints"]
        """
        return {
            "checkpoints": exp_dir / "checkpoints",
            "logs": exp_dir / "logs",
            "plots": exp_dir / "plots",
        }

    def save_config(self, config: ExperimentConfig, exp_dir: Path) -> None:
        """Save experiment configuration to YAML file.

        Args:
            config: Experiment configuration object
            exp_dir: Experiment directory path

        Saves:
            config.yaml in experiment directory

        Example:
            >>> manager = ExperimentManager()
            >>> exp_dir = manager.create_experiment_directory("test")
            >>> manager.save_config(config, exp_dir)
        """
        config_path = exp_dir / "config.yaml"
        ConfigLoaderService.save_config(config, str(config_path))

    def save_metadata(self, metadata: dict[str, Any], exp_dir: Path) -> None:
        """Save experiment metadata to JSON file.

        Args:
            metadata: Metadata dictionary (typically from MetadataLogger)
            exp_dir: Experiment directory path

        Saves:
            metadata.json in experiment directory

        Example:
            >>> from pinn.utils.metadata_logger import MetadataLogger
            >>> manager = ExperimentManager()
            >>> logger = MetadataLogger()
            >>> exp_dir = manager.create_experiment_directory("test")
            >>> metadata = logger.capture_full_metadata(config)
            >>> manager.save_metadata(metadata, exp_dir)
        """
        from pinn.utils.metadata_logger import MetadataLogger

        metadata_path = exp_dir / "metadata.json"
        MetadataLogger().save_metadata(metadata, metadata_path)
