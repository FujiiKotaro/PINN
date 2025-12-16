"""Metadata logging utility for experiment reproducibility.

This module provides functionality to capture and log software versions,
configuration hashes, seeds, and timestamps for reproducible experiments.
"""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import deepxde as dde
import numpy as np
import torch

from pinn.utils.config_loader import ExperimentConfig


class MetadataLogger:
    """Logger for capturing experiment metadata and software versions.

    This utility captures Python, PyTorch, DeepXDE, and NumPy versions,
    along with experiment configuration hashes, seeds, and timestamps
    to ensure reproducibility.
    """

    def capture_versions(self) -> dict[str, str]:
        """Capture software versions for all dependencies.

        Returns:
            Dictionary containing version strings for:
                - python_version: Python interpreter version
                - torch_version: PyTorch version
                - deepxde_version: DeepXDE version
                - numpy_version: NumPy version

        Example:
            >>> logger = MetadataLogger()
            >>> versions = logger.capture_versions()
            >>> print(versions["python_version"])
            '3.11.5'
        """
        return {
            "python_version": sys.version.split()[0],  # Extract version number only
            "torch_version": torch.__version__,
            "deepxde_version": dde.__version__,
            "numpy_version": np.__version__,
        }

    def save_metadata(self, metadata: dict[str, Any], output_path: Path) -> None:
        """Save metadata dictionary to JSON file.

        Args:
            metadata: Dictionary containing metadata fields
            output_path: Path to output JSON file

        Example:
            >>> logger = MetadataLogger()
            >>> metadata = {"seed": 42, "timestamp": "2025-12-15T12:00:00Z"}
            >>> logger.save_metadata(metadata, Path("metadata.json"))
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def capture_full_metadata(self, config: ExperimentConfig) -> dict[str, Any]:
        """Capture complete metadata including versions, seed, timestamp, and config hash.

        Args:
            config: Experiment configuration object

        Returns:
            Dictionary containing:
                - Software versions (python, torch, deepxde, numpy)
                - seed: Random seed from config
                - timestamp: ISO 8601 timestamp of metadata capture
                - config_hash: SHA256 hash of config for reproducibility
                - experiment_name: Name from config

        Example:
            >>> logger = MetadataLogger()
            >>> config = ExperimentConfig(experiment_name="test", seed=42, ...)
            >>> metadata = logger.capture_full_metadata(config)
            >>> print(metadata["seed"])
            42
        """
        metadata = self.capture_versions()

        # Add experiment-specific fields
        metadata["seed"] = config.seed
        metadata["experiment_name"] = config.experiment_name
        metadata["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        metadata["config_hash"] = self._compute_config_hash(config)

        return metadata

    def _compute_config_hash(self, config: ExperimentConfig) -> str:
        """Compute SHA256 hash of configuration for reproducibility tracking.

        Args:
            config: Experiment configuration object

        Returns:
            Hexadecimal SHA256 hash string

        Note:
            Hash is computed from JSON-serialized config to ensure deterministic
            ordering of fields.
        """
        # Serialize config to JSON string for hashing (Pydantic v2 uses model_dump + json.dumps)
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True)
        config_bytes = config_json.encode("utf-8")

        # Compute SHA256 hash
        hash_obj = hashlib.sha256(config_bytes)
        return hash_obj.hexdigest()
