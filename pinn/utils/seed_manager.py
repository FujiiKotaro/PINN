"""Seed management for reproducible experiments.

This module provides utilities for setting random seeds across NumPy,
PyTorch, and Python's random module to ensure reproducible results.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


class SeedManager:
    """Utility class for managing random seeds across frameworks."""

    _current_seed: int | None = None

    @classmethod
    def set_seed(cls, seed: int, verbose: bool = False) -> None:
        """Set random seeds for NumPy, PyTorch, and Python's random module.

        Args:
            seed: Random seed value
            verbose: If True, print seed information

        Note:
            This should be called before any random operations (model init,
            data sampling, collocation points) to ensure reproducibility.
        """
        cls._current_seed = seed

        # Set Python's random module seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seed (CPU)
        torch.manual_seed(seed)

        # Set PyTorch random seed (GPU) if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

            # Additional settings for deterministic behavior on CUDA
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if verbose:
            print(f"Random seed set to {seed} for NumPy, PyTorch, and Python random")

    @classmethod
    def get_seed_info(cls) -> dict[str, int]:
        """Get information about currently set seeds.

        Returns:
            Dictionary containing seed information for each framework
        """
        return {
            "seed": cls._current_seed if cls._current_seed is not None else -1,
            "numpy_seed": cls._current_seed if cls._current_seed is not None else -1,
            "torch_seed": cls._current_seed if cls._current_seed is not None else -1,
            "python_seed": cls._current_seed if cls._current_seed is not None else -1,
        }

    @classmethod
    def log_seed_metadata(cls, seed: int, metadata_file: str) -> None:
        """Log seed value to metadata JSON file.

        Args:
            seed: Random seed value to log
            metadata_file: Path to metadata JSON file

        Note:
            If the file exists, seed will be added to existing metadata.
            If the file doesn't exist, a new file will be created.
        """
        metadata_path = Path(metadata_file)

        # Load existing metadata if file exists
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Add seed to metadata
        metadata["seed"] = seed

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
