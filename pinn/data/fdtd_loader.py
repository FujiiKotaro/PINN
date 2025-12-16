"""FDTD data loader for .npz files.

This module provides utilities to load FDTD simulation data from .npz files,
extract spatiotemporal coordinates, wave field data, and metadata.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FDTDData:
    """Container for FDTD simulation data from .npz file.

    Attributes:
        x: Spatial x-coordinates (nt*nx*ny,)
        y: Spatial y-coordinates (nt*nx*ny,)
        t: Temporal coordinates (nt*nx*ny,)
        T1: Normal stress component 1 (nt*nx*ny,)
        T3: Normal stress component 3 (nt*nx*ny,)
        Ux: Particle velocity x-component (nt*nx*ny,)
        Uy: Particle velocity y-component (nt*nx*ny,)
        pitch: Crack pitch in meters
        depth: Crack depth in meters
        width: Crack width in meters
        seed: Random seed used for sampling
        nx_sample: Number of spatial samples in x-direction
        ny_sample: Number of spatial samples in y-direction
        nt_sample: Number of temporal samples
    """
    # Spatiotemporal coordinates
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    # Wave field data
    T1: np.ndarray  # Normal stress component 1
    T3: np.ndarray  # Normal stress component 3
    Ux: np.ndarray  # Particle velocity x-component
    Uy: np.ndarray  # Particle velocity y-component

    # Metadata
    pitch: float  # Crack pitch (m)
    depth: float  # Crack depth (m)
    width: float  # Crack width (m)
    seed: int     # Random seed used for sampling

    # Sampling info
    nx_sample: int
    ny_sample: int
    nt_sample: int


class FDTDDataLoaderService:
    """Service for loading and validating FDTD data from .npz files."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize FDTD data loader.

        Args:
            data_dir: Directory containing .npz files (default: /PINN_data/)
        """
        self.data_dir = data_dir or Path("/PINN_data/")

    def load_file(self, file_path: Path) -> FDTDData:
        """Load single .npz file and extract all fields.

        Args:
            file_path: Path to .npz file

        Returns:
            FDTDData container with all fields

        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If required keys missing
            ValueError: If array shapes malformed
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load .npz file
        data = np.load(file_path)

        # Required keys for FDTD data
        required_keys = [
            'x', 'y', 't', 'T1', 'T3', 'Ux', 'Uy',
            'p', 'd', 'w', 'seed',
            'nx_sample', 'ny_sample', 'nt_sample'
        ]

        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in data.keys()]
        if missing_keys:
            raise KeyError(f"Missing required keys in .npz file: {missing_keys}")

        # Extract data into FDTDData container
        fdtd_data = FDTDData(
            # Spatiotemporal coordinates
            x=data['x'],
            y=data['y'],
            t=data['t'],
            # Wave field data
            T1=data['T1'],
            T3=data['T3'],
            Ux=data['Ux'],
            Uy=data['Uy'],
            # Metadata (convert from numpy scalars to Python types)
            pitch=float(data['p']),
            depth=float(data['d']),
            width=float(data['w']),
            seed=int(data['seed']),
            # Sampling info
            nx_sample=int(data['nx_sample']),
            ny_sample=int(data['ny_sample']),
            nt_sample=int(data['nt_sample']),
        )

        return fdtd_data

    def validate_data(self, data: FDTDData) -> None:
        """Validate data shapes and metadata consistency.

        Args:
            data: FDTDData object to validate

        Raises:
            ValueError: If validation fails
        """
        # Expected size based on sampling info
        expected_size = data.nx_sample * data.ny_sample * data.nt_sample

        # Validate spatiotemporal coordinates
        if data.x.shape != (expected_size,):
            raise ValueError(
                f"Invalid x shape: expected ({expected_size},), got {data.x.shape}"
            )
        if data.y.shape != (expected_size,):
            raise ValueError(
                f"Invalid y shape: expected ({expected_size},), got {data.y.shape}"
            )
        if data.t.shape != (expected_size,):
            raise ValueError(
                f"Invalid t shape: expected ({expected_size},), got {data.t.shape}"
            )

        # Validate wave field data
        if data.T1.shape != (expected_size,):
            raise ValueError(
                f"Invalid T1 shape: expected ({expected_size},), got {data.T1.shape}"
            )
        if data.T3.shape != (expected_size,):
            raise ValueError(
                f"Invalid T3 shape: expected ({expected_size},), got {data.T3.shape}"
            )
        if data.Ux.shape != (expected_size,):
            raise ValueError(
                f"Invalid Ux shape: expected ({expected_size},), got {data.Ux.shape}"
            )
        if data.Uy.shape != (expected_size,):
            raise ValueError(
                f"Invalid Uy shape: expected ({expected_size},), got {data.Uy.shape}"
            )

        # Validate metadata (non-negative values)
        if data.pitch <= 0:
            raise ValueError(f"Invalid pitch: {data.pitch}, must be > 0")
        if data.depth <= 0:
            raise ValueError(f"Invalid depth: {data.depth}, must be > 0")
        if data.width <= 0:
            raise ValueError(f"Invalid width: {data.width}, must be > 0")

        # Validate sampling info (positive integers)
        if data.nx_sample <= 0:
            raise ValueError(f"Invalid nx_sample: {data.nx_sample}, must be > 0")
        if data.ny_sample <= 0:
            raise ValueError(f"Invalid ny_sample: {data.ny_sample}, must be > 0")
        if data.nt_sample <= 0:
            raise ValueError(f"Invalid nt_sample: {data.nt_sample}, must be > 0")

    def load_multiple(
        self,
        pitch_range: tuple[float, float] | None = None,
        depth_range: tuple[float, float] | None = None
    ) -> list[FDTDData]:
        """Load multiple .npz files filtered by parameter ranges.

        Args:
            pitch_range: (min_pitch, max_pitch) in meters
            depth_range: (min_depth, max_depth) in meters

        Returns:
            List of FDTDData objects

        Raises:
            ValueError: If data directory doesn't exist or no files found
        """
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        # Find all .npz files in data directory
        npz_files = sorted(self.data_dir.glob("p*_d*.npz"))

        if not npz_files:
            raise ValueError(f"No .npz files found in {self.data_dir}")

        # Load and filter files
        loaded_data = []
        for npz_file in npz_files:
            try:
                data = self.load_file(npz_file)

                # Apply filters if specified
                if pitch_range is not None:
                    min_pitch, max_pitch = pitch_range
                    if not (min_pitch <= data.pitch <= max_pitch):
                        continue

                if depth_range is not None:
                    min_depth, max_depth = depth_range
                    if not (min_depth <= data.depth <= max_depth):
                        continue

                # Validate before adding
                self.validate_data(data)
                loaded_data.append(data)

            except (KeyError, ValueError) as e:
                # Log warning but continue with other files
                print(f"Warning: Skipping {npz_file.name}: {e}")
                continue

        return loaded_data
