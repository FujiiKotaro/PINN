"""FDTD data loader for .npz files.

This module provides utilities to load FDTD simulation data from .npz files,
extract spatiotemporal coordinates, wave field data, and metadata.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinn.data.parameter_normalizer import ParameterNormalizer


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


@dataclass
class FDTDDataset2D:
    """Container for 2D parametric PINN dataset from multiple FDTD files.

    This dataset combines data from multiple .npz files with different crack
    parameters (pitch, depth), providing parametric training data for conditional PINN.

    Attributes:
        x: Spatial x-coordinates (N_total,) in meters
        y: Spatial y-coordinates (N_total,) in meters
        t: Temporal coordinates (N_total,) in seconds
        pitch_norm: Normalized pitch parameters (N_total,) in [0, 1]
        depth_norm: Normalized depth parameters (N_total,) in [0, 1]
        T1: Normal stress component 1 (N_total,) in Pa
        T3: Normal stress component 3 (N_total,) in Pa
        Ux: Particle velocity x-component (N_total,) in m/s
        Uy: Particle velocity y-component (N_total,) in m/s
        metadata: Dictionary with file paths and original parameters

    where N_total = sum of samples from all loaded files
    """
    # Spatiotemporal coordinates (physical units)
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    # Normalized crack parameters (replicated for each spatiotemporal point)
    pitch_norm: np.ndarray  # [0, 1]
    depth_norm: np.ndarray  # [0, 1]

    # Wave field data (physical units)
    T1: np.ndarray
    T3: np.ndarray
    Ux: np.ndarray
    Uy: np.ndarray

    # Metadata
    metadata: dict  # Contains 'files' list and 'params' list


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

    def load_multiple_files(
        self,
        file_paths: list[Path],
        pitch_range: tuple[float, float] | None = None,
        depth_range: tuple[float, float] | None = None,
        apply_dimensionless: bool = False,
        scaler: 'DimensionlessScalerService | None' = None
    ) -> FDTDDataset2D:
        """Load and concatenate multiple .npz files into parametric dataset.

        Args:
            file_paths: List of paths to .npz files
            pitch_range: Optional (min, max) pitch filter in meters
                        Default: (1.25e-3, 2.0e-3)
            depth_range: Optional (min, max) depth filter in meters
                        Default: (0.1e-3, 0.3e-3)
            apply_dimensionless: If True, apply dimensionless scaling to coordinates
                                and wave fields (default: False)
            scaler: DimensionlessScalerService for scaling (required if apply_dimensionless=True)

        Returns:
            FDTDDataset2D with concatenated data from all files
            If apply_dimensionless=True, all coordinates and fields are O(1) scale

        Raises:
            ValueError: If file_paths is empty, or if apply_dimensionless=True but scaler=None
            FileNotFoundError: If any file doesn't exist

        Preconditions:
            - All files must exist and have valid .npz format
            - file_paths must not be empty
            - If apply_dimensionless=True, scaler must be provided

        Postconditions:
            - Returns dataset with N_total samples (sum of all files)
            - pitch_norm, depth_norm in [0, 1] range
            - Parameters replicated for each spatiotemporal point
            - If apply_dimensionless=True: x, y, t, T1, T3, Ux, Uy are O(1) scale

        Example:
            >>> loader = FDTDDataLoaderService()
            >>> files = [Path("p1250_d100.npz"), Path("p1500_d200.npz")]
            >>> dataset = loader.load_multiple_files(files)
            >>> dataset.x.shape  # (N_total,)

            >>> # With dimensionless scaling
            >>> from pinn.data.dimensionless_scaler import CharacteristicScales, DimensionlessScalerService
            >>> scales = CharacteristicScales.from_physics(...)
            >>> scaler = DimensionlessScalerService(scales)
            >>> dataset = loader.load_multiple_files(files, apply_dimensionless=True, scaler=scaler)
        """
        if not file_paths:
            raise ValueError("No files provided")

        # Validate dimensionless scaling requirements
        if apply_dimensionless and scaler is None:
            raise ValueError("scaler must be provided when apply_dimensionless=True")

        # Use default ranges if not specified
        if pitch_range is None:
            pitch_range = (ParameterNormalizer.PITCH_MIN,
                          ParameterNormalizer.PITCH_MAX)
        if depth_range is None:
            depth_range = (ParameterNormalizer.DEPTH_MIN,
                          ParameterNormalizer.DEPTH_MAX)

        # Load individual files
        loaded_files = []
        metadata_files = []
        metadata_params = []

        for file_path in file_paths:
            # Load file
            data = self.load_file(file_path)

            # Filter by parameter ranges
            min_pitch, max_pitch = pitch_range
            if not (min_pitch <= data.pitch <= max_pitch):
                continue

            min_depth, max_depth = depth_range
            if not (min_depth <= data.depth <= max_depth):
                continue

            # Validate
            self.validate_data(data)

            loaded_files.append(data)
            metadata_files.append(str(file_path))
            metadata_params.append({
                'pitch': data.pitch,
                'depth': data.depth,
                'seed': data.seed
            })

        if not loaded_files:
            raise ValueError(
                f"No files passed filters: pitch {pitch_range}, depth {depth_range}"
            )

        # Concatenate data from all files
        all_x = []
        all_y = []
        all_t = []
        all_pitch_norm = []
        all_depth_norm = []
        all_T1 = []
        all_T3 = []
        all_Ux = []
        all_Uy = []

        for data in loaded_files:
            n_samples = len(data.x)

            # Normalize parameters
            pitch_norm = ParameterNormalizer.normalize_pitch(
                np.array([data.pitch])
            )[0]
            depth_norm = ParameterNormalizer.normalize_depth(
                np.array([data.depth])
            )[0]

            # Replicate normalized parameters for each spatiotemporal point
            pitch_norm_replicated = np.full(n_samples, pitch_norm)
            depth_norm_replicated = np.full(n_samples, depth_norm)

            # Append to lists
            all_x.append(data.x)
            all_y.append(data.y)
            all_t.append(data.t)
            all_pitch_norm.append(pitch_norm_replicated)
            all_depth_norm.append(depth_norm_replicated)
            all_T1.append(data.T1)
            all_T3.append(data.T3)
            all_Ux.append(data.Ux)
            all_Uy.append(data.Uy)

        # Concatenate all arrays
        x_concat = np.concatenate(all_x)
        y_concat = np.concatenate(all_y)
        t_concat = np.concatenate(all_t)
        T1_concat = np.concatenate(all_T1)
        T3_concat = np.concatenate(all_T3)
        Ux_concat = np.concatenate(all_Ux)
        Uy_concat = np.concatenate(all_Uy)

        # Apply dimensionless scaling if requested
        if apply_dimensionless:
            # Normalize spatiotemporal coordinates
            x_concat, y_concat, t_concat = scaler.normalize_inputs(
                x_concat, y_concat, t_concat
            )

            # Normalize wave fields
            T1_concat, T3_concat, Ux_concat, Uy_concat = scaler.normalize_outputs(
                T1_concat, T3_concat, Ux_concat, Uy_concat
            )

        # Create dataset
        dataset = FDTDDataset2D(
            x=x_concat,
            y=y_concat,
            t=t_concat,
            pitch_norm=np.concatenate(all_pitch_norm),
            depth_norm=np.concatenate(all_depth_norm),
            T1=T1_concat,
            T3=T3_concat,
            Ux=Ux_concat,
            Uy=Uy_concat,
            metadata={
                'files': metadata_files,
                'params': metadata_params
            }
        )

        return dataset

    def train_val_split(
        self,
        dataset: FDTDDataset2D,
        train_ratio: float = 0.8,
        seed: int = 42,
        validation_equals_train: bool = False
    ) -> tuple[FDTDDataset2D, FDTDDataset2D]:
        """Split dataset into train/validation sets.

        Args:
            dataset: FDTDDataset2D to split
            train_ratio: Fraction of data for training (default: 0.8)
            seed: Random seed for reproducibility (default: 42)
            validation_equals_train: If True, validation uses same data as training
                                    (for overfitting monitoring) (default: False)

        Returns:
            (train_dataset, val_dataset) both as FDTDDataset2D

        Raises:
            ValueError: If train_ratio not in (0, 1]

        Preconditions:
            - dataset must be valid FDTDDataset2D
            - train_ratio must be in (0, 1]

        Postconditions:
            - len(train) + len(val) == len(dataset) (unless validation_equals_train=True)
            - Both datasets preserve all fields
            - Metadata includes split_info with train_ratio and seed
            - Splitting is reproducible with same seed

        Example:
            >>> loader = FDTDDataLoaderService()
            >>> dataset = loader.load_multiple_files([Path("p1250_d100.npz")])
            >>> train, val = loader.train_val_split(dataset, train_ratio=0.8, seed=42)
            >>> len(train.x) + len(val.x) == len(dataset.x)  # True
        """
        # Validate train_ratio
        if train_ratio <= 0 or train_ratio > 1:
            raise ValueError("train_ratio must be between 0 and 1")

        # Special case: validation_equals_train
        if validation_equals_train and train_ratio == 1.0:
            # Both train and val use all data
            train_metadata = {
                **dataset.metadata,
                'split_info': {'train_ratio': train_ratio, 'seed': seed}
            }
            val_metadata = {
                **dataset.metadata,
                'split_info': {'train_ratio': train_ratio, 'seed': seed}
            }

            train_dataset = FDTDDataset2D(
                x=dataset.x.copy(),
                y=dataset.y.copy(),
                t=dataset.t.copy(),
                pitch_norm=dataset.pitch_norm.copy(),
                depth_norm=dataset.depth_norm.copy(),
                T1=dataset.T1.copy(),
                T3=dataset.T3.copy(),
                Ux=dataset.Ux.copy(),
                Uy=dataset.Uy.copy(),
                metadata=train_metadata
            )

            val_dataset = FDTDDataset2D(
                x=dataset.x.copy(),
                y=dataset.y.copy(),
                t=dataset.t.copy(),
                pitch_norm=dataset.pitch_norm.copy(),
                depth_norm=dataset.depth_norm.copy(),
                T1=dataset.T1.copy(),
                T3=dataset.T3.copy(),
                Ux=dataset.Ux.copy(),
                Uy=dataset.Uy.copy(),
                metadata=val_metadata
            )

            return train_dataset, val_dataset

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Get total size
        N_total = len(dataset.x)

        # Calculate split sizes
        N_train = int(N_total * train_ratio)

        # Create random permutation of indices
        indices = np.random.permutation(N_total)

        # Split indices
        train_indices = indices[:N_train]
        val_indices = indices[N_train:]

        # Create metadata with split info
        train_metadata = {
            **dataset.metadata,
            'split_info': {'train_ratio': train_ratio, 'seed': seed}
        }

        val_metadata = {
            **dataset.metadata,
            'split_info': {'train_ratio': train_ratio, 'seed': seed}
        }

        # Create train dataset
        train_dataset = FDTDDataset2D(
            x=dataset.x[train_indices],
            y=dataset.y[train_indices],
            t=dataset.t[train_indices],
            pitch_norm=dataset.pitch_norm[train_indices],
            depth_norm=dataset.depth_norm[train_indices],
            T1=dataset.T1[train_indices],
            T3=dataset.T3[train_indices],
            Ux=dataset.Ux[train_indices],
            Uy=dataset.Uy[train_indices],
            metadata=train_metadata
        )

        # Create val dataset
        val_dataset = FDTDDataset2D(
            x=dataset.x[val_indices],
            y=dataset.y[val_indices],
            t=dataset.t[val_indices],
            pitch_norm=dataset.pitch_norm[val_indices],
            depth_norm=dataset.depth_norm[val_indices],
            T1=dataset.T1[val_indices],
            T3=dataset.T3[val_indices],
            Ux=dataset.Ux[val_indices],
            Uy=dataset.Uy[val_indices],
            metadata=val_metadata
        )

        return train_dataset, val_dataset
