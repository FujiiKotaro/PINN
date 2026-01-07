"""Configuration loading with Pydantic validation.

This module provides Pydantic models for PINN experiment configuration
and a ConfigLoaderService for loading/saving YAML configuration files.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DomainConfig(BaseModel):
    """Configuration for spatiotemporal domain."""

    x_min: float = Field(0.0, description="Spatial domain lower bound")
    x_max: float = Field(1.0, description="Spatial domain upper bound")
    t_min: float = Field(0.0, description="Temporal domain lower bound")
    t_max: float = Field(1.0, description="Temporal domain upper bound")
    wave_speed: float = Field(1.0, gt=0, description="Wave propagation speed c")

    @field_validator("wave_speed")
    @classmethod
    def validate_wave_speed(cls, v: float) -> float:
        """Validate that wave_speed is positive."""
        if v <= 0:
            raise ValueError("wave_speed must be greater than 0")
        return v

    @field_validator("t_max")
    @classmethod
    def validate_t_max(cls, v: float, info) -> float:
        """Validate that t_max > t_min."""
        if "t_min" in info.data and v <= info.data["t_min"]:
            raise ValueError("t_max must be greater than t_min")
        return v

    @field_validator("x_max")
    @classmethod
    def validate_x_max(cls, v: float, info) -> float:
        """Validate that x_max > x_min."""
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v


class BoundaryConditionConfig(BaseModel):
    """Configuration for boundary conditions."""

    type: Literal["dirichlet", "neumann", "periodic"]
    left_value: float | None = None  # For Dirichlet
    right_value: float | None = None


class NetworkConfig(BaseModel):
    """Configuration for neural network architecture."""

    layer_sizes: list[int] = Field(
        [2, 50, 50, 50, 1],
        description="Neural network architecture"
    )
    activation: Literal["tanh", "relu", "sigmoid", "gelu"] = "tanh"

    # Fourier Feature Network settings
    use_fourier_features: bool = Field(
        False,
        description="Use Fourier Feature Embedding for improved high-frequency learning"
    )
    num_fourier_features: int = Field(
        256,
        gt=0,
        description="Number of Fourier features (output dim = 2 * num_fourier_features)"
    )
    fourier_scale: float = Field(
        10.0,
        gt=0,
        description="Scale parameter for Fourier feature frequency distribution"
    )

    @field_validator("layer_sizes")
    @classmethod
    def validate_layer_sizes(cls, v: list[int]) -> list[int]:
        """Validate that layer_sizes is non-empty."""
        if len(v) == 0:
            raise ValueError("layer_sizes must be a non-empty list")
        return v


class TrainingConfig(BaseModel):
    """Configuration for training hyperparameters."""

    epochs: int = Field(10000, gt=0)
    learning_rate: float = Field(1e-3, gt=0)
    optimizer: Literal["adam", "lbfgs"] = "adam"
    loss_weights: dict[str, float] = Field(
        {"data": 1.0, "pde": 1.0, "bc": 1.0, "ic": 1.0}
    )
    amp_enabled: bool = True
    checkpoint_interval: int = 100

    # Causal Training settings
    use_causal_training: bool = Field(
        False,
        description="Use causal (time-dependent) weighting for PDE loss"
    )
    causal_beta: float = Field(
        1.0,
        ge=0,
        description="Causal decay parameter (0=no causal, higher=stronger early-time emphasis)"
    )

    def get_bc_weight(self) -> float:
        """Get boundary condition loss weight."""
        return self.loss_weights.get("bc", 1.0)

    def get_ic_displacement_weight(self) -> float:
        """Get initial condition displacement loss weight."""
        return self.loss_weights.get("ic_displacement", self.loss_weights.get("ic", 1.0))

    def get_ic_velocity_weight(self) -> float:
        """Get initial condition velocity loss weight."""
        return self.loss_weights.get("ic_velocity", self.loss_weights.get("ic", 1.0))

    def get_pde_weight(self) -> float:
        """Get PDE residual loss weight."""
        return self.loss_weights.get("pde", 1.0)


class AnalyticalSolutionConfig(BaseModel):
    """Configuration for analytical solution type and parameters."""

    solution_type: Literal["standing_wave", "standing_wave_neumann", "traveling_wave", "none"]
    mode: int = Field(1, ge=0, description="Mode number (n) for wave solution")
    initial_amplitude: float = Field(1.0, description="Amplitude of initial condition")
    enable_validation: bool = Field(True, description="Enable validation against analytical solution")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    experiment_name: str
    seed: int = 42
    domain: DomainConfig
    boundary_conditions: BoundaryConditionConfig
    network: NetworkConfig
    training: TrainingConfig
    analytical_solution: AnalyticalSolutionConfig | None = None


class ConfigLoaderService:
    """Service for loading and saving configuration files."""

    @staticmethod
    def load_config(file_path: str) -> ExperimentConfig:
        """Load and validate YAML configuration file.

        Args:
            file_path: Path to YAML config file

        Returns:
            Validated ExperimentConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config schema invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {file_path}"
            )

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return ExperimentConfig(**config_dict)

    @staticmethod
    def save_config(config: ExperimentConfig, output_path: str) -> None:
        """Save config to YAML file (for reproducibility).

        Args:
            config: ExperimentConfig object to save
            output_path: Path where YAML file will be written
        """
        path = Path(output_path)

        # Convert Pydantic model to dict
        config_dict = config.model_dump()

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
