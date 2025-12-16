"""Tests for configuration loading with Pydantic validation.

This module tests the ConfigLoaderService for loading and validating
YAML configuration files using Pydantic models.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pinn.utils.config_loader import (
    BoundaryConditionConfig,
    ConfigLoaderService,
    DomainConfig,
    ExperimentConfig,
    NetworkConfig,
    TrainingConfig,
)


class TestDomainConfig:
    """Test suite for DomainConfig Pydantic model."""

    def test_valid_domain_config(self) -> None:
        """Test that valid domain config is accepted."""
        config = DomainConfig(
            x_min=0.0,
            x_max=1.0,
            t_min=0.0,
            t_max=1.0,
            wave_speed=1.0
        )
        assert config.x_min == 0.0
        assert config.x_max == 1.0
        assert config.wave_speed == 1.0

    def test_domain_config_defaults(self) -> None:
        """Test that domain config has sensible defaults."""
        config = DomainConfig()
        assert config.x_min == 0.0
        assert config.x_max == 1.0
        assert config.t_min == 0.0
        assert config.t_max == 1.0
        assert config.wave_speed == 1.0

    def test_wave_speed_must_be_positive(self) -> None:
        """Test that wave_speed > 0 validation works."""
        with pytest.raises(ValidationError) as exc_info:
            DomainConfig(wave_speed=0.0)
        assert "wave_speed" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            DomainConfig(wave_speed=-1.0)
        assert "wave_speed" in str(exc_info.value)

    def test_custom_validation_t_max_greater_than_t_min(self) -> None:
        """Test that t_max > t_min validation works."""
        with pytest.raises(ValidationError) as exc_info:
            DomainConfig(t_min=1.0, t_max=0.5)
        assert "t_max must be greater than t_min" in str(exc_info.value)

    def test_custom_validation_x_max_greater_than_x_min(self) -> None:
        """Test that x_max > x_min validation works."""
        with pytest.raises(ValidationError) as exc_info:
            DomainConfig(x_min=1.0, x_max=0.5)
        assert "x_max must be greater than x_min" in str(exc_info.value)


class TestBoundaryConditionConfig:
    """Test suite for BoundaryConditionConfig."""

    def test_valid_dirichlet_bc(self) -> None:
        """Test valid Dirichlet boundary condition config."""
        config = BoundaryConditionConfig(
            type="dirichlet",
            left_value=0.0,
            right_value=0.0
        )
        assert config.type == "dirichlet"
        assert config.left_value == 0.0
        assert config.right_value == 0.0

    def test_valid_neumann_bc(self) -> None:
        """Test valid Neumann boundary condition config."""
        config = BoundaryConditionConfig(type="neumann")
        assert config.type == "neumann"
        assert config.left_value is None
        assert config.right_value is None

    def test_valid_periodic_bc(self) -> None:
        """Test valid periodic boundary condition config."""
        config = BoundaryConditionConfig(type="periodic")
        assert config.type == "periodic"

    def test_invalid_bc_type_rejected(self) -> None:
        """Test that invalid BC types are rejected."""
        with pytest.raises(ValidationError):
            BoundaryConditionConfig(type="invalid_type")


class TestNetworkConfig:
    """Test suite for NetworkConfig."""

    def test_valid_network_config(self) -> None:
        """Test valid network configuration."""
        config = NetworkConfig(
            layer_sizes=[2, 50, 50, 50, 1],
            activation="tanh"
        )
        assert config.layer_sizes == [2, 50, 50, 50, 1]
        assert config.activation == "tanh"

    def test_network_config_defaults(self) -> None:
        """Test network config defaults."""
        config = NetworkConfig()
        assert config.layer_sizes == [2, 50, 50, 50, 1]
        assert config.activation == "tanh"

    def test_layer_sizes_must_be_non_empty(self) -> None:
        """Test that layer_sizes cannot be empty list."""
        with pytest.raises(ValidationError) as exc_info:
            NetworkConfig(layer_sizes=[])
        assert "layer_sizes" in str(exc_info.value)

    def test_invalid_activation_rejected(self) -> None:
        """Test that invalid activation functions are rejected."""
        with pytest.raises(ValidationError):
            NetworkConfig(activation="invalid_activation")


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_valid_training_config(self) -> None:
        """Test valid training configuration."""
        config = TrainingConfig(
            epochs=10000,
            learning_rate=1e-3,
            optimizer="adam",
            loss_weights={"data": 1.0, "pde": 1.0, "bc": 1.0},
            amp_enabled=True,
            checkpoint_interval=100
        )
        assert config.epochs == 10000
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adam"
        assert config.loss_weights == {"data": 1.0, "pde": 1.0, "bc": 1.0}
        assert config.amp_enabled is True
        assert config.checkpoint_interval == 100

    def test_training_config_defaults(self) -> None:
        """Test training config defaults."""
        config = TrainingConfig()
        assert config.epochs == 10000
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adam"
        assert config.amp_enabled is True
        assert config.checkpoint_interval == 100

    def test_epochs_must_be_positive(self) -> None:
        """Test that epochs > 0 validation works."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=-100)

    def test_learning_rate_must_be_positive(self) -> None:
        """Test that learning_rate > 0 validation works."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.001)

    def test_invalid_optimizer_rejected(self) -> None:
        """Test that invalid optimizers are rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="sgd")


class TestExperimentConfig:
    """Test suite for ExperimentConfig."""

    def test_valid_experiment_config(self) -> None:
        """Test valid complete experiment configuration."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig()
        )
        assert config.experiment_name == "test_experiment"
        assert config.seed == 42
        assert config.domain is not None
        assert config.boundary_conditions is not None
        assert config.network is not None
        assert config.training is not None

    def test_experiment_config_default_seed(self) -> None:
        """Test that default seed is 42."""
        config = ExperimentConfig(
            experiment_name="test",
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig()
        )
        assert config.seed == 42


class TestConfigLoaderService:
    """Test suite for ConfigLoaderService."""

    @pytest.fixture
    def valid_config_dict(self) -> dict:
        """Return a valid configuration dictionary."""
        return {
            "experiment_name": "standing_wave_test",
            "seed": 42,
            "domain": {
                "x_min": 0.0,
                "x_max": 1.0,
                "t_min": 0.0,
                "t_max": 1.0,
                "wave_speed": 1.0
            },
            "boundary_conditions": {
                "type": "dirichlet",
                "left_value": 0.0,
                "right_value": 0.0
            },
            "network": {
                "layer_sizes": [2, 50, 50, 50, 1],
                "activation": "tanh"
            },
            "training": {
                "epochs": 10000,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss_weights": {
                    "data": 1.0,
                    "pde": 1.0,
                    "bc": 1.0
                },
                "amp_enabled": True,
                "checkpoint_interval": 100
            }
        }

    @pytest.fixture
    def valid_yaml_file(self, valid_config_dict: dict, tmp_path: Path) -> Path:
        """Create a temporary YAML file with valid configuration."""
        import yaml
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config_dict, f)
        return config_file

    def test_load_config_from_valid_yaml(self, valid_yaml_file: Path) -> None:
        """Test loading a valid YAML configuration file."""
        config = ConfigLoaderService.load_config(str(valid_yaml_file))
        assert isinstance(config, ExperimentConfig)
        assert config.experiment_name == "standing_wave_test"
        assert config.seed == 42
        assert config.domain.wave_speed == 1.0
        assert config.boundary_conditions.type == "dirichlet"

    def test_load_config_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoaderService.load_config("/nonexistent/path/config.yaml")
        assert "not found" in str(exc_info.value).lower()

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises appropriate error."""
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content: [[[")

        with pytest.raises(Exception):  # YAML parsing error
            ConfigLoaderService.load_config(str(invalid_file))

    def test_load_config_invalid_schema(self, tmp_path: Path) -> None:
        """Test that schema validation catches invalid configs."""
        import yaml
        invalid_config = {
            "experiment_name": "test",
            "domain": {
                "wave_speed": -1.0  # Invalid: must be positive
            },
            "boundary_conditions": {"type": "dirichlet"},
            "network": {},
            "training": {}
        }
        config_file = tmp_path / "invalid_schema.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValidationError):
            ConfigLoaderService.load_config(str(config_file))

    def test_save_config_creates_yaml_file(
        self,
        valid_config_dict: dict,
        tmp_path: Path
    ) -> None:
        """Test that save_config writes valid YAML file."""
        config = ExperimentConfig(**valid_config_dict)
        output_path = tmp_path / "saved_config.yaml"

        ConfigLoaderService.save_config(config, str(output_path))

        assert output_path.exists()
        assert output_path.is_file()

    def test_save_and_load_roundtrip(
        self,
        valid_config_dict: dict,
        tmp_path: Path
    ) -> None:
        """Test that save/load roundtrip produces identical config."""
        original_config = ExperimentConfig(**valid_config_dict)
        output_path = tmp_path / "roundtrip.yaml"

        ConfigLoaderService.save_config(original_config, str(output_path))
        loaded_config = ConfigLoaderService.load_config(str(output_path))

        assert loaded_config.experiment_name == original_config.experiment_name
        assert loaded_config.seed == original_config.seed
        assert loaded_config.domain.wave_speed == original_config.domain.wave_speed
        assert loaded_config.training.epochs == original_config.training.epochs
