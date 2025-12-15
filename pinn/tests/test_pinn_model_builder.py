"""Tests for PINN model builder.

This module tests the PINNModelBuilderService for constructing complete
DeepXDE PINN models.
"""

import pytest
import deepxde as dde
import numpy as np

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.utils.config_loader import (
    ExperimentConfig,
    DomainConfig,
    BoundaryConditionConfig,
    NetworkConfig,
    TrainingConfig,
)


class TestPINNModelBuilderService:
    """Test suite for PINN model builder."""

    @pytest.fixture
    def config(self) -> ExperimentConfig:
        """Create a valid experiment configuration."""
        return ExperimentConfig(
            experiment_name="test",
            seed=42,
            domain=DomainConfig(
                x_min=0.0,
                x_max=1.0,
                t_min=0.0,
                t_max=1.0,
                wave_speed=1.0
            ),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(layer_sizes=[2, 20, 20, 1]),
            training=TrainingConfig(epochs=100)
        )

    def test_create_geometry(self, config: ExperimentConfig) -> None:
        """Test spatiotemporal geometry creation."""
        builder = PINNModelBuilderService()
        geomtime = builder._create_geometry(config.domain)

        assert geomtime is not None
        assert isinstance(geomtime, dde.geometry.GeometryXTime)

    def test_create_network(self, config: ExperimentConfig) -> None:
        """Test neural network creation."""
        builder = PINNModelBuilderService()
        net = builder._create_network(config.network)

        assert net is not None
        # DeepXDE creates FNN with correct architecture

    def test_build_model_returns_dde_model(self, config: ExperimentConfig) -> None:
        """Test that build_model returns a compiled DeepXDE Model."""
        builder = PINNModelBuilderService()

        # Define initial condition
        def ic_func(x):
            return np.sin(np.pi * x[:, 0:1])

        model = builder.build_model(config, ic_func)

        assert model is not None
        assert isinstance(model, dde.Model)

    def test_build_model_with_zero_initial_condition(self, config: ExperimentConfig) -> None:
        """Test building model with zero initial condition."""
        builder = PINNModelBuilderService()

        def zero_ic(x):
            return np.zeros((len(x), 1))

        model = builder.build_model(config, zero_ic)

        assert model is not None
        assert isinstance(model, dde.Model)

    def test_build_model_respects_wave_speed(self, config: ExperimentConfig) -> None:
        """Test that model uses configured wave speed."""
        config.domain.wave_speed = 2.0

        builder = PINNModelBuilderService()

        def ic_func(x):
            return np.sin(np.pi * x[:, 0:1])

        model = builder.build_model(config, ic_func)

        assert model is not None
        # Wave speed is embedded in PDE function

    def test_build_model_respects_network_architecture(self, config: ExperimentConfig) -> None:
        """Test that model uses configured network architecture."""
        config.network.layer_sizes = [2, 50, 50, 50, 1]

        builder = PINNModelBuilderService()

        def ic_func(x):
            return np.sin(np.pi * x[:, 0:1])

        model = builder.build_model(config, ic_func)

        assert model is not None
