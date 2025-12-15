"""PINN model builder for 1D wave equation.

This module provides the PINNModelBuilderService for constructing complete
DeepXDE PINN models by combining geometry, PDE, boundary conditions, and
neural network architecture.
"""

from typing import Callable
import deepxde as dde

from pinn.utils.config_loader import ExperimentConfig, DomainConfig, NetworkConfig
from pinn.models.pde_definition import PDEDefinitionService
from pinn.models.boundary_conditions import BoundaryConditionsService


class PINNModelBuilderService:
    """Service for building complete PINN models."""

    def _create_geometry(self, domain: DomainConfig) -> dde.geometry.GeometryXTime:
        """Create spatiotemporal domain [x_min, x_max] × [t_min, t_max].

        Args:
            domain: Domain configuration with spatial and temporal bounds

        Returns:
            DeepXDE GeometryXTime object
        """
        # Create spatial domain (1D interval)
        geom = dde.geometry.Interval(domain.x_min, domain.x_max)

        # Create temporal domain
        timedomain = dde.geometry.TimeDomain(domain.t_min, domain.t_max)

        # Combine into spatiotemporal geometry
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        return geomtime

    def _create_network(self, network_config: NetworkConfig) -> dde.nn.FNN:
        """Create feedforward neural network with specified architecture.

        Args:
            network_config: Network configuration with layer sizes and activation

        Returns:
            DeepXDE FNN (feedforward neural network)
        """
        return dde.nn.FNN(
            network_config.layer_sizes,
            network_config.activation,
            "Glorot normal"  # Weight initialization
        )

    def build_model(
        self,
        config: ExperimentConfig,
        initial_condition_func: Callable,
        compile_model: bool = True
    ) -> dde.Model:
        """Construct DeepXDE PINN model from configuration.

        Args:
            config: Validated experiment configuration
            initial_condition_func: Function defining u(x, 0) = f(x)
            compile_model: If True, compile model with optimizer (default: True)

        Returns:
            Compiled dde.Model ready for training

        Example:
            >>> config = ExperimentConfig(...)
            >>> def ic(x): return np.sin(np.pi * x[:, 0:1])
            >>> model = builder.build_model(config, ic)
            >>> model.train(epochs=10000)
        """
        # Create spatiotemporal geometry
        geomtime = self._create_geometry(config.domain)

        # Create PDE function with configured wave speed
        pde_func = PDEDefinitionService.create_pde_function(
            config.domain.wave_speed
        )

        # Create boundary conditions based on configuration
        if config.boundary_conditions.type == "dirichlet":
            bc = BoundaryConditionsService.create_zero_dirichlet_bc(geomtime)
        elif config.boundary_conditions.type == "neumann":
            # For Neumann BC with zero normal derivative
            import numpy as np

            def zero_neumann(x):
                return np.zeros((len(x), 1))

            def on_boundary(x, on_boundary):
                return on_boundary

            bc = BoundaryConditionsService.create_neumann_bc(
                geomtime, zero_neumann, on_boundary
            )
        else:  # periodic
            # Periodic BC not implemented in this phase
            raise NotImplementedError("Periodic BC not yet implemented")

        # Create initial conditions
        ic_displacement = BoundaryConditionsService.create_initial_condition(
            geomtime, initial_condition_func
        )
        ic_velocity = BoundaryConditionsService.create_zero_initial_velocity(
            geomtime
        )

        # Combine all constraints
        constraints = [bc, ic_displacement, ic_velocity]

        # Create TimePDE data object
        data = dde.data.TimePDE(
            geomtime,
            pde_func,
            constraints,
            num_domain=2540,     # Number of collocation points in domain
            num_boundary=80,     # Number of points on boundary
            num_initial=160      # Number of points at t=0
        )

        # Create neural network
        net = self._create_network(config.network)

        # Create and compile model
        model = dde.Model(data, net)

        if compile_model:
            # Compile with optimizer and loss weights
            model.compile(
                config.training.optimizer,
                lr=config.training.learning_rate,
                loss_weights=[
                    config.training.loss_weights.get("bc", 1.0),
                    config.training.loss_weights.get("data", 1.0),
                    config.training.loss_weights.get("pde", 1.0)
                ]
            )

        return model
