"""PINN model builder for 2D elastic wave equations.

This module provides the PINNModelBuilder2DService for constructing 2D PINN models
with 5D input (x, y, t, pitch, depth) and 4D output (T1, T3, Ux, Uy).

Extends Phase 1's PINNModelBuilderService to support 2D geometry and parametric learning.
"""

import deepxde as dde

from pinn.models.pde_definition_2d import PDEDefinition2DService
from pinn.utils.config_loader import DomainConfig, ExperimentConfig


class PINNModelBuilder2DService:
    """Service for building 2D parametric PINN models.

    Constructs PINN models with:
    - 2D Rectangle + TimeDomain geometry
    - 5D input: (x, y, t, pitch_norm, depth_norm)
    - 4D output: (T1, T3, Ux, Uy)
    - 2D elastic wave PDE constraints
    """

    def _create_network(
        self,
        layer_sizes: list[int],
        activation: str
    ) -> dde.nn.FNN:
        """Create feedforward neural network for 2D parametric PINN.

        Args:
            layer_sizes: Network architecture (e.g., [5, 64, 64, 64, 4])
                - First element: input dimension (5 for x, y, t, pitch, depth)
                - Last element: output dimension (4 for T1, T3, Ux, Uy)
            activation: Activation function name ("tanh", "relu", etc.)

        Returns:
            DeepXDE FNN (feedforward neural network)

        Preconditions:
            - layer_sizes[0] == 5 (5D input)
            - layer_sizes[-1] == 4 (4D output)
            - len(layer_sizes) >= 2

        Postconditions:
            - Returns FNN with Glorot normal initialization
            - Network accepts (N, 5) input and produces (N, 4) output

        Example:
            >>> builder = PINNModelBuilder2DService()
            >>> net = builder._create_network([5, 64, 64, 64, 4], "tanh")
            >>> x = torch.randn(10, 5)
            >>> y = net(x)  # Shape: (10, 4)
        """
        # Validate layer_sizes
        assert len(layer_sizes) >= 2, "layer_sizes must have at least 2 elements"
        assert layer_sizes[0] == 5, \
            f"Input dimension must be 5, got {layer_sizes[0]}"
        assert layer_sizes[-1] == 4, \
            f"Output dimension must be 4, got {layer_sizes[-1]}"

        # Create DeepXDE FNN with Glorot normal initialization
        return dde.nn.FNN(
            layer_sizes,
            activation,
            "Glorot normal"  # Weight initialization
        )

    def _create_geometry(self, domain: DomainConfig) -> dde.geometry.GeometryXTime:
        """Create 2D Rectangle + TimeDomain spatiotemporal geometry.

        Args:
            domain: Domain configuration with spatial (x, y) and temporal bounds

        Returns:
            DeepXDE GeometryXTime object for 3D (x, y, t) domain

        Preconditions:
            - domain.x_min < domain.x_max
            - domain.y_min < domain.y_max
            - domain.t_min < domain.t_max

        Postconditions:
            - Returns GeometryXTime with dim=3 (x, y, t)
            - Spatial domain is 2D Rectangle
            - Temporal domain is TimeDomain

        Example:
            >>> domain = DomainConfig(x_min=0, x_max=0.04, y_min=0, y_max=0.02,
            ...                       t_min=3.5e-6, t_max=6.5e-6)
            >>> geomtime = builder._create_geometry(domain)
            >>> geomtime.dim  # Returns 3 (x, y, t)
        """
        # Create 2D spatial domain (Rectangle)
        spatial_geom = dde.geometry.Rectangle(
            [domain.x_min, domain.y_min],
            [domain.x_max, domain.y_max]
        )

        # Create temporal domain
        timedomain = dde.geometry.TimeDomain(domain.t_min, domain.t_max)

        # Combine into spatiotemporal geometry
        geomtime = dde.geometry.GeometryXTime(spatial_geom, timedomain)

        return geomtime

    def build_model(
        self,
        config: ExperimentConfig,
        compile_model: bool = True
    ) -> dde.Model:
        """Construct 2D parametric PINN model from configuration.

        Args:
            config: Validated experiment configuration with domain, network, elastic constants
            compile_model: If True, compile model with optimizer (default: True)

        Returns:
            Compiled dde.Model ready for training

        Preconditions:
            - config.domain defines valid 2D Rectangle bounds
            - config.domain.elastic_lambda, elastic_mu, density > 0
            - config.network.layer_sizes = [5, ..., 4]

        Postconditions:
            - Returns compiled DeepXDE Model with 2D PDE constraints
            - Model ready for training via TrainingPipelineService

        Invariants:
            - Geometry spatial bounds match config.domain
            - Network input/output dimensions match 5D/4D specification

        Example:
            >>> config = ExperimentConfig(...)
            >>> builder = PINNModelBuilder2DService()
            >>> model = builder.build_model(config)
            >>> # Train with FDTD data
        """
        # Create spatiotemporal geometry (2D Rectangle + Time)
        geomtime = self._create_geometry(config.domain)

        # Create 2D elastic wave PDE function (dimensionless form)
        pde_func = PDEDefinition2DService.create_pde_function(
            elastic_lambda=config.domain.elastic_lambda,
            elastic_mu=config.domain.elastic_mu,
            density=config.domain.density
        )

        # No explicit boundary/initial conditions (rely on FDTD data supervision)
        # TimePDE with empty constraints list
        constraints = []

        # Create TimePDE data object with collocation points
        # num_domain: PDE collocation points (default 10000 per design)
        # num_boundary/num_initial: 0 (no explicit BC/IC, rely on data)
        data = dde.data.TimePDE(
            geomtime,
            pde_func,
            constraints,
            num_domain=config.training.get("num_domain", 10000),
            num_boundary=0,  # No explicit BC
            num_initial=0    # No explicit IC
        )

        # Create neural network (5D input → 4D output)
        net = self._create_network(
            config.network.layer_sizes,
            config.network.activation
        )

        # Create DeepXDE Model
        model = dde.Model(data, net)

        if compile_model:
            # Compile with optimizer and loss weights
            # Loss weights: [pde]
            pde_weight = config.training.loss_weights.get("pde", 1.0)

            model.compile(
                config.training.optimizer,
                lr=config.training.learning_rate,
                loss_weights=[pde_weight]  # Only PDE loss (no BC/IC)
            )

        return model
