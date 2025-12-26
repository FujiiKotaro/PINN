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

    def _get_attr(self, obj, key, default=None):
        """Helper to get attribute or dict item."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _create_geometry(self, domain) -> dde.geometry.GeometryXTime:
        """Create 2D Rectangle + TimeDomain spatiotemporal geometry.

        Args:
            domain: Domain configuration (DomainConfig object or dict)

        Returns:
            DeepXDE GeometryXTime object for 3D (x, y, t) domain
        """
        # Extract bounds handling both object and dict
        x_min = self._get_attr(domain, "x_min")
        x_max = self._get_attr(domain, "x_max")
        y_min = self._get_attr(domain, "y_min")
        y_max = self._get_attr(domain, "y_max")
        t_min = self._get_attr(domain, "t_min")
        t_max = self._get_attr(domain, "t_max")

        # Create 2D spatial domain (Rectangle)
        spatial_geom = dde.geometry.Rectangle(
            [x_min, y_min],
            [x_max, y_max]
        )

        # Create temporal domain
        timedomain = dde.geometry.TimeDomain(t_min, t_max)

        # Combine into spatiotemporal geometry
        geomtime = dde.geometry.GeometryXTime(spatial_geom, timedomain)

        return geomtime

    def build_model(
        self,
        config,
        compile_model: bool = True
    ) -> dde.Model:
        """Construct 2D parametric PINN model from configuration.

        Args:
            config: Validated experiment configuration (ExperimentConfig object or dict)
            compile_model: If True, compile model with optimizer (default: True)

        Returns:
            Compiled dde.Model ready for training
        """
        # Extract components handling both object and dict
        domain = self._get_attr(config, "domain")
        network = self._get_attr(config, "network")
        training = self._get_attr(config, "training")

        # Create spatiotemporal geometry
        geomtime = self._create_geometry(domain)

        # Create 2D elastic wave PDE function (dimensionless form)
        # Note: DomainConfig in config_loader might not have elastic constants,
        # but the notebook config dict does.
        pde_service = PDEDefinition2DService(
            elastic_lambda=self._get_attr(domain, "elastic_lambda"),
            elastic_mu=self._get_attr(domain, "elastic_mu"),
            density=self._get_attr(domain, "density")
        )
        pde_func = pde_service.create_pde_function()

        # No explicit boundary/initial conditions (rely on FDTD data supervision)
        constraints = []

        # Get number of domain points
        num_domain = self._get_attr(training, "num_domain", 10000)
        if isinstance(training, dict):
            num_domain = training.get("num_domain", 10000)
        
        # Create TimePDE data object
        data = dde.data.TimePDE(
            geomtime,
            pde_func,
            constraints,
            num_domain=num_domain,
            num_boundary=0,
            num_initial=0
        )

        # Create neural network
        layer_sizes = self._get_attr(network, "layer_sizes")
        activation = self._get_attr(network, "activation")
        
        net = self._create_network(layer_sizes, activation)

        # Create DeepXDE Model
        model = dde.Model(data, net)

        if compile_model:
            # Compile with optimizer and loss weights
            optimizer = self._get_attr(training, "optimizer", "adam")
            lr = self._get_attr(training, "learning_rate")
            loss_weights_config = self._get_attr(training, "loss_weights")
            
            pde_weight = 1.0
            if isinstance(loss_weights_config, dict):
                pde_weight = loss_weights_config.get("pde", 1.0)
            else:
                 # Assume it's an object/dict accessible
                 pde_weight = getattr(loss_weights_config, "pde", 1.0) # Or handle dict access if it's a dict

            # Ensure pde_weight is extracted correctly if loss_weights is just a dict
            if isinstance(loss_weights_config, dict):
                 pde_weight = loss_weights_config.get("pde", 1.0)
            
            model.compile(
                optimizer,
                lr=lr,
                loss_weights=[pde_weight]
            )

        return model
