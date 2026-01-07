"""Causal PDE definition with time-dependent loss weighting.

This module implements causal training for PINNs, where PDE residuals at earlier
time steps are weighted more heavily to enforce temporal causality.

Reference:
    Wang et al. (2022) "When and why PINNs fail to train: A neural tangent
    kernel perspective" (Journal of Computational Physics)
"""

from collections.abc import Callable

import deepxde as dde
import torch


class CausalPDEDefinitionService:
    """Service for defining PDE residuals with causal (time-dependent) weighting."""

    @staticmethod
    def wave_equation_residual_causal(
        x: torch.Tensor,
        u: torch.Tensor,
        c: float,
        beta: float = 1.0
    ) -> torch.Tensor:
        """Compute causal PDE residual for 1D wave equation.

        The residual is weighted by exp(-β * t) to emphasize earlier time steps,
        enforcing temporal causality in the learning process.

        Args:
            x: Input coordinates (batch_size, 2) where x[:, 0] = spatial coord,
               x[:, 1] = temporal coord
            u: Network predictions (batch_size, 1)
            c: Wave speed parameter
            beta: Causal decay parameter (higher = stronger emphasis on early times)

        Returns:
            Weighted PDE residual (batch_size, 1)
        """
        # Compute first-order partial derivatives
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_t = dde.grad.jacobian(u, x, i=0, j=1)

        # Compute second-order partial derivatives
        du_xx = dde.grad.jacobian(du_x, x, i=0, j=0)
        du_tt = dde.grad.jacobian(du_t, x, i=0, j=1)

        # Compute wave equation residual
        residual = du_tt - (c ** 2) * du_xx

        # Extract time coordinate
        t = x[:, 1:2]

        # Compute causal weight: w(t) = exp(-β * t)
        # Early times have higher weight, emphasizing correct wave propagation
        # from initial conditions
        causal_weight = torch.exp(-beta * t)

        # Apply causal weighting to residual
        weighted_residual = causal_weight * residual

        return weighted_residual

    @staticmethod
    def create_causal_pde_function(
        c: float,
        beta: float = 1.0
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create a causal PDE function with fixed wave speed and causal parameter.

        Args:
            c: Wave speed parameter
            beta: Causal decay parameter (default: 1.0)
                  - beta = 0: No causal weighting (standard PDE)
                  - beta > 0: Stronger emphasis on early times
                  - Typical values: 0.5 - 2.0

        Returns:
            Callable PDE function that takes (x, u) and returns weighted residual

        Example:
            >>> pde_fn = CausalPDEDefinitionService.create_causal_pde_function(
            ...     c=1.0, beta=1.0
            ... )
            >>> data = dde.data.TimePDE(geomtime, pde_fn, [bc, ic], ...)
        """
        def pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return CausalPDEDefinitionService.wave_equation_residual_causal(
                x, u, c, beta
            )

        return pde

    @staticmethod
    def wave_equation_residual_adaptive_causal(
        x: torch.Tensor,
        u: torch.Tensor,
        c: float,
        t_max: float,
        beta: float = 2.0
    ) -> torch.Tensor:
        """Compute adaptive causal PDE residual.

        Uses normalized time t_norm = t / t_max for consistent weighting
        across different temporal domains.

        Args:
            x: Input coordinates (batch_size, 2)
            u: Network predictions (batch_size, 1)
            c: Wave speed parameter
            t_max: Maximum time in domain (for normalization)
            beta: Causal decay parameter

        Returns:
            Weighted PDE residual (batch_size, 1)
        """
        # Compute first-order partial derivatives
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_t = dde.grad.jacobian(u, x, i=0, j=1)

        # Compute second-order partial derivatives
        du_xx = dde.grad.jacobian(du_x, x, i=0, j=0)
        du_tt = dde.grad.jacobian(du_t, x, i=0, j=1)

        # Compute wave equation residual
        residual = du_tt - (c ** 2) * du_xx

        # Extract and normalize time coordinate
        t = x[:, 1:2]
        t_norm = t / t_max

        # Compute adaptive causal weight
        # w(t) = exp(-β * t_norm) ensures consistent behavior across domains
        causal_weight = torch.exp(-beta * t_norm)

        # Apply causal weighting
        weighted_residual = causal_weight * residual

        return weighted_residual

    @staticmethod
    def create_adaptive_causal_pde_function(
        c: float,
        t_max: float,
        beta: float = 2.0
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create an adaptive causal PDE function.

        Args:
            c: Wave speed parameter
            t_max: Maximum time in temporal domain
            beta: Causal decay parameter (default: 2.0)

        Returns:
            Callable PDE function with adaptive causal weighting

        Example:
            >>> pde_fn = CausalPDEDefinitionService.create_adaptive_causal_pde_function(
            ...     c=1.5, t_max=2.0, beta=2.0
            ... )
        """
        def pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return CausalPDEDefinitionService.wave_equation_residual_adaptive_causal(
                x, u, c, t_max, beta
            )

        return pde


class CausalWeightScheduler:
    """Schedule causal weight decay parameter during training.

    Allows gradual transition from strong causal weighting (early training)
    to uniform weighting (late training).
    """

    def __init__(
        self,
        beta_initial: float = 2.0,
        beta_final: float = 0.0,
        transition_epochs: int = 5000
    ):
        """Initialize causal weight scheduler.

        Args:
            beta_initial: Initial causal decay parameter (strong causality)
            beta_final: Final causal decay parameter (weak/no causality)
            transition_epochs: Number of epochs for transition
        """
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.transition_epochs = transition_epochs

    def get_beta(self, epoch: int) -> float:
        """Get causal decay parameter for current epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Causal decay parameter β

        Example:
            >>> scheduler = CausalWeightScheduler(beta_initial=2.0, beta_final=0.0)
            >>> beta = scheduler.get_beta(epoch=2500)  # Returns ~1.0
        """
        if epoch >= self.transition_epochs:
            return self.beta_final

        # Linear decay from beta_initial to beta_final
        progress = epoch / self.transition_epochs
        beta = self.beta_initial + (self.beta_final - self.beta_initial) * progress

        return beta
