"""PDE definition for 1D wave equation.

This module provides the PDE residual computation for the 1D wave equation
using DeepXDE's automatic differentiation capabilities.
"""

from collections.abc import Callable

import deepxde as dde
import torch


class PDEDefinitionService:
    """Service for defining PDE residuals using automatic differentiation."""

    @staticmethod
    def wave_equation_residual(
        x: torch.Tensor,
        u: torch.Tensor,
        c: float
    ) -> torch.Tensor:
        """Compute PDE residual for 1D wave equation: ∂²u/∂t² - c²∂²u/∂x².

        The 1D wave equation describes wave propagation in a homogeneous medium:
            ∂²u/∂t² = c² ∂²u/∂x²

        Rearranged to residual form (should equal zero for exact solutions):
            residual = ∂²u/∂t² - c² ∂²u/∂x²

        Args:
            x: Input coordinates (batch_size, 2) where x[:, 0] = spatial coord,
               x[:, 1] = temporal coord
            u: Network predictions (batch_size, 1)
            c: Wave speed parameter (must be positive)

        Returns:
            PDE residual (batch_size, 1). For exact solutions satisfying the
            wave equation, this residual should be close to zero.

        Note:
            This function uses DeepXDE's grad module for automatic differentiation.
            The input tensor x must have requires_grad=True for gradient computation.
        """
        # Compute first-order partial derivatives
        # du_x: ∂u/∂x (spatial derivative)
        # du_t: ∂u/∂t (temporal derivative)
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_t = dde.grad.jacobian(u, x, i=0, j=1)

        # Compute second-order partial derivatives
        # du_xx: ∂²u/∂x² (spatial second derivative)
        # du_tt: ∂²u/∂t² (temporal second derivative)
        du_xx = dde.grad.jacobian(du_x, x, i=0, j=0)
        du_tt = dde.grad.jacobian(du_t, x, i=0, j=1)

        # Compute wave equation residual
        # residual = ∂²u/∂t² - c² ∂²u/∂x²
        residual = du_tt - (c ** 2) * du_xx

        return residual

    @staticmethod
    def create_pde_function(c: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Create a PDE function with fixed wave speed for use with DeepXDE.

        Args:
            c: Wave speed parameter

        Returns:
            Callable PDE function that takes (x, u) and returns residual

        Example:
            >>> pde_fn = PDEDefinitionService.create_pde_function(c=1.0)
            >>> # Use with DeepXDE's TimePDE data object
            >>> data = dde.data.TimePDE(geomtime, pde_fn, [bc, ic], ...)
        """
        def pde(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return PDEDefinitionService.wave_equation_residual(x, u, c)

        return pde
