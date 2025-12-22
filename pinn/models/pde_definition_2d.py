"""PDE definition for 2D elastic wave equations.

This module provides the PDE residual computation for the dimensionless 2D elastic
wave equations using DeepXDE's automatic differentiation capabilities.

The dimensionless formulation ensures PDE coefficients are O(1), addressing
loss scaling problems observed in Phase 1.
"""

from collections.abc import Callable

import deepxde as dde
import numpy as np
import torch


class PDEDefinition2DService:
    """Service for defining 2D elastic wave PDE residuals."""

    @staticmethod
    def create_pde_function(
        elastic_lambda: float,
        elastic_mu: float,
        density: float
    ) -> Callable:
        """Create 2D elastic wave PDE function for DeepXDE (dimensionless form).

        The dimensionless 2D elastic wave equations are:

        Longitudinal wave (P-wave):
            ∂²Ũx/∂t̃² - (∂²Ũx/∂x̃² + ∂²Ũx/∂ỹ²) = 0    (coefficient = 1)

        Transverse wave (S-wave):
            ∂²Ũy/∂t̃² - (c_t/c_l)² (∂²Ũy/∂x̃² + ∂²Ũy/∂ỹ²) = 0    (coefficient ≈ 0.24 for Al)

        Stress residuals:
            Simplified to zero (rely on FDTD data supervision)

        Args:
            elastic_lambda: Lamé's first parameter (Pa)
            elastic_mu: Shear modulus (Pa)
            density: Material density (kg/m³)

        Returns:
            PDE function with signature (x, y) -> residual where:
                x: (N, 5) dimensionless input [x̃, ỹ, t̃, pitch_norm, depth_norm]
                y: (N, 4) dimensionless output [T̃1, T̃3, Ũx, Ũy]
                residual: (N, 4) PDE residual for each output (O(1) scale)

        Preconditions:
            - elastic_lambda, elastic_mu, density > 0

        Postconditions:
            - Returns callable compatible with dde.data.PDE
            - Residual shape matches output shape (N, 4)
            - All PDE coefficients O(1) (addresses loss scaling problem)

        Invariants:
            - PDE residual = 0 at true solution (physics constraint)

        Example:
            >>> pde_fn = PDEDefinition2DService.create_pde_function(
            ...     elastic_lambda=58e9,
            ...     elastic_mu=26e9,
            ...     density=2700.0
            ... )
            >>> # Use with DeepXDE's TimePDE data object
            >>> data = dde.data.TimePDE(geomtime, pde_fn, [], num_domain=10000)
        """
        # Compute wave speeds
        c_l = np.sqrt((elastic_lambda + 2*elastic_mu) / density)  # Longitudinal wave speed
        c_t = np.sqrt(elastic_mu / density)  # Transverse wave speed

        # Dimensionless wave speed ratio
        c_ratio = c_t / c_l  # ≈ 0.49 for Aluminum 6061

        def pde(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Compute PDE residual for 2D elastic wave equations (dimensionless).

            x: (N, 5) [x̃, ỹ, t̃, pitch_norm, depth_norm] (all dimensionless)
            y: (N, 4) [T̃1, T̃3, Ũx, Ũy] (all dimensionless)
            Returns: (N, 4) [residual_T̃1, residual_T̃3, residual_Ũx, residual_Ũy]

            Note: Input already normalized by DimensionlessScalerService
            """
            # Extract outputs (already dimensionless)
            T1_tilde = y[:, 0:1]  # (N, 1)
            T3_tilde = y[:, 1:2]  # (N, 1)
            Ux_tilde = y[:, 2:3]  # (N, 1)
            Uy_tilde = y[:, 3:4]  # (N, 1)

            # Compute spatial second derivatives (Hessian) w.r.t. dimensionless coords
            # component=2 refers to Ux (3rd output), component=3 refers to Uy (4th output)
            # i, j = 0, 1, 2 correspond to x̃, ỹ, t̃ (first 3 input dimensions)
            Ux_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)  # ∂²Ũx/∂x̃²
            Ux_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)  # ∂²Ũx/∂ỹ²
            Uy_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)  # ∂²Ũy/∂x̃²
            Uy_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)  # ∂²Ũy/∂ỹ²

            # Compute temporal second derivatives w.r.t. dimensionless time
            Ux_tt = dde.grad.hessian(y, x, component=2, i=2, j=2)  # ∂²Ũx/∂t̃²
            Uy_tt = dde.grad.hessian(y, x, component=3, i=2, j=2)  # ∂²Ũy/∂t̃²

            # Dimensionless PDE residuals (coefficients are O(1))
            # Longitudinal wave: ∂²Ũx/∂t̃² - (∂²Ũx/∂x̃² + ∂²Ũx/∂ỹ²) = 0
            # (coefficient = 1 due to T_ref = L_ref/c_l by design)
            residual_Ux = Ux_tt - (Ux_xx + Ux_yy)

            # Transverse wave: ∂²Ũy/∂t̃² - (c_t/c_l)² (∂²Ũy/∂x̃² + ∂²Ũy/∂ỹ²) = 0
            # (coefficient ≈ 0.24 for Aluminum, still O(1))
            residual_Uy = Uy_tt - (c_ratio**2) * (Uy_xx + Uy_yy)

            # Stress residuals (simplified: assume FDTD data provides stress supervision)
            residual_T1 = torch.zeros_like(T1_tilde)
            residual_T3 = torch.zeros_like(T3_tilde)

            # Concatenate all residuals (N, 4)
            return torch.cat([residual_T1, residual_T3, residual_Ux, residual_Uy], dim=1)

        return pde
