"""Tests for PDE definition (wave equation residual computation).

This module tests the PDEDefinitionService for computing the wave equation
PDE residual using automatic differentiation.
"""

import numpy as np
import torch

from pinn.models.pde_definition import PDEDefinitionService


class TestPDEDefinitionService:
    """Test suite for wave equation PDE definition."""

    def test_wave_equation_residual_for_exact_solution(self) -> None:
        """Test that residual is near-zero for exact analytical solution.

        Use standing wave: u(x,t) = sin(πx) cos(πct)
        This satisfies wave equation: ∂²u/∂t² - c²∂²u/∂x² = 0
        """
        c = 1.0

        # Create test points
        x_vals = np.linspace(0, 1, 10)
        t_vals = np.linspace(0, 1, 10)
        X, T = np.meshgrid(x_vals, t_vals)
        xt = np.stack([X.flatten(), T.flatten()], axis=1)

        # Convert to tensor with requires_grad
        xt_tensor = torch.tensor(xt, dtype=torch.float32, requires_grad=True)

        # Compute exact solution: sin(πx) cos(πct)
        u_exact = torch.sin(np.pi * xt_tensor[:, 0:1]) * torch.cos(np.pi * c * xt_tensor[:, 1:2])

        # Compute residual
        residual = PDEDefinitionService.wave_equation_residual(xt_tensor, u_exact, c)

        # Residual should be near zero for exact solution
        assert residual.shape == (len(xt), 1)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-4)

    def test_wave_equation_residual_shape(self) -> None:
        """Test that residual has correct output shape."""
        batch_size = 100
        c = 1.0

        # Random input points
        x = torch.rand(batch_size, 2, requires_grad=True)
        # Create u as a function of x to establish computational graph
        u = torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])

        residual = PDEDefinitionService.wave_equation_residual(x, u, c)

        assert residual.shape == (batch_size, 1)

    def test_wave_equation_residual_requires_grad(self) -> None:
        """Test that input tensors require gradients."""
        x = torch.rand(10, 2, requires_grad=True)
        # Create u as a function of x
        u = torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])
        c = 1.0

        residual = PDEDefinitionService.wave_equation_residual(x, u, c)
        assert residual.shape == (10, 1)

    def test_wave_equation_residual_with_different_wave_speeds(self) -> None:
        """Test residual computation with different wave speeds."""
        x = torch.rand(20, 2, requires_grad=True)
        # Create u as a function of x
        u = torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])

        # Test with different wave speeds
        for c in [0.5, 1.0, 2.0, 5.0]:
            residual = PDEDefinitionService.wave_equation_residual(x, u, c)
            assert residual.shape == (20, 1)
            assert not torch.isnan(residual).any()
            assert not torch.isinf(residual).any()

    def test_wave_equation_residual_is_pde_aware(self) -> None:
        """Test that residual changes when solution is not exact."""
        c = 1.0
        # Use different spatial and temporal coordinates to avoid accidental zeros
        x = torch.tensor([[0.3, 0.7]], dtype=torch.float32, requires_grad=True)

        # Exact solution at (0.3, 0.7)
        u_exact = torch.sin(np.pi * x[:, 0:1]) * torch.cos(np.pi * c * x[:, 1:2])

        # Wrong solution (cubic polynomial that doesn't satisfy wave equation)
        # residual = ∂²u/∂t² - c²∂²u/∂x² = 6t - c²·6x = 6(0.7 - 1·0.3) = 2.4 ≠ 0
        u_wrong = x[:, 0:1]**3 + x[:, 1:2]**3

        residual_exact = PDEDefinitionService.wave_equation_residual(x, u_exact, c)
        residual_wrong = PDEDefinitionService.wave_equation_residual(x, u_wrong, c)

        # Exact solution should have near-zero residual
        assert torch.abs(residual_exact).item() < 1e-3

        # Wrong solution should have non-zero residual
        assert torch.abs(residual_wrong).item() > 1e-3

    def test_create_pde_function_returns_callable(self) -> None:
        """Test that create_pde_function returns a callable for DeepXDE."""
        c = 1.0
        pde_fn = PDEDefinitionService.create_pde_function(c)

        assert callable(pde_fn)

        # Test that the function works with DeepXDE-style inputs
        x = torch.rand(10, 2, requires_grad=True)
        # Create u as a function of x
        u = torch.sin(x[:, 0:1]) * torch.cos(x[:, 1:2])

        residual = pde_fn(x, u)
        assert residual.shape == (10, 1)

    def test_pde_function_docstring_present(self) -> None:
        """Test that wave_equation_residual has proper documentation."""
        assert PDEDefinitionService.wave_equation_residual.__doc__ is not None
        docstring = PDEDefinitionService.wave_equation_residual.__doc__
        assert "PDE residual" in docstring or "wave equation" in docstring.lower()
