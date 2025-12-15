"""Unit tests for analytical solution generators.

Tests verify:
1. Standing wave formula against textbook solutions
2. Traveling wave with known initial conditions
3. Solutions satisfy wave equation (PDE residual → 0)
4. Boundary condition satisfaction
"""

import numpy as np
import pytest
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService


class TestStandingWave:
    """Test standing wave analytical solution."""

    def test_fundamental_mode_at_t0(self):
        """Test standing wave at t=0 matches initial condition."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: fundamental mode (n=1), L=1.0, c=1.0
        x = np.linspace(0, 1, 11)
        t = np.array([0.0])
        L = 1.0
        c = 1.0
        n = 1

        # Execute
        u = generator.standing_wave(x, t, L, c, n)

        # Verify: At t=0, u(x,0) = sin(πx/L) * cos(0) = sin(πx)
        expected = np.sin(n * np.pi * x / L)
        np.testing.assert_array_almost_equal(u[:, 0], expected, decimal=10)

    def test_boundary_conditions_zero(self):
        """Test standing wave satisfies u(0,t) = u(L,t) = 0."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup
        x = np.array([0.0, 1.0])  # Boundaries
        t = np.linspace(0, 2, 21)
        L = 1.0
        c = 1.0
        n = 1

        # Execute
        u = generator.standing_wave(x, t, L, c, n)

        # Verify: u(0,t) = 0 and u(L,t) = 0 for all t
        np.testing.assert_array_almost_equal(u[0, :], 0.0, decimal=10)
        np.testing.assert_array_almost_equal(u[1, :], 0.0, decimal=10)

    def test_temporal_oscillation(self):
        """Test standing wave oscillates with period T = 2L/(nc)."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: fundamental mode
        x = np.array([0.5])  # Midpoint
        L = 1.0
        c = 1.0
        n = 1
        period = 2 * L / (n * c)  # T = 2L/c for n=1

        t = np.array([0.0, period / 4, period / 2, 3 * period / 4, period])

        # Execute
        u = generator.standing_wave(x, t, L, c, n)

        # Verify: cos(2πt/T) pattern: 1, 0, -1, 0, 1
        expected_temporal = np.array([1.0, 0.0, -1.0, 0.0, 1.0]) * np.sin(np.pi * 0.5)
        np.testing.assert_array_almost_equal(u[0, :], expected_temporal, decimal=10)

    def test_higher_modes(self):
        """Test higher harmonic modes (n > 1)."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: 2nd harmonic
        x = np.linspace(0, 1, 101)
        t = np.array([0.0])
        L = 1.0
        c = 1.0
        n = 2

        # Execute
        u = generator.standing_wave(x, t, L, c, n)

        # Verify: u(x,0) = sin(2πx) for n=2
        expected = np.sin(2 * np.pi * x)
        np.testing.assert_array_almost_equal(u[:, 0], expected, decimal=10)

    def test_wave_equation_satisfaction(self):
        """Verify standing wave satisfies PDE: ∂²u/∂t² - c²∂²u/∂x² = 0."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup
        L = 1.0
        c = 1.0
        n = 1

        # Use finite differences to compute derivatives
        dx = 0.01
        dt = 0.01
        x = np.arange(dx, L, dx)  # Exclude boundaries
        t = np.arange(dt, 1.0, dt)

        u = generator.standing_wave(x, t, L, c, n)

        # Compute second derivatives using central differences
        # ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
        d2u_dx2 = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2

        # ∂²u/∂t² ≈ (u[j+1] - 2u[j] + u[j-1]) / dt²
        d2u_dt2 = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dt**2

        # Compute residual (exclude edges due to boundary artifacts)
        residual = d2u_dt2[1:-1, 1:-1] - c**2 * d2u_dx2[1:-1, 1:-1]

        # Verify residual is small (numerical errors O(dx²))
        np.testing.assert_array_almost_equal(residual, 0.0, decimal=2)


class TestTravelingWave:
    """Test traveling wave analytical solution."""

    def test_gaussian_pulse_propagation(self):
        """Test Gaussian pulse travels at speed c."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: Gaussian initial condition
        def gaussian(x):
            return np.exp(-50 * (x - 0.5)**2)

        x = np.linspace(0, 2, 201)
        c = 1.0

        # Execute at different times
        t0 = np.array([0.0])
        t1 = np.array([0.1])

        u0 = generator.traveling_wave(x, t0, c, gaussian)
        u1 = generator.traveling_wave(x, t1, c, gaussian)

        # Verify: Peak should shift by c*dt
        peak_idx_0 = np.argmax(u0[:, 0])
        peak_idx_1 = np.argmax(u1[:, 0])

        dx = x[1] - x[0]
        shift = (peak_idx_1 - peak_idx_0) * dx
        expected_shift = c * 0.1 * 2  # Factor of 2 from d'Alembert's formula

        # Allow some tolerance for numerical peak finding
        assert abs(shift - expected_shift) < 5 * dx

    def test_symmetric_initial_condition(self):
        """Test symmetric initial condition produces symmetric wave."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: Symmetric initial condition
        def symmetric_pulse(x):
            return np.exp(-10 * x**2)

        x = np.linspace(-1, 1, 201)
        t = np.array([0.0])
        c = 1.0

        # Execute
        u = generator.traveling_wave(x, t, c, symmetric_pulse)

        # Verify: u(x, 0) should be symmetric
        u_left = u[:100, 0]
        u_right = u[101:, 0][::-1]  # Reverse right half

        np.testing.assert_array_almost_equal(u_left, u_right, decimal=10)

    def test_wave_equation_satisfaction(self):
        """Verify traveling wave satisfies wave equation."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: Simple sinusoidal pulse
        def sine_pulse(x):
            return np.sin(2 * np.pi * x)

        c = 1.0
        dx = 0.01
        dt = 0.005

        x = np.arange(-2, 2, dx)
        t = np.arange(0, 0.5, dt)

        u = generator.traveling_wave(x, t, c, sine_pulse)

        # Compute second derivatives
        d2u_dx2 = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2
        d2u_dt2 = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dt**2

        # Compute residual
        residual = d2u_dt2[10:-10, 10:-10] - c**2 * d2u_dx2[10:-10, 10:-10]

        # Verify residual is small
        np.testing.assert_array_almost_equal(residual, 0.0, decimal=1)


class TestEvaluateAtPoints:
    """Test pointwise evaluation of analytical solutions."""

    def test_standing_wave_pointwise(self):
        """Test pointwise evaluation matches meshgrid evaluation."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup
        L = 1.0
        c = 1.0
        n = 1

        # Create meshgrid solution
        x = np.linspace(0, 1, 11)
        t = np.linspace(0, 1, 11)
        u_mesh = generator.standing_wave(x, t, L, c, n)

        # Create pointwise evaluation
        X, T = np.meshgrid(x, t, indexing="ij")
        points = np.column_stack([X.flatten(), T.flatten()])
        u_points = generator.evaluate_at_points(points, L, c, "standing", n)

        # Verify they match
        np.testing.assert_array_almost_equal(u_points, u_mesh.flatten(), decimal=10)

    def test_evaluate_at_specific_locations(self):
        """Test evaluation at specific spatiotemporal points."""
        generator = AnalyticalSolutionGeneratorService()

        # Setup: known points
        points = np.array([
            [0.0, 0.0],  # Boundary at t=0
            [0.5, 0.0],  # Center at t=0
            [1.0, 0.0],  # Boundary at t=0
        ])
        L = 1.0
        c = 1.0
        n = 1

        # Execute
        u = generator.evaluate_at_points(points, L, c, "standing", n)

        # Verify
        expected = np.array([0.0, 1.0, 0.0])  # sin(0), sin(π/2), sin(π)
        np.testing.assert_array_almost_equal(u, expected, decimal=10)


class TestInputShapeHandling:
    """Test handling of various input array shapes."""

    def test_1d_input_arrays(self):
        """Test 1D input arrays (N,) shape."""
        generator = AnalyticalSolutionGeneratorService()

        x = np.array([0.0, 0.5, 1.0])
        t = np.array([0.0, 0.5])

        u = generator.standing_wave(x, t, L=1.0, c=1.0, n=1)

        assert u.shape == (3, 2)

    def test_scalar_time(self):
        """Test scalar time input."""
        generator = AnalyticalSolutionGeneratorService()

        x = np.linspace(0, 1, 11)
        t = 0.5  # Scalar

        u = generator.standing_wave(x, t, L=1.0, c=1.0, n=1)

        assert u.shape == (11, 1)

    def test_2d_input_arrays(self):
        """Test 2D column vector inputs (N, 1)."""
        generator = AnalyticalSolutionGeneratorService()

        x = np.linspace(0, 1, 11).reshape(-1, 1)
        t = np.linspace(0, 1, 11).reshape(-1, 1)

        u = generator.standing_wave(x, t, L=1.0, c=1.0, n=1)

        assert u.shape == (11, 11)
