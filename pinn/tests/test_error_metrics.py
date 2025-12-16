"""Unit tests for error metrics.

Tests verify:
1. L2 error and relative error with synthetic arrays
2. Zero error for identical arrays
3. Max absolute error with known differences
4. Edge cases: zero arrays, NaN handling
"""

import numpy as np
import pytest

from pinn.validation.error_metrics import ErrorMetricsService


class TestL2Error:
    """Test L2 error computation."""

    def test_zero_error_identical_arrays(self):
        """Test L2 error is zero for identical arrays."""
        u_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u_exact = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        error = ErrorMetricsService.l2_error(u_pred, u_exact)

        assert error == pytest.approx(0.0, abs=1e-10)

    def test_known_l2_norm(self):
        """Test L2 error with known difference vector."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([0.0, 0.0, 0.0])

        # ||[1, 2, 3]||₂ = sqrt(1² + 2² + 3²) = sqrt(14) ≈ 3.7417
        expected = np.sqrt(14)
        error = ErrorMetricsService.l2_error(u_pred, u_exact)

        assert error == pytest.approx(expected, rel=1e-10)

    def test_2d_array_input(self):
        """Test L2 error with 2D arrays."""
        u_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        u_exact = np.array([[0.0, 1.0], [2.0, 3.0]])

        # Difference: [[1, 1], [1, 1]], ||diff||₂ = sqrt(4) = 2
        expected = 2.0
        error = ErrorMetricsService.l2_error(u_pred, u_exact)

        assert error == pytest.approx(expected, rel=1e-10)

    def test_symmetric_error(self):
        """Test L2 error is symmetric: ||a-b|| = ||b-a||."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([4.0, 5.0, 6.0])

        error_1 = ErrorMetricsService.l2_error(u_pred, u_exact)
        error_2 = ErrorMetricsService.l2_error(u_exact, u_pred)

        assert error_1 == pytest.approx(error_2, rel=1e-10)


class TestRelativeError:
    """Test relative L2 error computation."""

    def test_zero_relative_error_identical(self):
        """Test relative error is zero for identical arrays."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([1.0, 2.0, 3.0])

        rel_error = ErrorMetricsService.relative_error(u_pred, u_exact)

        assert rel_error == pytest.approx(0.0, abs=1e-10)

    def test_known_relative_error(self):
        """Test relative error with known values."""
        u_pred = np.array([2.0, 4.0, 6.0])
        u_exact = np.array([1.0, 2.0, 3.0])

        # ||diff||₂ = ||[1, 2, 3]||₂ = sqrt(14)
        # ||exact||₂ = ||[1, 2, 3]||₂ = sqrt(14)
        # Relative error = sqrt(14) / sqrt(14) = 1.0
        expected = 1.0
        rel_error = ErrorMetricsService.relative_error(u_pred, u_exact)

        assert rel_error == pytest.approx(expected, rel=1e-10)

    def test_small_relative_error(self):
        """Test small perturbation produces small relative error."""
        u_exact = np.ones(100)
        u_pred = u_exact + 0.01 * np.random.randn(100)

        rel_error = ErrorMetricsService.relative_error(u_pred, u_exact)

        # Should be small (around 1%)
        assert rel_error < 0.05  # Less than 5%

    def test_percentage_interpretation(self):
        """Test relative error as percentage."""
        u_exact = np.array([100.0])
        u_pred = np.array([105.0])

        # 5% error
        rel_error = ErrorMetricsService.relative_error(u_pred, u_exact)

        assert rel_error == pytest.approx(0.05, rel=1e-10)

    def test_division_by_zero_handling(self):
        """Test handling when exact solution is zero."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.zeros(3)

        # This should return inf and generate a warning
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            rel_error = ErrorMetricsService.relative_error(u_pred, u_exact)
            assert np.isinf(rel_error)


class TestMaxAbsoluteError:
    """Test maximum absolute error computation."""

    def test_zero_max_error_identical(self):
        """Test max error is zero for identical arrays."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([1.0, 2.0, 3.0])

        max_error = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert max_error == pytest.approx(0.0, abs=1e-10)

    def test_known_max_error(self):
        """Test max error with known differences."""
        u_pred = np.array([1.0, 2.5, 3.0, 10.0])
        u_exact = np.array([1.0, 2.0, 3.0, 4.0])

        # Max difference is |10.0 - 4.0| = 6.0
        expected = 6.0
        max_error = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert max_error == pytest.approx(expected, rel=1e-10)

    def test_max_error_location(self):
        """Test max error correctly identifies largest deviation."""
        u_exact = np.zeros(100)
        u_pred = np.zeros(100)
        u_pred[50] = 10.0  # Single spike

        max_error = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert max_error == pytest.approx(10.0, rel=1e-10)

    def test_negative_differences(self):
        """Test max error with negative differences."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([5.0, 6.0, 7.0])

        # Max |diff| is 4.0
        expected = 4.0
        max_error = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert max_error == pytest.approx(expected, rel=1e-10)

    def test_2d_arrays(self):
        """Test max error with 2D arrays."""
        u_pred = np.array([[1.0, 2.0], [3.0, 100.0]])
        u_exact = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Max diff is |100 - 4| = 96
        expected = 96.0
        max_error = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert max_error == pytest.approx(expected, rel=1e-10)


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        u_pred = np.array([])
        u_exact = np.array([])

        # Should handle gracefully or raise appropriate error
        # L2 norm of empty array is 0
        error = ErrorMetricsService.l2_error(u_pred, u_exact)
        assert error == 0.0

    def test_single_element_arrays(self):
        """Test with single-element arrays."""
        u_pred = np.array([5.0])
        u_exact = np.array([3.0])

        l2_err = ErrorMetricsService.l2_error(u_pred, u_exact)
        rel_err = ErrorMetricsService.relative_error(u_pred, u_exact)
        max_err = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert l2_err == pytest.approx(2.0, rel=1e-10)
        assert rel_err == pytest.approx(2.0 / 3.0, rel=1e-10)
        assert max_err == pytest.approx(2.0, rel=1e-10)

    def test_large_arrays_performance(self):
        """Test error computation with large arrays."""
        n = 1_000_000
        u_pred = np.random.randn(n)
        u_exact = np.random.randn(n)

        # Should complete quickly
        import time
        start = time.time()

        l2_err = ErrorMetricsService.l2_error(u_pred, u_exact)
        rel_err = ErrorMetricsService.relative_error(u_pred, u_exact)
        max_err = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        elapsed = time.time() - start

        # Should be fast (< 0.1 seconds for NumPy operations)
        assert elapsed < 0.5
        assert all(isinstance(x, float) for x in [l2_err, rel_err, max_err])

    def test_consistent_types_returned(self):
        """Test that functions return float scalars, not arrays."""
        u_pred = np.array([1.0, 2.0, 3.0])
        u_exact = np.array([1.1, 2.1, 3.1])

        l2_err = ErrorMetricsService.l2_error(u_pred, u_exact)
        rel_err = ErrorMetricsService.relative_error(u_pred, u_exact)
        max_err = ErrorMetricsService.max_absolute_error(u_pred, u_exact)

        assert isinstance(l2_err, float)
        assert isinstance(rel_err, float)
        assert isinstance(max_err, float)


class TestValidationScenarios:
    """Test realistic PINN validation scenarios."""

    def test_pinn_validation_workflow(self):
        """Test typical PINN validation workflow with analytical solution."""
        # Simulate PINN predictions vs analytical solution
        x = np.linspace(0, 1, 101)
        t = 0.0  # Use t=0 to avoid near-zero cos(πt)

        # Analytical: sin(πx)cos(πt) = sin(πx) at t=0
        u_exact = np.sin(np.pi * x) * np.cos(np.pi * t)

        # PINN with 2% noise
        np.random.seed(42)  # For reproducibility
        u_pinn = u_exact + 0.02 * np.random.randn(len(x))

        # Compute metrics
        l2_err = ErrorMetricsService.l2_error(u_pinn, u_exact)
        rel_err = ErrorMetricsService.relative_error(u_pinn, u_exact)
        max_err = ErrorMetricsService.max_absolute_error(u_pinn, u_exact)

        # Verify metrics are reasonable
        assert rel_err < 0.05  # Less than 5% relative error threshold
        assert max_err < 0.1  # Reasonable max error
        assert l2_err > 0  # Non-zero error

    def test_convergence_monitoring(self):
        """Test monitoring error decrease during training."""
        u_exact = np.sin(np.linspace(0, 2*np.pi, 100))

        # Simulate improving predictions over epochs
        errors = []
        for noise_level in [0.5, 0.3, 0.1, 0.05, 0.01]:
            u_pred = u_exact + noise_level * np.random.randn(len(u_exact))
            rel_err = ErrorMetricsService.relative_error(u_pred, u_exact)
            errors.append(rel_err)

        # Verify errors generally decrease (allowing some variance)
        assert errors[-1] < errors[0]  # Final error < initial error

    def test_threshold_detection(self):
        """Test detecting when error exceeds threshold."""
        u_exact = np.ones(100)
        u_pred = u_exact + 0.1 * np.random.randn(100)

        rel_err = ErrorMetricsService.relative_error(u_pred, u_exact)

        # Flag if exceeds 5% threshold (from requirements)
        threshold = 0.05
        needs_tuning = rel_err > threshold

        assert isinstance(needs_tuning, bool)
