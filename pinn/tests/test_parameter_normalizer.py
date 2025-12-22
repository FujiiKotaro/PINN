"""Tests for parameter normalization service.

Test-Driven Development: Tests written before implementation.
Tests cover pitch/depth normalization for parametric PINN learning.
"""

import numpy as np
import pytest

from pinn.data.parameter_normalizer import ParameterNormalizer


class TestParameterNormalizer:
    """Test parameter normalization to [0, 1] range."""

    def test_normalize_pitch_at_min_bound(self):
        """Test pitch normalization at minimum bound (1.25mm → 0.0)."""
        pitch = np.array([1.25e-3])  # 1.25mm in meters

        normalized = ParameterNormalizer.normalize_pitch(pitch)

        assert np.isclose(normalized[0], 0.0, atol=1e-10), \
            f"Expected 0.0 at min bound, got {normalized[0]}"

    def test_normalize_pitch_at_max_bound(self):
        """Test pitch normalization at maximum bound (2.0mm → 1.0)."""
        pitch = np.array([2.0e-3])  # 2.0mm in meters

        normalized = ParameterNormalizer.normalize_pitch(pitch)

        assert np.isclose(normalized[0], 1.0, atol=1e-10), \
            f"Expected 1.0 at max bound, got {normalized[0]}"

    def test_normalize_pitch_at_midpoint(self):
        """Test pitch normalization at midpoint (1.625mm → 0.5)."""
        pitch = np.array([1.625e-3])  # Midpoint between 1.25 and 2.0mm

        normalized = ParameterNormalizer.normalize_pitch(pitch)

        expected = (1.625e-3 - 1.25e-3) / (2.0e-3 - 1.25e-3)
        assert np.isclose(normalized[0], expected, atol=1e-10), \
            f"Expected {expected}, got {normalized[0]}"

    def test_normalize_pitch_array(self):
        """Test pitch normalization with array input."""
        pitch = np.array([1.25e-3, 1.5e-3, 1.75e-3, 2.0e-3])

        normalized = ParameterNormalizer.normalize_pitch(pitch)

        expected = np.array([0.0, 1/3, 2/3, 1.0])
        np.testing.assert_allclose(normalized, expected, atol=1e-10)

    def test_normalize_depth_at_min_bound(self):
        """Test depth normalization at minimum bound (0.1mm → 0.0)."""
        depth = np.array([0.1e-3])  # 0.1mm in meters

        normalized = ParameterNormalizer.normalize_depth(depth)

        assert np.isclose(normalized[0], 0.0, atol=1e-10), \
            f"Expected 0.0 at min bound, got {normalized[0]}"

    def test_normalize_depth_at_max_bound(self):
        """Test depth normalization at maximum bound (0.3mm → 1.0)."""
        depth = np.array([0.3e-3])  # 0.3mm in meters

        normalized = ParameterNormalizer.normalize_depth(depth)

        assert np.isclose(normalized[0], 1.0, atol=1e-10), \
            f"Expected 1.0 at max bound, got {normalized[0]}"

    def test_normalize_depth_at_midpoint(self):
        """Test depth normalization at midpoint (0.2mm → 0.5)."""
        depth = np.array([0.2e-3])  # Midpoint between 0.1 and 0.3mm

        normalized = ParameterNormalizer.normalize_depth(depth)

        expected = (0.2e-3 - 0.1e-3) / (0.3e-3 - 0.1e-3)
        assert np.isclose(normalized[0], expected, atol=1e-10), \
            f"Expected {expected}, got {normalized[0]}"

    def test_normalize_depth_array(self):
        """Test depth normalization with array input."""
        depth = np.array([0.1e-3, 0.15e-3, 0.2e-3, 0.25e-3, 0.3e-3])

        normalized = ParameterNormalizer.normalize_depth(depth)

        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(normalized, expected, atol=1e-10)

    def test_denormalize_pitch_inverts_normalization(self):
        """Test that denormalize_pitch inverts normalize_pitch."""
        original = np.array([1.25e-3, 1.5e-3, 1.75e-3, 2.0e-3])

        normalized = ParameterNormalizer.normalize_pitch(original)
        recovered = ParameterNormalizer.denormalize_pitch(normalized)

        np.testing.assert_allclose(recovered, original, rtol=1e-12)

    def test_denormalize_depth_inverts_normalization(self):
        """Test that denormalize_depth inverts normalize_depth."""
        original = np.array([0.1e-3, 0.15e-3, 0.2e-3, 0.25e-3, 0.3e-3])

        normalized = ParameterNormalizer.normalize_depth(original)
        recovered = ParameterNormalizer.denormalize_depth(normalized)

        np.testing.assert_allclose(recovered, original, rtol=1e-12)

    def test_denormalize_pitch_at_bounds(self):
        """Test denormalize_pitch at normalized bounds [0, 1]."""
        normalized = np.array([0.0, 0.5, 1.0])

        denormalized = ParameterNormalizer.denormalize_pitch(normalized)

        expected = np.array([1.25e-3, 1.625e-3, 2.0e-3])
        np.testing.assert_allclose(denormalized, expected, atol=1e-10)

    def test_denormalize_depth_at_bounds(self):
        """Test denormalize_depth at normalized bounds [0, 1]."""
        normalized = np.array([0.0, 0.5, 1.0])

        denormalized = ParameterNormalizer.denormalize_depth(normalized)

        expected = np.array([0.1e-3, 0.2e-3, 0.3e-3])
        np.testing.assert_allclose(denormalized, expected, atol=1e-10)

    def test_constants_are_correct(self):
        """Test that class constants match specification."""
        assert ParameterNormalizer.PITCH_MIN == 1.25e-3, "PITCH_MIN should be 1.25mm"
        assert ParameterNormalizer.PITCH_MAX == 2.0e-3, "PITCH_MAX should be 2.0mm"
        assert ParameterNormalizer.DEPTH_MIN == 0.1e-3, "DEPTH_MIN should be 0.1mm"
        assert ParameterNormalizer.DEPTH_MAX == 0.3e-3, "DEPTH_MAX should be 0.3mm"

    def test_normalize_preserves_array_shape(self):
        """Test that normalization preserves input array shape."""
        pitch_1d = np.array([1.5e-3, 1.75e-3])
        depth_2d = np.array([[0.1e-3, 0.2e-3], [0.25e-3, 0.3e-3]])

        pitch_norm = ParameterNormalizer.normalize_pitch(pitch_1d)
        depth_norm = ParameterNormalizer.normalize_depth(depth_2d)

        assert pitch_norm.shape == pitch_1d.shape, "Shape should be preserved"
        assert depth_norm.shape == depth_2d.shape, "Shape should be preserved"
