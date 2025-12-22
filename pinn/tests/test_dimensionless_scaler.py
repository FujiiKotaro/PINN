"""Tests for dimensionless scaler service.

Test-Driven Development: Tests written before implementation.
Tests cover characteristic scale computation, normalization, and denormalization.
"""

import numpy as np
import pytest

from pinn.data.dimensionless_scaler import (
    CharacteristicScales,
    DimensionlessScalerService,
)


class TestCharacteristicScales:
    """Test characteristic scale computation from physics parameters."""

    def test_from_physics_computes_correct_scales(self):
        """Test that characteristic scales are computed correctly from material properties."""
        # Aluminum 6061 properties
        elastic_lambda = 58e9  # Pa
        elastic_mu = 26e9      # Pa
        density = 2700.0       # kg/m³
        domain_length = 0.04   # m

        scales = CharacteristicScales.from_physics(
            domain_length=domain_length,
            elastic_lambda=elastic_lambda,
            elastic_mu=elastic_mu,
            density=density,
            displacement_amplitude=1e-9  # 1 nm
        )

        # Verify L_ref equals domain length
        assert scales.L_ref == domain_length

        # Verify longitudinal wave speed c_l = sqrt((λ+2μ)/ρ)
        c_l_expected = np.sqrt((elastic_lambda + 2*elastic_mu) / density)
        assert np.isclose(scales.velocity_ref, c_l_expected, rtol=1e-6)

        # Verify T_ref = L_ref / c_l
        T_ref_expected = domain_length / c_l_expected
        assert np.isclose(scales.T_ref, T_ref_expected, rtol=1e-6)

        # Verify U_ref is set correctly
        assert scales.U_ref == 1e-9

        # Verify σ_ref = ρ * c_l²
        sigma_ref_expected = density * c_l_expected**2
        assert np.isclose(scales.sigma_ref, sigma_ref_expected, rtol=1e-6)

    def test_from_physics_validates_positive_parameters(self):
        """Test that negative or zero parameters raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            CharacteristicScales.from_physics(
                domain_length=-0.04,  # Invalid: negative
                elastic_lambda=58e9,
                elastic_mu=26e9,
                density=2700.0
            )

        with pytest.raises(ValueError, match="must be positive"):
            CharacteristicScales.from_physics(
                domain_length=0.04,
                elastic_lambda=0.0,  # Invalid: zero
                elastic_mu=26e9,
                density=2700.0
            )


class TestDimensionlessScalerService:
    """Test normalization and denormalization of spatiotemporal and field variables."""

    @pytest.fixture
    def scaler(self):
        """Create scaler with known characteristic scales."""
        scales = CharacteristicScales(
            L_ref=0.04,       # 40 mm
            T_ref=6.35e-6,    # ~6.35 microseconds
            U_ref=1e-9,       # 1 nm
            sigma_ref=1e11,   # 100 GPa
            velocity_ref=6300.0  # ~6300 m/s
        )
        return DimensionlessScalerService(scales)

    def test_normalize_inputs_scales_coordinates(self, scaler):
        """Test that spatial and temporal coordinates are normalized correctly."""
        # Physical coordinates
        x = np.array([0.0, 0.02, 0.04])  # 0, 20, 40 mm
        y = np.array([0.0, 0.01, 0.02])  # 0, 10, 20 mm
        t = np.array([3.5e-6, 5.0e-6, 6.5e-6])  # Time in seconds

        x_tilde, y_tilde, t_tilde = scaler.normalize_inputs(x, y, t)

        # Expected dimensionless coordinates
        expected_x = x / scaler.scales.L_ref  # [0, 0.5, 1]
        expected_y = y / scaler.scales.L_ref  # [0, 0.5, 1]
        expected_t = t / scaler.scales.T_ref  # [0.551, 0.787, 1.024]

        np.testing.assert_allclose(x_tilde, expected_x, rtol=1e-6)
        np.testing.assert_allclose(y_tilde, expected_y, rtol=1e-6)
        np.testing.assert_allclose(t_tilde, expected_t, rtol=1e-6)

    def test_normalize_outputs_scales_fields(self, scaler):
        """Test that output fields (stress and displacement) are normalized."""
        # Physical units
        T1 = np.array([1e9, 5e9, 1e10])  # Pa (stress)
        T3 = np.array([2e9, 6e9, 1.2e10])  # Pa (stress)
        Ux = np.array([1e-10, 5e-10, 1e-9])  # m (displacement)
        Uy = np.array([2e-10, 6e-10, 1.2e-9])  # m (displacement)

        T1_tilde, T3_tilde, Ux_tilde, Uy_tilde = scaler.normalize_outputs(
            T1, T3, Ux, Uy
        )

        # Expected dimensionless values
        expected_T1 = T1 / scaler.scales.sigma_ref
        expected_T3 = T3 / scaler.scales.sigma_ref
        expected_Ux = Ux / scaler.scales.U_ref
        expected_Uy = Uy / scaler.scales.U_ref

        np.testing.assert_allclose(T1_tilde, expected_T1, rtol=1e-6)
        np.testing.assert_allclose(T3_tilde, expected_T3, rtol=1e-6)
        np.testing.assert_allclose(Ux_tilde, expected_Ux, rtol=1e-6)
        np.testing.assert_allclose(Uy_tilde, expected_Uy, rtol=1e-6)

    def test_denormalize_outputs_inverts_normalization(self, scaler):
        """Test that denormalization recovers original physical units."""
        # Original physical values
        T1_orig = np.array([1e9, 5e9, 1e10])
        T3_orig = np.array([2e9, 6e9, 1.2e10])
        Ux_orig = np.array([1e-10, 5e-10, 1e-9])
        Uy_orig = np.array([2e-10, 6e-10, 1.2e-9])

        # Normalize then denormalize
        T1_tilde, T3_tilde, Ux_tilde, Uy_tilde = scaler.normalize_outputs(
            T1_orig, T3_orig, Ux_orig, Uy_orig
        )

        T1_recovered, T3_recovered, Ux_recovered, Uy_recovered = scaler.denormalize_outputs(
            T1_tilde, T3_tilde, Ux_tilde, Uy_tilde
        )

        # Verify recovery (within numerical precision)
        np.testing.assert_allclose(T1_recovered, T1_orig, rtol=1e-12)
        np.testing.assert_allclose(T3_recovered, T3_orig, rtol=1e-12)
        np.testing.assert_allclose(Ux_recovered, Ux_orig, rtol=1e-12)
        np.testing.assert_allclose(Uy_recovered, Uy_orig, rtol=1e-12)

    def test_normalized_values_are_order_one(self, scaler):
        """Test that normalized values are O(1) as intended for loss scaling."""
        # Typical physical values from FDTD data
        x = np.linspace(0, 0.04, 100)  # Full spatial domain
        y = np.linspace(0, 0.02, 50)
        t = np.linspace(3.5e-6, 6.5e-6, 30)

        T1 = np.random.uniform(1e9, 1e10, 100)  # Typical stress range
        Ux = np.random.uniform(1e-10, 1e-9, 100)  # Typical displacement range

        x_tilde, y_tilde, t_tilde = scaler.normalize_inputs(
            x[:100], y[:50][:100], t[:30][:100]
        )
        T1_tilde, T3_tilde, Ux_tilde, Uy_tilde = scaler.normalize_outputs(
            T1, T1, Ux, Ux
        )

        # All normalized values should be O(1), i.e., in range [0.01, 100]
        assert np.all(x_tilde >= 0) and np.all(x_tilde <= 1.1)
        assert np.all(y_tilde >= 0) and np.all(y_tilde <= 1.1)
        assert np.all(t_tilde >= 0) and np.all(t_tilde <= 2.0)

        # Stress and displacement should be O(1)
        assert np.all(np.abs(T1_tilde) < 100)
        assert np.all(np.abs(Ux_tilde) < 100)
