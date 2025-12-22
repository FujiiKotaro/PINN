"""Dimensionless scaler for physics-informed neural network training.

This module provides functionality to non-dimensionalize spatiotemporal
coordinates and field variables to address loss scaling problems in PINN training.
By scaling all variables to O(1), PDE loss and data loss become comparable in magnitude.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CharacteristicScales:
    """Characteristic scales for non-dimensionalization.

    Attributes:
        L_ref: Spatial reference scale (m)
        T_ref: Temporal reference scale (s)
        U_ref: Displacement reference scale (m)
        sigma_ref: Stress reference scale (Pa)
        velocity_ref: Characteristic velocity = L_ref / T_ref (m/s)
    """
    L_ref: float
    T_ref: float
    U_ref: float
    sigma_ref: float
    velocity_ref: float

    @classmethod
    def from_physics(
        cls,
        domain_length: float,
        elastic_lambda: float,
        elastic_mu: float,
        density: float,
        displacement_amplitude: float = 1e-9
    ) -> 'CharacteristicScales':
        """Compute characteristic scales from physics parameters.

        Args:
            domain_length: Spatial domain size (x_max), m
            elastic_lambda: Lamé's first parameter, Pa
            elastic_mu: Shear modulus, Pa
            density: Material density, kg/m³
            displacement_amplitude: Typical displacement amplitude (from FDTD stats), m

        Returns:
            CharacteristicScales with derived reference scales

        Raises:
            ValueError: If any parameter is not positive

        Example:
            >>> scales = CharacteristicScales.from_physics(
            ...     domain_length=0.04,
            ...     elastic_lambda=58e9,
            ...     elastic_mu=26e9,
            ...     density=2700.0,
            ...     displacement_amplitude=1e-9
            ... )
            >>> print(f"T_ref = {scales.T_ref:.2e} s")
        """
        # Validate positive parameters
        if domain_length <= 0:
            raise ValueError(f"domain_length must be positive, got {domain_length}")
        if elastic_lambda <= 0:
            raise ValueError(f"elastic_lambda must be positive, got {elastic_lambda}")
        if elastic_mu <= 0:
            raise ValueError(f"elastic_mu must be positive, got {elastic_mu}")
        if density <= 0:
            raise ValueError(f"density must be positive, got {density}")
        if displacement_amplitude <= 0:
            raise ValueError(f"displacement_amplitude must be positive, got {displacement_amplitude}")

        # Spatial reference scale (domain size)
        L_ref = domain_length

        # Longitudinal wave speed: c_l = sqrt((λ+2μ)/ρ)
        c_l = np.sqrt((elastic_lambda + 2*elastic_mu) / density)

        # Temporal reference scale: T_ref = L_ref / c_l
        T_ref = L_ref / c_l

        # Displacement reference scale (from typical FDTD data)
        U_ref = displacement_amplitude

        # Stress reference scale (characteristic impedance): σ_ref = ρ * c_l²
        sigma_ref = density * c_l**2

        return cls(
            L_ref=L_ref,
            T_ref=T_ref,
            U_ref=U_ref,
            sigma_ref=sigma_ref,
            velocity_ref=c_l
        )


class DimensionlessScalerService:
    """Non-dimensionalization service for PINN training.

    This service normalizes spatiotemporal coordinates and field variables
    using characteristic scales derived from physics parameters.
    Ensures all variables are O(1) to prevent loss scaling imbalances.
    """

    def __init__(self, scales: CharacteristicScales):
        """Initialize scaler with characteristic scales.

        Args:
            scales: Characteristic scales from physics or data

        Preconditions:
            - All scales must be positive
        """
        self.scales = scales

    def normalize_inputs(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize spatial and temporal coordinates.

        Args:
            x: x-coordinates (N,) in meters
            y: y-coordinates (N,) in meters
            t: time coordinates (N,) in seconds

        Returns:
            (x_tilde, y_tilde, t_tilde) dimensionless coordinates

        Postconditions:
            - x_tilde = x / L_ref ∈ [0, 1] for domain [0, L_ref]
            - y_tilde = y / L_ref ∈ [0, 0.5] for domain [0, L_ref/2]
            - t_tilde = t / T_ref ∈ [0.5, 1.0] for typical simulation window

        Example:
            >>> scaler = DimensionlessScalerService(scales)
            >>> x_tilde, y_tilde, t_tilde = scaler.normalize_inputs(x, y, t)
        """
        x_tilde = x / self.scales.L_ref
        y_tilde = y / self.scales.L_ref
        t_tilde = t / self.scales.T_ref
        return x_tilde, y_tilde, t_tilde

    def normalize_outputs(
        self,
        T1: np.ndarray,
        T3: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize output fields (stress and displacement).

        Args:
            T1, T3: Stress components (N,) in Pa
            Ux, Uy: Displacement components (N,) in m

        Returns:
            (T1_tilde, T3_tilde, Ux_tilde, Uy_tilde) dimensionless fields

        Postconditions:
            - T1_tilde, T3_tilde = O(1) (stress scaled by sigma_ref)
            - Ux_tilde, Uy_tilde = O(1) (displacement scaled by U_ref)

        Example:
            >>> T1_tilde, T3_tilde, Ux_tilde, Uy_tilde = scaler.normalize_outputs(
            ...     T1, T3, Ux, Uy
            ... )
        """
        T1_tilde = T1 / self.scales.sigma_ref
        T3_tilde = T3 / self.scales.sigma_ref
        Ux_tilde = Ux / self.scales.U_ref
        Uy_tilde = Uy / self.scales.U_ref
        return T1_tilde, T3_tilde, Ux_tilde, Uy_tilde

    def denormalize_outputs(
        self,
        T1_tilde: np.ndarray,
        T3_tilde: np.ndarray,
        Ux_tilde: np.ndarray,
        Uy_tilde: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert dimensionless outputs back to physical units.

        Args:
            T1_tilde, T3_tilde, Ux_tilde, Uy_tilde: Dimensionless outputs

        Returns:
            (T1, T3, Ux, Uy) in physical units (Pa, Pa, m, m)

        Postconditions:
            - Inverse of normalize_outputs
            - normalize(x) → denormalize → recovers x (within numerical precision)

        Example:
            >>> T1, T3, Ux, Uy = scaler.denormalize_outputs(
            ...     T1_tilde, T3_tilde, Ux_tilde, Uy_tilde
            ... )
        """
        T1 = T1_tilde * self.scales.sigma_ref
        T3 = T3_tilde * self.scales.sigma_ref
        Ux = Ux_tilde * self.scales.U_ref
        Uy = Uy_tilde * self.scales.U_ref
        return T1, T3, Ux, Uy
