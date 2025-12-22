"""Parameter normalization service for parametric PINN learning.

This module provides normalization/denormalization for crack parameters (pitch, depth)
to enable parametric learning across parameter space.
"""

import numpy as np


class ParameterNormalizer:
    """Normalize crack parameters to [0, 1] range for neural network input.

    Normalizes:
    - Pitch: [1.25mm, 2.0mm] → [0, 1]
    - Depth: [0.1mm, 0.3mm] → [0, 1]

    This normalization enables the neural network to learn parameter dependencies
    more effectively by bringing all inputs to similar scales.
    """

    # Parameter ranges in meters
    PITCH_MIN = 1.25e-3  # 1.25mm
    PITCH_MAX = 2.0e-3   # 2.0mm
    DEPTH_MIN = 0.1e-3   # 0.1mm
    DEPTH_MAX = 0.3e-3   # 0.3mm

    @staticmethod
    def normalize_pitch(pitch: np.ndarray) -> np.ndarray:
        """Normalize pitch to [0, 1] range.

        Args:
            pitch: Pitch values in meters (raw physical units)

        Returns:
            Normalized pitch in [0, 1]

        Preconditions:
            - pitch values should be in [PITCH_MIN, PITCH_MAX] range
              (values outside range are still normalized linearly)

        Postconditions:
            - PITCH_MIN → 0.0
            - PITCH_MAX → 1.0
            - Linear mapping preserves relative distances

        Invariants:
            - normalize → denormalize recovers original value

        Example:
            >>> pitch = np.array([1.25e-3, 1.625e-3, 2.0e-3])
            >>> normalized = ParameterNormalizer.normalize_pitch(pitch)
            >>> # [0.0, 0.5, 1.0]
        """
        return (pitch - ParameterNormalizer.PITCH_MIN) / \
               (ParameterNormalizer.PITCH_MAX - ParameterNormalizer.PITCH_MIN)

    @staticmethod
    def normalize_depth(depth: np.ndarray) -> np.ndarray:
        """Normalize depth to [0, 1] range.

        Args:
            depth: Depth values in meters (raw physical units)

        Returns:
            Normalized depth in [0, 1]

        Preconditions:
            - depth values should be in [DEPTH_MIN, DEPTH_MAX] range
              (values outside range are still normalized linearly)

        Postconditions:
            - DEPTH_MIN → 0.0
            - DEPTH_MAX → 1.0
            - Linear mapping preserves relative distances

        Invariants:
            - normalize → denormalize recovers original value

        Example:
            >>> depth = np.array([0.1e-3, 0.2e-3, 0.3e-3])
            >>> normalized = ParameterNormalizer.normalize_depth(depth)
            >>> # [0.0, 0.5, 1.0]
        """
        return (depth - ParameterNormalizer.DEPTH_MIN) / \
               (ParameterNormalizer.DEPTH_MAX - ParameterNormalizer.DEPTH_MIN)

    @staticmethod
    def denormalize_pitch(pitch_norm: np.ndarray) -> np.ndarray:
        """Denormalize pitch from [0, 1] back to physical units (meters).

        Args:
            pitch_norm: Normalized pitch in [0, 1]

        Returns:
            Pitch in meters

        Postconditions:
            - Inverse of normalize_pitch
            - 0.0 → PITCH_MIN
            - 1.0 → PITCH_MAX

        Example:
            >>> pitch_norm = np.array([0.0, 0.5, 1.0])
            >>> pitch = ParameterNormalizer.denormalize_pitch(pitch_norm)
            >>> # [1.25e-3, 1.625e-3, 2.0e-3] meters
        """
        return pitch_norm * (ParameterNormalizer.PITCH_MAX - ParameterNormalizer.PITCH_MIN) + \
               ParameterNormalizer.PITCH_MIN

    @staticmethod
    def denormalize_depth(depth_norm: np.ndarray) -> np.ndarray:
        """Denormalize depth from [0, 1] back to physical units (meters).

        Args:
            depth_norm: Normalized depth in [0, 1]

        Returns:
            Depth in meters

        Postconditions:
            - Inverse of normalize_depth
            - 0.0 → DEPTH_MIN
            - 1.0 → DEPTH_MAX

        Example:
            >>> depth_norm = np.array([0.0, 0.5, 1.0])
            >>> depth = ParameterNormalizer.denormalize_depth(depth_norm)
            >>> # [0.1e-3, 0.2e-3, 0.3e-3] meters
        """
        return depth_norm * (ParameterNormalizer.DEPTH_MAX - ParameterNormalizer.DEPTH_MIN) + \
               ParameterNormalizer.DEPTH_MIN
