"""Error Metrics for PINN Validation."""

import numpy as np


class ErrorMetricsService:
    """Compute L2 and relative errors between PINN predictions and analytical solutions."""

    @staticmethod
    def l2_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
        """Compute L2 norm of error: ||u_pred - u_exact||₂.

        Args:
            u_pred: Predicted values
            u_exact: Exact analytical values

        Returns:
            float: L2 error
        """
        return float(np.linalg.norm(u_pred - u_exact))

    @staticmethod
    def relative_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
        """Compute relative L2 error: ||u_pred - u_exact||₂ / ||u_exact||₂.

        Args:
            u_pred: Predicted values
            u_exact: Exact analytical values

        Returns:
            float: Relative L2 error
        """
        return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))

    @staticmethod
    def max_absolute_error(u_pred: np.ndarray, u_exact: np.ndarray) -> float:
        """Compute maximum pointwise error.

        Args:
            u_pred: Predicted values
            u_exact: Exact analytical values

        Returns:
            float: Maximum absolute error
        """
        return float(np.max(np.abs(u_pred - u_exact)))
