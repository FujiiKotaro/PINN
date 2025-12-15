"""Analytical Solution Generator for 1D Wave Equation."""

import numpy as np
from typing import Callable, Literal


class AnalyticalSolutionGeneratorService:
    """Generate analytical solutions for standing and traveling waves."""

    @staticmethod
    def standing_wave(
        x: np.ndarray,
        t: np.ndarray,
        L: float,
        c: float,
        n: int = 1
    ) -> np.ndarray:
        """Generate standing wave solution: u(x,t) = sin(nπx/L) cos(nπct/L).

        Args:
            x: Spatial coordinates (N,) or (N, 1)
            t: Temporal coordinates (M,) or (M, 1)
            L: Domain length
            c: Wave speed
            n: Mode number (default: fundamental mode)

        Returns:
            np.ndarray: Solution array (N, M) or (N,) if t is scalar
        """
        # Ensure proper shapes
        x = np.atleast_1d(x).flatten()
        t = np.atleast_1d(t).flatten()

        # Create meshgrid for evaluation
        X, T = np.meshgrid(x, t, indexing="ij")

        # Standing wave formula
        return np.sin(n * np.pi * X / L) * np.cos(n * np.pi * c * T / L)

    @staticmethod
    def traveling_wave(
        x: np.ndarray,
        t: np.ndarray,
        c: float,
        initial_condition: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Generate traveling wave: u(x,t) = f(x - ct) + g(x + ct).

        Args:
            x: Spatial coordinates
            t: Temporal coordinates
            c: Wave speed
            initial_condition: Function f(x) for initial displacement

        Returns:
            np.ndarray: Solution array (N, M)
        """
        # Ensure proper shapes
        x = np.atleast_1d(x).flatten()
        t = np.atleast_1d(t).flatten()

        # Create meshgrid
        X, T = np.meshgrid(x, t, indexing="ij")

        # d'Alembert's solution: superposition of left and right traveling waves
        return initial_condition(X - c*T) + initial_condition(X + c*T)

    def evaluate_at_points(
        self,
        points: np.ndarray,
        L: float,
        c: float,
        solution_type: Literal["standing", "traveling"] = "standing",
        n: int = 1
    ) -> np.ndarray:
        """Evaluate analytical solution at given spatiotemporal points.

        Args:
            points: Array of (x, t) coordinates, shape (N, 2)
            L: Domain length
            c: Wave speed
            solution_type: Type of solution ("standing" or "traveling")
            n: Mode number for standing wave

        Returns:
            np.ndarray: Solution values at each point, shape (N,)
        """
        x = points[:, 0]
        t = points[:, 1]

        if solution_type == "standing":
            # Evaluate standing wave pointwise
            return np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
        else:
            raise NotImplementedError("Traveling wave evaluation not yet implemented")
