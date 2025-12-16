"""Analytical Solution Generator for 1D Wave Equation."""

from collections.abc import Callable
from typing import Literal

import numpy as np


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
        """Generate standing wave solution for Dirichlet BC: u(x,t) = sin(nπx/L) cos(nπct/L).

        This solution satisfies Dirichlet boundary conditions: u(0,t) = u(L,t) = 0.

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

        # Standing wave formula for Dirichlet BC
        return np.sin(n * np.pi * X / L) * np.cos(n * np.pi * c * T / L)

    @staticmethod
    def standing_wave_neumann(
        x: np.ndarray,
        t: np.ndarray,
        L: float,
        c: float,
        n: int = 0
    ) -> np.ndarray:
        """Generate standing wave solution for Neumann BC: u(x,t) = cos(nπx/L) cos(nπct/L).

        This solution satisfies Neumann boundary conditions: ∂u/∂x(0,t) = ∂u/∂x(L,t) = 0.

        Args:
            x: Spatial coordinates (N,) or (N, 1)
            t: Temporal coordinates (M,) or (M, 1)
            L: Domain length
            c: Wave speed
            n: Mode number (default: 0 for fundamental mode with Neumann BC)

        Returns:
            np.ndarray: Solution array (N, M) or (N,) if t is scalar

        Note:
            - For n=0: u(x,t) = cos(0) = 1 (constant solution)
            - For n=1: u(x,t) = cos(πx/L) cos(πct/L) (first mode)
            - For n=2: u(x,t) = cos(2πx/L) cos(2πct/L) (second mode)
        """
        # Ensure proper shapes
        x = np.atleast_1d(x).flatten()
        t = np.atleast_1d(t).flatten()

        # Create meshgrid for evaluation
        X, T = np.meshgrid(x, t, indexing="ij")

        # Standing wave formula for Neumann BC
        return np.cos(n * np.pi * X / L) * np.cos(n * np.pi * c * T / L)

    @staticmethod
    def traveling_wave(
        x: np.ndarray,
        t: np.ndarray,
        c: float,
        initial_condition: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Generate traveling wave: u(x,t) = 0.5 * [f(x - ct) + f(x + ct)].

        Uses d'Alembert's formula for zero initial velocity.

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

        # For d'Alembert's solution, we need to evaluate IC at shifted positions
        # Reshape to format expected by initial_condition: (N*M, 1)
        X_flat = X.flatten()
        T_flat = T.flatten()

        # Evaluate left-traveling wave: f(x - ct)
        pos_left = (X_flat - c * T_flat).reshape(-1, 1)
        u_left = initial_condition(pos_left)

        # Evaluate right-traveling wave: f(x + ct)
        pos_right = (X_flat + c * T_flat).reshape(-1, 1)
        u_right = initial_condition(pos_right)

        # d'Alembert's solution with zero initial velocity
        u_total = 0.5 * (u_left + u_right)

        # Reshape back to (nx, nt)
        return u_total.reshape(X.shape)

    def evaluate_at_points(
        self,
        points: np.ndarray,
        L: float,
        c: float,
        solution_type: Literal["standing", "traveling", "standing_neumann"] = "standing",
        n: int = 1
    ) -> np.ndarray:
        """Evaluate analytical solution at given spatiotemporal points.

        Args:
            points: Array of (x, t) coordinates, shape (N, 2)
            L: Domain length
            c: Wave speed
            solution_type: Type of solution ("standing", "standing_neumann", or "traveling")
            n: Mode number for standing wave

        Returns:
            np.ndarray: Solution values at each point, shape (N,)
        """
        x = points[:, 0]
        t = points[:, 1]

        if solution_type == "standing":
            # Evaluate standing wave pointwise (Dirichlet BC)
            return np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
        elif solution_type == "standing_neumann":
            # Evaluate standing wave pointwise (Neumann BC)
            return np.cos(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
        else:
            raise NotImplementedError("Traveling wave evaluation not yet implemented")
