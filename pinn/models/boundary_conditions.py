"""Boundary and initial condition builders for DeepXDE.

This module provides utilities for creating boundary conditions (Dirichlet, Neumann)
and initial conditions (displacement, velocity) for the 1D wave equation PINN.
"""

from collections.abc import Callable

import deepxde as dde
import numpy as np


class BoundaryConditionsService:
    """Service for creating boundary and initial conditions."""

    @staticmethod
    def create_dirichlet_bc(
        geomtime: dde.geometry.GeometryXTime,
        func: Callable,
        on_boundary: Callable,
        component: int = 0
    ) -> dde.icbc.DirichletBC:
        """Create Dirichlet BC: u(x, t) = func(x, t) on specified boundary.

        Args:
            geomtime: DeepXDE GeometryXTime object
            func: Function defining boundary values, signature: func(x, on_boundary) -> values
            on_boundary: Function defining boundary location, signature: on_boundary(x, on_boundary) -> bool
            component: Component index for multi-output networks (default: 0)

        Returns:
            DeepXDE DirichletBC object
        """
        return dde.icbc.DirichletBC(geomtime, func, on_boundary, component=component)

    @staticmethod
    def create_neumann_bc(
        geomtime: dde.geometry.GeometryXTime,
        func: Callable,
        on_boundary: Callable,
        component: int = 0
    ) -> dde.icbc.NeumannBC:
        """Create Neumann BC: ∂u/∂n(x, t) = func(x, t) on specified boundary.

        Args:
            geomtime: DeepXDE GeometryXTime object
            func: Function defining normal derivative values
            on_boundary: Function defining boundary location
            component: Component index for multi-output networks (default: 0)

        Returns:
            DeepXDE NeumannBC object
        """
        return dde.icbc.NeumannBC(geomtime, func, on_boundary, component=component)

    @staticmethod
    def create_initial_condition(
        geomtime: dde.geometry.GeometryXTime,
        func: Callable,
        component: int = 0
    ) -> dde.icbc.IC:
        """Create initial condition: u(x, 0) = func(x).

        Args:
            geomtime: DeepXDE GeometryXTime object
            func: Function defining initial displacement, signature: func(x) -> values
            component: Component index for multi-output networks (default: 0)

        Returns:
            DeepXDE IC object
        """
        return dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial, component=component)

    @staticmethod
    def create_initial_velocity(
        geomtime: dde.geometry.GeometryXTime,
        func: Callable
    ) -> dde.icbc.OperatorBC:
        """Create initial velocity condition: ∂u/∂t(x, 0) = func(x).

        Args:
            geomtime: DeepXDE GeometryXTime object
            func: Function defining initial velocity

        Returns:
            DeepXDE OperatorBC object for temporal derivative at t=0
        """
        import torch

        def velocity_op(x, y, X):
            """Operator for computing ∂u/∂t - func(x) at t=0.

            Args:
                x: Input tensor for autograd (requires_grad=True)
                y: Network output tensor
                X: Input values as numpy array
            """
            du_dt = dde.grad.jacobian(y, x, j=1)
            # func takes numpy array X and returns numpy array
            func_val = func(X)
            # Convert func_val to tensor if needed
            if isinstance(du_dt, torch.Tensor):
                func_val = torch.from_numpy(func_val).to(du_dt.device).to(du_dt.dtype)
            return du_dt - func_val

        return dde.icbc.OperatorBC(
            geomtime,
            velocity_op,
            lambda x, on_initial: on_initial
        )

    @staticmethod
    def create_zero_dirichlet_bc(
        geomtime: dde.geometry.GeometryXTime
    ) -> dde.icbc.DirichletBC:
        """Create zero Dirichlet BC: u = 0 at boundaries.

        Convenience method for homogeneous Dirichlet boundary conditions.

        Args:
            geomtime: DeepXDE GeometryXTime object

        Returns:
            DeepXDE DirichletBC object with u=0
        """
        def zero_func(x):
            return np.zeros((len(x), 1))

        def on_boundary(x, on_boundary):
            return on_boundary

        return BoundaryConditionsService.create_dirichlet_bc(
            geomtime, zero_func, on_boundary
        )

    @staticmethod
    def create_zero_initial_velocity(
        geomtime: dde.geometry.GeometryXTime
    ) -> dde.icbc.OperatorBC:
        """Create zero initial velocity: ∂u/∂t(x, 0) = 0.

        Convenience method for stationary initial condition.

        Args:
            geomtime: DeepXDE GeometryXTime object

        Returns:
            DeepXDE OperatorBC object with ∂u/∂t=0 at t=0
        """
        def velocity_op(x, y, X):
            """Operator for computing ∂u/∂t at t=0.

            Args:
                x: Input tensor for autograd (requires_grad=True)
                y: Network output tensor
                X: Input values as numpy array
            """
            return dde.grad.jacobian(y, x, j=1)

        return dde.icbc.OperatorBC(
            geomtime,
            velocity_op,
            lambda x, on_initial: on_initial
        )
