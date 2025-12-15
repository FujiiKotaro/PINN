"""Tests for boundary and initial condition builders.

This module tests the BoundaryConditionsService for creating DeepXDE
boundary and initial condition objects.
"""

import pytest
import torch
import numpy as np
import deepxde as dde

from pinn.models.boundary_conditions import BoundaryConditionsService


class TestBoundaryConditionsService:
    """Test suite for boundary and initial condition builders."""

    @pytest.fixture
    def geomtime(self):
        """Create a simple geometry for testing."""
        geom = dde.geometry.Interval(0, 1)
        timedomain = dde.geometry.TimeDomain(0, 1)
        return dde.geometry.GeometryXTime(geom, timedomain)

    def test_create_dirichlet_bc(self, geomtime) -> None:
        """Test creation of Dirichlet boundary condition."""
        def bc_func(x, _):
            return np.zeros((len(x), 1))

        def on_boundary(x, on_boundary):
            return on_boundary

        bc = BoundaryConditionsService.create_dirichlet_bc(
            geomtime, bc_func, on_boundary
        )

        assert bc is not None
        assert isinstance(bc, dde.icbc.DirichletBC)

    def test_create_neumann_bc(self, geomtime) -> None:
        """Test creation of Neumann boundary condition."""
        def bc_func(x):
            return np.zeros((len(x), 1))

        def on_boundary(x, on_boundary):
            return on_boundary

        bc = BoundaryConditionsService.create_neumann_bc(
            geomtime, bc_func, on_boundary
        )

        assert bc is not None
        assert isinstance(bc, dde.icbc.NeumannBC)

    def test_create_initial_condition(self, geomtime) -> None:
        """Test creation of initial condition for displacement."""
        def ic_func(x):
            return np.sin(np.pi * x[:, 0:1])

        ic = BoundaryConditionsService.create_initial_condition(
            geomtime, ic_func
        )

        assert ic is not None
        assert isinstance(ic, dde.icbc.IC)

    def test_create_initial_velocity(self, geomtime) -> None:
        """Test creation of initial velocity condition."""
        def ic_func(x):
            return np.zeros((len(x), 1))

        ic = BoundaryConditionsService.create_initial_velocity(
            geomtime, ic_func
        )

        assert ic is not None
        assert isinstance(ic, dde.icbc.OperatorBC)

    def test_create_zero_dirichlet_bc(self, geomtime) -> None:
        """Test helper for zero Dirichlet BC."""
        bc = BoundaryConditionsService.create_zero_dirichlet_bc(geomtime)

        assert bc is not None
        assert isinstance(bc, dde.icbc.DirichletBC)

    def test_create_zero_initial_velocity(self, geomtime) -> None:
        """Test helper for zero initial velocity."""
        ic = BoundaryConditionsService.create_zero_initial_velocity(geomtime)

        assert ic is not None
        assert isinstance(ic, dde.icbc.OperatorBC)
