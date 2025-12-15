import pytest
import numpy as np
import deepxde as dde
from src.pinn import PINNModelBuilder, is_on_rough_surface, InverseProblemSolver

def test_pinn_model_builder_creation():
    """Tests that the PINNModelBuilder can be instantiated."""
    # These would come from the NonDimensionalizer
    dimensionless_params = {
        'x_domain': [0, 1],
        'y_domain': [0, 2],
        't_domain': [0, 1],
    }
    builder = PINNModelBuilder(dimensionless_params)
    assert builder is not None

def test_get_geometry():
    """Tests that the get_geometry method returns a valid GeometryXTime object."""
    dimensionless_params = {
        'x_domain': [0, 1],
        'y_domain': [0, 2],
        't_domain': [0, 1],
    }
    builder = PINNModelBuilder(dimensionless_params)
    geom = builder.get_geometry()
    
    assert isinstance(geom, dde.geometry.GeometryXTime)
    assert isinstance(geom.geometry, dde.geometry.Rectangle)
    assert isinstance(geom.timedomain, dde.geometry.TimeDomain)
    
    # Check dimensions
    assert geom.geometry.xmin[0] == 0
    assert geom.geometry.xmax[0] == 1
    assert geom.geometry.xmin[1] == 0
    assert geom.geometry.xmax[1] == 2
    assert geom.timedomain.t0 == 0
    assert geom.timedomain.t1 == 1

def test_define_pde_system_with_quadratic_input():
    """
    Tests that the PDE system correctly computes residuals for a quadratic input field.
    """
    params = {
        'x_domain': [0, 1], 'y_domain': [0, 2], 't_domain': [0, 1],
        'c11': 1.0, 'c13': 0.5, 'c55': 0.25, 'rho': 1.0 # Dummy dimensionless constants
    }
    builder = PINNModelBuilder(params)

    import torch
    # x represents (x_coord, y_coord, t_coord)
    mock_x = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    
    x_coord = mock_x[:, 0:1]
    y_coord = mock_x[:, 1:2]
    
    # Define a quadratic function for u and v
    u = 0.1 * x_coord**2
    v = 0.5 * y_coord**2
    mock_y_quad = torch.cat((u, v), dim=1)

    residuals = builder.define_pde_system(mock_x, mock_y_quad)

    assert isinstance(residuals, list)
    assert len(residuals) == 2

    # Analytical residuals
    # pde1 = rho*u_tt - (c11*u_xx + c55*u_yy + (c13+c55)*v_xy)
    # u_tt=0, u_xx=0.2, u_yy=0, v_xy=0
    # pde1_expected = 0 - (1.0 * 0.2 + 0 + 0) = -0.2
    pde1_expected = -params['c11'] * 0.2

    # pde2 = rho*v_tt - (c55*v_xx + c11*v_yy + (c13+c55)*u_xy)
    # v_tt=0, v_xx=0, v_yy=1.0, u_xy=0
    # pde2_expected = 0 - (0 + 1.0 * 1.0 + 0) = -1.0
    pde2_expected = -params['c11'] * 1.0

    assert torch.allclose(residuals[0], torch.tensor(pde1_expected))
    assert torch.allclose(residuals[1], torch.tensor(pde2_expected))

def test_define_boundary_conditions():
    """Tests that the boundary condition method returns the correct number of BCs without a source."""
    params = {
        'x_domain': [0, 1], 'y_domain': [0, 2], 't_domain': [0, 1],
        'c11': 1.0, 'c13': 0.5, 'c55': 0.25, 'rho': 1.0,
        'f_pitch': 0.2, 'f_width': 0.1, 'f_depth': 0.1, # Needed for rough surface
    }
    builder = PINNModelBuilder(params)
    bcs = builder.define_boundary_conditions(source_func=None)

    assert isinstance(bcs, list)
    # 2 for y=0, 2 for y=max, 2 for rough bottom = 6
    assert len(bcs) == 6
    assert all(isinstance(bc, dde.icbc.BC) for bc in bcs)

def test_define_source_wave_bc():
    """Tests that providing a source function adds an extra BC."""
    params = {
        'x_domain': [0, 1], 'y_domain': [0, 2], 't_domain': [0, 1],
        'c11': 1.0, 'c13': 0.5, 'c55': 0.25, 'rho': 1.0,
        'f_pitch': 0.2, 'f_width': 0.1, 'f_depth': 0.1,
    }
    def source_func(t):
        return t * 10
    
    builder = PINNModelBuilder(params)
    bcs = builder.define_boundary_conditions(source_func=source_func)

    assert isinstance(bcs, list)
    # 6 stress-free BCs + 1 source BC = 7
    assert len(bcs) == 7
    assert all(isinstance(bc, dde.icbc.BC) for bc in bcs)

# --- Boundary Logic Tests ---

@pytest.fixture
def rough_surface_params():
    return {
        'x_domain': [0, 1.0],
        'f_pitch': 0.2, # pitch
        'f_width': 0.1, # width of notch (so flat part is 0.1 wide)
        'f_depth': 0.1, # depth of notch
    }

def test_point_on_flat_part(rough_surface_params):
    # This point is on the bottom boundary, in the flat section
    point = np.array([1.0, 0.05, 0.5])
    assert is_on_rough_surface(point, rough_surface_params)

def test_point_on_notch_bottom(rough_surface_params):
    # This point is on the bottom of a notch
    point = np.array([0.9, 0.15, 0.5])
    assert is_on_rough_surface(point, rough_surface_params)

def test_point_on_vertical_wall(rough_surface_params):
    # This point is on the vertical wall starting a notch
    point = np.array([0.95, 0.1, 0.5])
    assert is_on_rough_surface(point, rough_surface_params)

def test_point_inside_material(rough_surface_params):
    # This point is inside the material, not on a boundary
    point = np.array([0.5, 0.5, 0.5])
    assert not is_on_rough_surface(point, rough_surface_params)

def test_point_inside_notch_but_not_on_boundary(rough_surface_params):
    # This point is inside the notch space, not on the material boundary
    point = np.array([0.95, 0.15, 0.5])
    assert not is_on_rough_surface(point, rough_surface_params)

# --- Inverse Problem Tests ---

def test_inverse_problem_solver_creation():
    """Tests that the InverseProblemSolver can be instantiated."""
    solver = InverseProblemSolver(initial_pitch=1.5, initial_depth=0.2)
    assert solver is not None

def test_get_inverse_variables():
    """Tests that the correct trainable variables are returned."""
    solver = InverseProblemSolver(initial_pitch=1.5, initial_depth=0.2)
    variables = solver.get_inverse_variables()
    
    assert isinstance(variables, list)
    assert len(variables) == 2
    
    # Check that they are backend tensors with gradients enabled
    import torch
    assert isinstance(variables[0], torch.Tensor)
    assert variables[0].requires_grad
    assert isinstance(variables[1], torch.Tensor)
    assert variables[1].requires_grad

    # Check initial values
    assert np.isclose(variables[0].item(), 1.5)
    assert np.isclose(variables[1].item(), 0.2)

def test_define_observation_bc():
    """Tests that the observation BC is created correctly."""
    solver = InverseProblemSolver(initial_pitch=1.5, initial_depth=0.2)
    
    # Mock observation data
    # 10 points in time at x=0, y=1.0
    obs_points = np.zeros((10, 3))
    obs_points[:, 2] = np.linspace(0, 1, 10) # Time
    obs_points[:, 0] = 0.0  # x_coord
    obs_points[:, 1] = 1.0  # y_coord
    
    obs_values = np.sin(obs_points[:, 2:3] * np.pi) # Some sine wave
    
    # The BC should be applied to a specific component of the output, let's say stress_xx
    # which is not a direct output. The PointSetBC will be on a direct output, e.g. `u`.
    # Let's assume for the test it's on the first component 'u'.
    bc = solver.define_observation_bc(obs_points, obs_values, component=0)
    
    assert isinstance(bc, dde.icbc.PointSetBC)
    assert np.allclose(bc.points, obs_points)
    
    # Move tensor to CPU for numpy comparison
    bc_values_numpy = bc.values.cpu().numpy()
    assert np.allclose(bc_values_numpy, obs_values)
    assert bc.component == 0
