import deepxde as dde
from typing import Dict, List
import numpy as np

def is_on_rough_surface(x, params):
    """
    Checks if a point x lies on the rough surface boundary.
    x is a numpy array with shape (..., 3) where the last dimension is (x, y, t).
    """
    x_coord, y_coord = x[..., 0], x[..., 1]
    
    x_max = params['x_domain'][1]
    y_min = params['y_domain'][0]
    y_max = params['y_domain'][1]
    f_depth = params['f_depth']
    f_pitch = params['f_pitch']
    f_width = params['f_width']

    # Check if the point is on the side walls, which are not part of the rough surface BC.
    on_y_side_walls = np.isclose(y_coord, y_min) | np.isclose(y_coord, y_max)
    
    y_in_period = y_coord % f_pitch
    flat_width = f_pitch - f_width

    # On the flat bottom surface
    on_flat = np.isclose(x_coord, x_max) & (y_in_period <= flat_width)
    
    # On the vertical wall starting a notch
    on_wall1 = np.isclose(y_in_period, flat_width) & (x_coord >= (x_max - f_depth)) & (x_coord <= x_max)

    # On the bottom of the notch
    on_notch_bottom = np.isclose(x_coord, x_max - f_depth) & (y_in_period > flat_width)
    
    # On the vertical wall ending a notch
    # This wall is at y positions that are multiples of f_pitch.
    # We must exclude the y_min and y_max boundaries of the whole domain.
    on_wall2 = np.isclose(y_in_period, 0) & (x_coord >= (x_max - f_depth)) & (x_coord <= x_max)

    # The final condition should not include the y-side walls
    return (on_flat | on_wall1 | on_notch_bottom | on_wall2) & ~on_y_side_walls


class PINNModelBuilder:
    """
    Constructs the `deepxde` forward model (geometry, PDE, BCs).
    """
    def __init__(self, dimensionless_params: Dict):
        """
        Initializes the builder with dimensionless parameters.

        Args:
            dimensionless_params: A dictionary containing the domain boundaries.
                                  e.g., {'x_domain': [0, 1], 'y_domain': [0, 2], 't_domain': [0, 1]}
        """
        self.params = dimensionless_params
        self._geom = None

    def get_geometry(self) -> dde.geometry.GeometryXTime:
        """
        Creates and returns the spatio-temporal geometry for the problem.
        """
        if self._geom is None:
            space_domain = dde.geometry.Rectangle(
                xmin=[self.params['x_domain'][0], self.params['y_domain'][0]],
                xmax=[self.params['x_domain'][1], self.params['y_domain'][1]]
            )
            time_domain = dde.geometry.TimeDomain(
                self.params['t_domain'][0], self.params['t_domain'][1]
            )
            self._geom = dde.geometry.GeometryXTime(space_domain, time_domain)
        return self._geom

    def define_pde_system(self, x, y) -> List:
        """
        Defines the system of PDEs for the 2D elastic wave equation.
        The network output y has two components, y = (u, v), representing
        displacements in x and y directions respectively.
        The input x has three components, x = (x, y, t).
        """
        u = y[:, 0:1]
        v = y[:, 1:2]
        
        # Second derivatives with respect to time
        u_tt = dde.grad.hessian(y, x, component=0, i=2, j=2)
        v_tt = dde.grad.hessian(y, x, component=1, i=2, j=2)
        
        # Second derivatives with respect to space
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

        # Mixed derivatives
        u_xy = dde.grad.hessian(y, x, component=0, i=0, j=1)
        v_xy = dde.grad.hessian(y, x, component=1, i=0, j=1)
        
        # Dimensionless constants from params
        rho = self.params['rho']
        c11 = self.params['c11']
        c13 = self.params['c13']
        c55 = self.params['c55']

        # PDE residuals
        pde1 = rho * u_tt - (c11 * u_xx + c55 * u_yy + (c13 + c55) * v_xy)
        pde2 = rho * v_tt - (c55 * v_xx + c11 * v_yy + (c13 + c55) * u_xy)
        
        return [pde1, pde2]

    def define_boundary_conditions(self, source_func=None):
        """
        Defines the boundary conditions for the problem.
        Implements stress-free BCs on the sides and the rough bottom surface.
        Optionally adds a source wave boundary condition at the top.
        """
        geom = self.get_geometry()

        def stress_xx(x, y, _):
            u_x = dde.grad.jacobian(y, x, i=0, j=0)
            v_y = dde.grad.jacobian(y, x, i=1, j=1)
            return self.params['c11'] * u_x + self.params['c13'] * v_y
        
        def stress_yy(x, y, _):
            u_x = dde.grad.jacobian(y, x, i=0, j=0)
            v_y = dde.grad.jacobian(y, x, i=1, j=1)
            return self.params['c13'] * u_x + self.params['c11'] * v_y

        def stress_xy(x, y, _):
            u_y = dde.grad.jacobian(y, x, i=0, j=1)
            v_x = dde.grad.jacobian(y, x, i=1, j=0)
            return self.params['c55'] * (u_y + v_x)

        # Boundary definitions
        def on_side_y0(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.params['y_domain'][0])

        def on_side_y_max(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.params['y_domain'][1])

        def on_rough_bottom(x, on_boundary):
            # This helper function needs access to self.params, so we wrap it
            return is_on_rough_surface(x, self.params)

        bcs = []

        # BCs for side y=0 (t_x = sigma_xy = 0, t_y = sigma_yy = 0)
        bcs.append(dde.icbc.OperatorBC(geom, stress_xy, on_side_y0))
        bcs.append(dde.icbc.OperatorBC(geom, stress_yy, on_side_y0))

        # BCs for side y=max (t_x = sigma_xy = 0, t_y = sigma_yy = 0)
        bcs.append(dde.icbc.OperatorBC(geom, stress_xy, on_side_y_max))
        bcs.append(dde.icbc.OperatorBC(geom, stress_yy, on_side_y_max))

        # BCs for rough bottom boundary
        bcs.append(dde.icbc.OperatorBC(geom, stress_xx, on_rough_bottom))
        bcs.append(dde.icbc.OperatorBC(geom, stress_xy, on_rough_bottom))

        if source_func is not None:
            # Source wave at the top boundary (x=0)
            def on_source(x, on_boundary):
                return on_boundary and np.isclose(x[0], self.params['x_domain'][0])
            
            # This is a time-varying Dirichlet BC on stress_xx.
            # We can implement this with an OperatorBC where the operator is `stress_xx - source_func`.
            def source_operator(x, y, _):
                # x[:, 2:3] is time
                return stress_xx(x, y, _) - source_func(x[:, 2:3])

            bc_source = dde.icbc.OperatorBC(geom, source_operator, on_source)
            bcs.append(bc_source)
        
        return bcs

class InverseProblemSolver:
    """
    Configures the `deepxde` inverse problem.
    """
    def __init__(self, initial_pitch: float, initial_depth: float):
        """
        Initializes the solver with initial guesses for the inverse parameters.
        
        Args:
            initial_pitch: Initial guess for the pitch.
            initial_depth: Initial guess for the depth.
        """
        self.pitch = dde.Variable(initial_pitch)
        self.depth = dde.Variable(initial_depth)

    def get_inverse_variables(self) -> List[dde.Variable]:
        """
        Returns the list of inverse variables (pitch and depth).
        """
        return [self.pitch, self.depth]

    def define_observation_bc(self, points: np.ndarray, values: np.ndarray, component: int) -> dde.icbc.PointSetBC:
        """
        Defines the PointSetBC for the observation data.

        Args:
            points: The coordinates where the data was observed.
            values: The observed values at those points.
            component: The component of the network output this observation corresponds to.

        Returns:
            A PointSetBC object.
        """
        return dde.icbc.PointSetBC(points, values, component=component)

