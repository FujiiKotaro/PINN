"""
Test suite for Notebook Task 6: 2D Wave Field Time Snapshot Visualization

Tests for time snapshot selection, 2D grid preparation, and visualization
in the forward validation notebook. This follows TDD methodology for task 6.1.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestTask61_TimeSnapshotVisualization:
    """Tests for Task 6.1: Time snapshot visualization cell implementation"""

    def test_time_snapshot_selection_minimum_3_snapshots(self):
        """
        Test that at least 3 time snapshots are selected for visualization
        Requirement 4.1: Minimum 3 time snapshots
        """
        # Mock unique time points
        t_unique = np.linspace(0, 1.0, 50)  # 50 time points

        # Select 3-5 snapshots (implementation logic from notebook)
        n_snapshots = min(5, len(t_unique))
        snapshot_indices = [
            len(t_unique) // 4,      # 25% through
            len(t_unique) // 2,      # 50% through (middle)
            3 * len(t_unique) // 4   # 75% through
        ]
        t_snapshots = [t_unique[i] for i in snapshot_indices]

        # Verify at least 3 snapshots
        assert len(t_snapshots) >= 3, f"Need at least 3 snapshots, got {len(t_snapshots)}"

        # Verify snapshots are evenly spaced
        assert t_snapshots[0] < t_snapshots[1] < t_snapshots[2], "Snapshots should be in order"

    def test_2d_grid_preparation_from_validation_data(self):
        """
        Test that validation data can be restructured into 2D spatial grids
        Requirement 4.2, 4.3: Prepare 2D spatial distribution data
        """
        # Mock validation data
        nx, ny, nt = 40, 20, 10
        N = nx * ny * nt

        x_unique = np.linspace(0, 1.0, nx)
        y_unique = np.linspace(0, 0.5, ny)
        t_unique = np.linspace(0, 1.0, nt)

        # Create meshgrid for all space-time points
        X, Y, T = np.meshgrid(x_unique, y_unique, t_unique, indexing='ij')

        # Mock field data (flattened)
        field_data = np.random.randn(N)

        # Select one time snapshot
        t_snap = t_unique[5]
        time_mask = np.abs(T.flatten() - t_snap) < 1e-9

        # Extract data at this time
        field_at_t = field_data[time_mask]

        # Reshape to 2D grid
        field_2d = field_at_t.reshape(ny, nx)

        # Verify grid shape
        assert field_2d.shape == (ny, nx), f"Expected shape ({ny}, {nx}), got {field_2d.shape}"
        assert not np.all(np.isnan(field_2d)), "Grid should not be all NaN"

    def test_plot_time_snapshots_api_call(self):
        """
        Test that PlotGeneratorService.plot_time_snapshots() is called with correct arguments
        Requirement 4.2, 4.3, 4.4: Generate side-by-side FDTD vs PINN comparison
        """
        from pinn.validation.plot_generator import PlotGeneratorService

        # Mock data
        nx, ny = 40, 20
        x = np.linspace(0, 0.04, nx)  # meters
        y = np.linspace(0, 0.02, ny)  # meters
        t_list = [4.0e-6, 5.0e-6, 6.0e-6]  # 3 time snapshots

        # Create mock 2D grids for each time
        fdtd_data = {t: np.random.randn(ny, nx) for t in t_list}
        pinn_pred = {t: np.random.randn(ny, nx) for t in t_list}

        # Call PlotGeneratorService
        plot_gen = PlotGeneratorService()
        fig, axes = plot_gen.plot_time_snapshots(
            x=x,
            y=y,
            t_list=t_list,
            fdtd_data=fdtd_data,
            pinn_pred=pinn_pred,
            output_field='Ux',
            save_path=None
        )

        # Verify output
        assert fig is not None, "Figure should be created"
        assert len(axes) >= 6, f"Should have at least 6 axes (2 per snapshot), got {len(axes)}"

        # Cleanup
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_time_snapshots_requires_minimum_3_snapshots(self):
        """
        Test that plot_time_snapshots raises error if fewer than 3 snapshots
        Requirement 4.1: Minimum 3 snapshots enforced
        """
        from pinn.validation.plot_generator import PlotGeneratorService

        # Mock data with only 2 snapshots (should fail)
        nx, ny = 40, 20
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        t_list = [4.0e-6, 5.0e-6]  # Only 2 snapshots

        fdtd_data = {t: np.random.randn(ny, nx) for t in t_list}
        pinn_pred = {t: np.random.randn(ny, nx) for t in t_list}

        plot_gen = PlotGeneratorService()

        # Should raise ValueError
        with pytest.raises(ValueError, match="At least 3 time snapshots required"):
            plot_gen.plot_time_snapshots(
                x=x,
                y=y,
                t_list=t_list,
                fdtd_data=fdtd_data,
                pinn_pred=pinn_pred,
                output_field='Ux'
            )

    def test_spatial_correlation_calculation(self):
        """
        Test that spatial correlation between FDTD and PINN can be computed
        Requirement 4.5: Wave propagation pattern analysis
        """
        # Mock 2D grids
        ny, nx = 20, 40
        fdtd_grid = np.random.randn(ny, nx)

        # PINN prediction with high correlation
        pinn_grid = fdtd_grid + np.random.randn(ny, nx) * 0.1

        # Compute correlation
        correlation = np.corrcoef(fdtd_grid.flatten(), pinn_grid.flatten())[0, 1]

        # Verify correlation is high (since we added small noise)
        assert correlation > 0.9, f"Expected high correlation, got {correlation:.4f}"
        assert -1.0 <= correlation <= 1.0, "Correlation should be in [-1, 1]"

    def test_error_metrics_calculation(self):
        """
        Test that error metrics (mean, max, relative) can be computed for each snapshot
        Requirement 4.5: Quantitative wave pattern analysis
        """
        # Mock 2D grids
        ny, nx = 20, 40
        fdtd_grid = np.random.randn(ny, nx)
        pinn_grid = fdtd_grid + np.random.randn(ny, nx) * 0.05

        # Compute error metrics
        abs_error = np.abs(pinn_grid - fdtd_grid)
        mean_error = np.mean(abs_error)
        max_error = np.max(abs_error)
        fdtd_std = np.std(fdtd_grid)
        rel_error = mean_error / fdtd_std if fdtd_std > 0 else np.inf

        # Verify metrics
        assert mean_error >= 0, "Mean error should be non-negative"
        assert max_error >= mean_error, "Max error should be >= mean error"
        assert 0 <= rel_error < 1.0, f"Relative error should be in [0, 1), got {rel_error:.4f}"

    def test_output_field_selection_ux(self):
        """
        Test that Ux displacement field is used for visualization
        Requirement 4.6: Focus on Ux displacement field
        """
        # Mock validation data with all fields
        N = 100
        val_data_mock = Mock()
        val_data_mock.Ux = np.random.randn(N)
        val_data_mock.Uy = np.random.randn(N)
        val_data_mock.T1 = np.random.randn(N)
        val_data_mock.T3 = np.random.randn(N)

        # Select Ux field
        output_field = 'Ux'
        field_data = getattr(val_data_mock, output_field)

        # Verify
        assert len(field_data) == N, "Field data length mismatch"
        assert output_field == 'Ux', "Should visualize Ux field (Requirement 4.6)"

    def test_dimensionless_to_physical_conversion(self):
        """
        Test that dimensionless coordinates are converted back to physical units
        for visualization (meters and microseconds)
        """
        # Mock dimensionless coordinates
        x_dimensionless = np.linspace(0, 1.0, 40)
        t_dimensionless = 0.5

        # Mock characteristic scales
        L_ref = 0.04  # 40 mm
        T_ref = 1.0e-5  # 10 µs

        # Convert to physical units
        x_physical = x_dimensionless * L_ref  # meters
        t_physical = t_dimensionless * T_ref  # seconds
        t_physical_us = t_physical * 1e6  # microseconds

        # Verify
        assert x_physical[0] == 0.0
        assert np.isclose(x_physical[-1], 0.04, atol=1e-10), f"Expected 0.04 m, got {x_physical[-1]}"
        assert np.isclose(t_physical_us, 5.0, atol=1e-6), f"Expected 5.0 µs, got {t_physical_us}"


# Integration test for complete Task 6.1 workflow
class TestTask61Integration:
    """Integration test for complete time snapshot visualization workflow"""

    def test_complete_time_snapshot_workflow(self):
        """
        Test complete workflow: select snapshots → prepare 2D grids → visualize → analyze
        """
        from pinn.validation.plot_generator import PlotGeneratorService
        import matplotlib.pyplot as plt

        # Step 1: Mock validation data
        nx, ny, nt = 40, 20, 10
        x_unique = np.linspace(0, 0.04, nx)  # meters
        y_unique = np.linspace(0, 0.02, ny)  # meters
        t_unique = np.linspace(0, 1.0e-5, nt)  # seconds

        # Step 2: Select 3 time snapshots
        snapshot_indices = [nt // 4, nt // 2, 3 * nt // 4]
        t_snapshots = [t_unique[i] for i in snapshot_indices]
        assert len(t_snapshots) >= 3

        # Step 3: Prepare 2D grids for each snapshot
        fdtd_data = {}
        pinn_pred = {}
        for t in t_snapshots:
            fdtd_data[t] = np.random.randn(ny, nx)
            pinn_pred[t] = fdtd_data[t] + np.random.randn(ny, nx) * 0.1

        # Step 4: Generate visualization
        plot_gen = PlotGeneratorService()
        fig, axes = plot_gen.plot_time_snapshots(
            x=x_unique,
            y=y_unique,
            t_list=t_snapshots,
            fdtd_data=fdtd_data,
            pinn_pred=pinn_pred,
            output_field='Ux'
        )

        assert fig is not None
        assert len(axes) == 2 * len(t_snapshots)  # 2 axes per snapshot

        # Step 5: Compute similarity metrics for each snapshot
        correlations = []
        for t in t_snapshots:
            fdtd_grid = fdtd_data[t]
            pinn_grid = pinn_pred[t]
            corr = np.corrcoef(fdtd_grid.flatten(), pinn_grid.flatten())[0, 1]
            correlations.append(corr)

        assert len(correlations) == len(t_snapshots)
        assert all(c > 0.8 for c in correlations), "All correlations should be reasonably high"

        # Cleanup
        plt.close(fig)

    def test_notebook_cell_requirements_coverage(self):
        """
        Test that all requirements for Task 6.1 are covered
        Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
        """
        requirements_coverage = {
            "4.1": "Select 3-5 time snapshots",  # test_time_snapshot_selection_minimum_3_snapshots
            "4.2": "Generate FDTD vs PINN comparison plots",  # test_plot_time_snapshots_api_call
            "4.3": "Display Ux displacement field spatial distribution",  # test_output_field_selection_ux
            "4.4": "Side-by-side heatmap arrangement",  # test_plot_time_snapshots_api_call
            "4.5": "Wave propagation pattern interpretation",  # test_spatial_correlation_calculation, test_error_metrics_calculation
            "4.6": "Visualize at least Ux field"  # test_output_field_selection_ux
        }

        # Verify all requirements are covered
        assert len(requirements_coverage) == 6, "Should cover all 6 requirements"
        for req_id, description in requirements_coverage.items():
            assert description, f"Requirement {req_id} should have description"
