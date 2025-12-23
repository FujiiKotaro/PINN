"""Tests for 2D plot generation (Tasks 4.3 and 4.4).

Test-Driven Development: Tests written before implementation.
Tests cover 2D spatial heatmap visualization for PINN-FDTD comparison.
"""

import numpy as np
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile

from pinn.validation.plot_generator import PlotGeneratorService


class TestPlotTimeSnapshots:
    """Test time-series snapshot visualization (Task 4.3)."""

    def test_plot_time_snapshots_basic(self):
        """Test basic time snapshot plotting with 3 time points."""
        plot_generator = PlotGeneratorService()

        # Create synthetic 2D data
        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        X, Y = np.meshgrid(x, y)

        # Create 3 time snapshots
        t_list = [4.0e-6, 5.0e-6, 6.0e-6]

        # FDTD data (ground truth)
        fdtd_data = {}
        for t in t_list:
            fdtd_data[t] = np.sin(2 * np.pi * X / 0.04) * np.cos(2 * np.pi * t / 1e-6)

        # PINN predictions (with small error)
        pinn_pred = {}
        for t in t_list:
            pinn_pred[t] = fdtd_data[t] + 0.01 * np.random.randn(*fdtd_data[t].shape)

        # Create plot
        fig, axes = plot_generator.plot_time_snapshots(
            x=x,
            y=y,
            t_list=t_list,
            fdtd_data=fdtd_data,
            pinn_pred=pinn_pred,
            output_field='Ux'
        )

        # Verify figure and axes created
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, (list, np.ndarray))
        assert len(axes) >= 3  # At least 3 time snapshots

        plt.close(fig)

    def test_plot_time_snapshots_saves_to_file(self):
        """Test that plot is saved to file when save_path provided."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        X, Y = np.meshgrid(x, y)

        t_list = [4.0e-6, 5.0e-6, 6.0e-6]

        fdtd_data = {}
        pinn_pred = {}
        for t in t_list:
            data = np.sin(2 * np.pi * X / 0.04)
            fdtd_data[t] = data
            pinn_pred[t] = data

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = Path(tmp.name)

        try:
            fig, axes = plot_generator.plot_time_snapshots(
                x=x,
                y=y,
                t_list=t_list,
                fdtd_data=fdtd_data,
                pinn_pred=pinn_pred,
                output_field='Ux',
                save_path=save_path
            )

            # Check file was created
            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)
        finally:
            # Cleanup
            if save_path.exists():
                save_path.unlink()

    def test_plot_time_snapshots_minimum_three_snapshots(self):
        """Test that at least 3 snapshots are required (Requirement 7.4)."""
        plot_generator = PlotGeneratorService()

        nx, ny = 10, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        X, Y = np.meshgrid(x, y)

        # Only 2 time points (should raise error or warning)
        t_list = [4.0e-6, 5.0e-6]

        fdtd_data = {}
        pinn_pred = {}
        for t in t_list:
            data = np.random.randn(ny, nx)
            fdtd_data[t] = data
            pinn_pred[t] = data

        # Should raise ValueError for insufficient snapshots
        with pytest.raises(ValueError, match="At least 3"):
            plot_generator.plot_time_snapshots(
                x=x,
                y=y,
                t_list=t_list,
                fdtd_data=fdtd_data,
                pinn_pred=pinn_pred,
                output_field='Ux'
            )

    def test_plot_time_snapshots_multiple_fields(self):
        """Test plotting for different output fields (T1, T3, Ux, Uy)."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)

        t_list = [4.0e-6, 5.0e-6, 6.0e-6]

        fdtd_data = {}
        pinn_pred = {}
        for t in t_list:
            data = np.random.randn(ny, nx)
            fdtd_data[t] = data
            pinn_pred[t] = data

        # Test each output field
        for field in ['T1', 'T3', 'Ux', 'Uy']:
            fig, axes = plot_generator.plot_time_snapshots(
                x=x,
                y=y,
                t_list=t_list,
                fdtd_data=fdtd_data,
                pinn_pred=pinn_pred,
                output_field=field
            )

            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_time_snapshots_shape_consistency(self):
        """Test that data shapes are consistent across snapshots."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)

        t_list = [4.0e-6, 5.0e-6, 6.0e-6]

        # Inconsistent shapes (should raise error)
        fdtd_data = {
            t_list[0]: np.random.randn(ny, nx),
            t_list[1]: np.random.randn(ny, nx),
            t_list[2]: np.random.randn(ny + 1, nx)  # Wrong shape
        }
        pinn_pred = {t: np.random.randn(ny, nx) for t in t_list}

        with pytest.raises(ValueError, match="[Ss]hape"):
            plot_generator.plot_time_snapshots(
                x=x,
                y=y,
                t_list=t_list,
                fdtd_data=fdtd_data,
                pinn_pred=pinn_pred,
                output_field='Ux'
            )


class TestPlotSpatialHeatmap:
    """Test spatial error distribution heatmap (Task 4.4)."""

    def test_plot_spatial_heatmap_basic(self):
        """Test basic heatmap generation."""
        plot_generator = PlotGeneratorService()

        # Create 2D grid data
        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        X, Y = np.meshgrid(x, y)

        # Error field (absolute error)
        error = np.abs(np.sin(2 * np.pi * X / 0.04) * np.cos(2 * np.pi * Y / 0.02))

        # Create heatmap
        fig, ax = plot_generator.plot_spatial_heatmap(
            x=x,
            y=y,
            error=error,
            output_field='Ux'
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        plt.close(fig)

    def test_plot_spatial_heatmap_saves_to_file(self):
        """Test that heatmap is saved to file."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        X, Y = np.meshgrid(x, y)
        error = np.random.rand(ny, nx)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = Path(tmp.name)

        try:
            fig, ax = plot_generator.plot_spatial_heatmap(
                x=x,
                y=y,
                error=error,
                output_field='T1',
                save_path=save_path
            )

            assert save_path.exists()
            assert save_path.stat().st_size > 0

            plt.close(fig)
        finally:
            if save_path.exists():
                save_path.unlink()

    def test_plot_spatial_heatmap_identifies_error_regions(self):
        """Test that heatmap visualizes high error regions."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)

        # Create error with high region in center
        error = np.zeros((ny, nx))
        error[ny//2-2:ny//2+2, nx//2-2:nx//2+2] = 10.0  # High error region

        fig, ax = plot_generator.plot_spatial_heatmap(
            x=x,
            y=y,
            error=error,
            output_field='Uy'
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_spatial_heatmap_all_fields(self):
        """Test heatmap generation for all output fields."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        error = np.random.rand(ny, nx)

        for field in ['T1', 'T3', 'Ux', 'Uy']:
            fig, ax = plot_generator.plot_spatial_heatmap(
                x=x,
                y=y,
                error=error,
                output_field=field
            )

            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_spatial_heatmap_handles_zero_error(self):
        """Test heatmap with all zero errors (perfect prediction)."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        error = np.zeros((ny, nx))

        fig, ax = plot_generator.plot_spatial_heatmap(
            x=x,
            y=y,
            error=error,
            output_field='T3'
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_spatial_heatmap_shape_validation(self):
        """Test that error shape matches x, y coordinates."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)

        # Wrong shape error (should raise ValueError)
        error_wrong = np.random.rand(ny + 1, nx)

        with pytest.raises(ValueError, match="[Ss]hape"):
            plot_generator.plot_spatial_heatmap(
                x=x,
                y=y,
                error=error_wrong,
                output_field='Ux'
            )

    def test_plot_spatial_heatmap_uses_physical_units(self):
        """Test that heatmap axes use physical units (meters)."""
        plot_generator = PlotGeneratorService()

        nx, ny = 20, 10
        # Physical units (meters)
        x = np.linspace(0, 0.04, nx)
        y = np.linspace(0, 0.02, ny)
        error = np.random.rand(ny, nx)

        fig, ax = plot_generator.plot_spatial_heatmap(
            x=x,
            y=y,
            error=error,
            output_field='T1'
        )

        # Check that axes labels mention physical units
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # Should contain 'x' or 'm' for meters
        assert 'x' in xlabel.lower() or 'm' in xlabel.lower()
        assert 'y' in ylabel.lower() or 'm' in ylabel.lower()

        plt.close(fig)
