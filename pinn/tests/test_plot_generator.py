"""Unit tests for plot generator.

Tests verify:
1. Training curves generation from loss history
2. Solution comparison plots (PINN vs analytical)
3. File saving and plot formatting
4. Multiple time snapshots visualization
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Non-interactive backend for testing
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt


class TestPlotTrainingCurves:
    """Test training curves plotting."""

    def test_plot_created_successfully(self):
        """Test that plot_training_curves creates a figure."""
        from pinn.validation.plot_generator import PlotGeneratorService

        # Setup: mock training history
        history = {
            "total_loss": [1.0, 0.5, 0.3, 0.2, 0.1],
            "L_data": [0.5, 0.25, 0.15, 0.1, 0.05],
            "L_pde": [0.3, 0.15, 0.1, 0.07, 0.03],
            "L_bc": [0.2, 0.1, 0.05, 0.03, 0.02]
        }

        generator = PlotGeneratorService()

        # Execute
        fig, ax = generator.plot_training_curves(history)

        # Verify
        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 4  # 4 loss components

        plt.close(fig)

    def test_saves_to_file(self):
        """Test saving training curves to file."""
        from pinn.validation.plot_generator import PlotGeneratorService

        history = {
            "total_loss": [1.0, 0.5, 0.1],
            "L_data": [0.5, 0.25, 0.05],
            "L_pde": [0.3, 0.15, 0.03],
            "L_bc": [0.2, 0.1, 0.02]
        }

        generator = PlotGeneratorService()

        # Execute: save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "training_curves.png"
            fig, _ = generator.plot_training_curves(history, save_path=output_path)

            # Verify file created
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # Non-empty file

            plt.close(fig)

    def test_plot_labels_and_title(self):
        """Test plot has proper labels and title."""
        from pinn.validation.plot_generator import PlotGeneratorService

        history = {
            "total_loss": [1.0, 0.5],
            "L_data": [0.5, 0.25],
            "L_pde": [0.3, 0.15],
            "L_bc": [0.2, 0.1]
        }

        generator = PlotGeneratorService()
        fig, ax = generator.plot_training_curves(history)

        # Verify labels
        assert ax.get_xlabel().lower() == "epoch"
        assert "loss" in ax.get_ylabel().lower()
        assert ax.get_title() != ""  # Has a title

        # Verify legend exists
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4

        plt.close(fig)

    def test_log_scale_option(self):
        """Test log scale for y-axis."""
        from pinn.validation.plot_generator import PlotGeneratorService

        history = {
            "total_loss": [100, 10, 1, 0.1, 0.01],
            "L_data": [50, 5, 0.5, 0.05, 0.005],
            "L_pde": [30, 3, 0.3, 0.03, 0.003],
            "L_bc": [20, 2, 0.2, 0.02, 0.002]
        }

        generator = PlotGeneratorService()
        fig, ax = generator.plot_training_curves(history, log_scale=True)

        # Verify log scale
        assert ax.get_yscale() == 'log'

        plt.close(fig)


class TestPlotSolutionComparison:
    """Test solution comparison plotting."""

    def test_plot_comparison_at_time_snapshots(self):
        """Test plotting PINN vs analytical at multiple times."""
        from pinn.validation.plot_generator import PlotGeneratorService

        # Setup: mock solutions
        x = np.linspace(0, 1, 51)
        time_snapshots = [0.0, 0.25, 0.5, 0.75]

        u_pinn = {t: np.sin(np.pi * x) * np.cos(np.pi * t) for t in time_snapshots}
        u_analytical = {t: np.sin(np.pi * x) * np.cos(np.pi * t) for t in time_snapshots}

        generator = PlotGeneratorService()

        # Execute
        fig, axes = generator.plot_solution_comparison(
            x, time_snapshots, u_pinn, u_analytical
        )

        # Verify
        assert fig is not None
        assert len(axes) == 4  # 4 time snapshots
        for ax in axes:
            assert len(ax.lines) == 2  # PINN and analytical lines

        plt.close(fig)

    def test_saves_comparison_plot(self):
        """Test saving solution comparison to file."""
        from pinn.validation.plot_generator import PlotGeneratorService

        x = np.linspace(0, 1, 51)
        time_snapshots = [0.0, 0.5]

        u_pinn = {t: np.sin(np.pi * x) * t for t in time_snapshots}
        u_analytical = {t: np.sin(np.pi * x) * t for t in time_snapshots}

        generator = PlotGeneratorService()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "solution_comparison.png"
            fig, _ = generator.plot_solution_comparison(
                x, time_snapshots, u_pinn, u_analytical, save_path=output_path
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 0

            plt.close(fig)

    def test_subplot_layout(self):
        """Test proper subplot layout for multiple time snapshots."""
        from pinn.validation.plot_generator import PlotGeneratorService

        x = np.linspace(0, 1, 51)
        time_snapshots = [0.0, 0.25, 0.5]

        u_pinn = {t: np.zeros(len(x)) for t in time_snapshots}
        u_analytical = {t: np.zeros(len(x)) for t in time_snapshots}

        generator = PlotGeneratorService()
        fig, axes = generator.plot_solution_comparison(
            x, time_snapshots, u_pinn, u_analytical
        )

        # Verify subplot count
        assert len(axes) == 3

        # Verify each subplot has title indicating time
        for i, ax in enumerate(axes):
            title = ax.get_title()
            assert str(time_snapshots[i]) in title or "t=" in title.lower()

        plt.close(fig)

    def test_legend_labels(self):
        """Test legend contains PINN and analytical labels."""
        from pinn.validation.plot_generator import PlotGeneratorService

        x = np.linspace(0, 1, 11)
        time_snapshots = [0.0]

        u_pinn = {0.0: np.ones(len(x))}
        u_analytical = {0.0: np.ones(len(x))}

        generator = PlotGeneratorService()
        fig, axes = generator.plot_solution_comparison(
            x, time_snapshots, u_pinn, u_analytical
        )

        # Verify legend
        legend = axes[0].get_legend()
        assert legend is not None

        legend_texts = [t.get_text().lower() for t in legend.get_texts()]
        assert any("pinn" in text for text in legend_texts)
        assert any("analytical" in text or "exact" in text for text in legend_texts)

        plt.close(fig)


class TestPlotGeneratorIntegration:
    """Test plot generator with realistic scenarios."""

    def test_full_validation_workflow(self):
        """Test complete plotting workflow with training history and solutions."""
        from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
        from pinn.validation.plot_generator import PlotGeneratorService

        # Setup: realistic training history
        epochs = 100
        history = {
            "total_loss": np.exp(-0.05 * np.arange(epochs)),  # Exponential decay
            "L_data": np.exp(-0.05 * np.arange(epochs)) * 0.5,
            "L_pde": np.exp(-0.05 * np.arange(epochs)) * 0.3,
            "L_bc": np.exp(-0.05 * np.arange(epochs)) * 0.2,
        }

        # Setup: analytical vs PINN solutions
        analytical_gen = AnalyticalSolutionGeneratorService()
        x = np.linspace(0, 1, 51)
        time_snapshots = [0.0, 0.25, 0.5]

        u_analytical = {}
        u_pinn = {}
        for t in time_snapshots:
            u_exact = analytical_gen.standing_wave(
                x, np.array([t]), L=1.0, c=1.0, n=1
            )[:, 0]
            u_analytical[t] = u_exact
            # Simulate PINN with small error
            u_pinn[t] = u_exact + 0.01 * np.random.randn(len(x))

        plot_gen = PlotGeneratorService()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate both plots
            fig1, _ = plot_gen.plot_training_curves(
                history,
                save_path=Path(tmpdir) / "training.png"
            )
            fig2, _ = plot_gen.plot_solution_comparison(
                x, time_snapshots, u_pinn, u_analytical,
                save_path=Path(tmpdir) / "comparison.png"
            )

            # Verify both files created
            assert (Path(tmpdir) / "training.png").exists()
            assert (Path(tmpdir) / "comparison.png").exists()

            plt.close(fig1)
            plt.close(fig2)

    def test_handles_missing_loss_components(self):
        """Test plotting works even if some loss components missing."""
        from pinn.validation.plot_generator import PlotGeneratorService

        # Only total_loss and L_pde available
        history = {
            "total_loss": [1.0, 0.5, 0.1],
            "L_pde": [0.8, 0.4, 0.08],
        }

        generator = PlotGeneratorService()
        fig, ax = generator.plot_training_curves(history)

        # Should still plot available components
        assert fig is not None
        assert len(ax.lines) >= 2  # At least 2 lines

        plt.close(fig)
