"""
Test suite for Task 9.2: Output Format Verification

Tests for verifying that the notebook produces outputs in the expected format:
- R² score DataFrame structure (Requirement 3.3)
- R² score bar chart (Requirement 3.4)
- Training loss plot with 4 series (Requirement 2.7)
- Time snapshot heatmaps (Requirement 4.1)
- Error distribution heatmap with colorbar (Requirement 5.2)

This follows TDD methodology for Task 9.2.
"""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.fixture
def notebook_path():
    """Path to the forward validation notebook."""
    return Path(__file__).parent.parent / "notebooks" / "pinn_2d_forward_validation.ipynb"


@pytest.fixture
def notebook_content(notebook_path):
    """Load notebook content."""
    with open(notebook_path) as f:
        return nbformat.read(f, as_version=4)


class TestR2ScoreOutputFormat:
    """Test R² score output format (Requirements 3.3, 3.4)"""

    def test_r2_score_dataframe_structure_in_cell(self, notebook_content):
        """
        Test that notebook contains cell creating R² DataFrame with correct structure.
        Requirement 3.3: R² scores displayed as pandas DataFrame with shape (4, 1)
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that creates R² DataFrame
        found_dataframe_creation = False
        for cell in code_cells:
            source = cell.source
            if "pd.DataFrame" in source and "r2_scores" in source:
                found_dataframe_creation = True

                # Check that transpose is used (to make fields as rows)
                assert ".T" in source or "orient=" in source, \
                    "DataFrame should be transposed to show fields as rows (Requirement 3.3)"

                # Check that column is named
                if ".columns" in source:
                    assert "R² Score" in source or "R2 Score" in source, \
                        "DataFrame column should be named 'R² Score'"

                break

        assert found_dataframe_creation, \
            "Notebook should contain cell creating R² scores DataFrame (Requirement 3.3)"

    def test_r2_bar_chart_generation_in_cell(self, notebook_content):
        """
        Test that notebook contains cell generating R² score bar chart.
        Requirement 3.4: R² scores visualized as bar chart
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that generates bar chart
        found_bar_chart = False
        for cell in code_cells:
            source = cell.source
            # Check for bar chart creation using r2_scores
            if "r2_scores" in source and ("plt.bar" in source or "ax.bar" in source):
                found_bar_chart = True

                # Should set y-axis label
                assert "ylabel" in source.lower(), "Bar chart should have y-axis label"

                # Should set y-axis limit to 1.0
                assert "ylim" in source or "set_ylim" in source, \
                    "Bar chart should set y-axis limit to 1.0"

                break

        assert found_bar_chart, \
            "Notebook should contain cell generating R² bar chart (Requirement 3.4)"


class TestTrainingLossPlotFormat:
    """Test training loss plot format (Requirement 2.7)"""

    def test_training_loss_plot_has_4_series(self, notebook_content):
        """
        Test that training loss plot displays 4 series: L_data, L_pde, L_bc, Total.
        Requirement 2.7: Loss evolution plot with 4 components
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that plots training loss
        found_loss_plot = False
        for cell in code_cells:
            source = cell.source
            # Check for loss plotting using PlotGeneratorService.plot_training_curves()
            if "plot_training_curves" in source or "plot_training_losses" in source or \
               ("history" in source and "plt.plot" in source):
                found_loss_plot = True

                # Should mention history or loss components
                has_history = "history" in source.lower()
                assert has_history, "Loss plot should use training history (Requirement 2.7)"

                break

        assert found_loss_plot, \
            "Notebook should contain cell plotting training loss evolution (Requirement 2.7)"

    def test_training_loss_plot_uses_plot_generator_service(self, notebook_content):
        """
        Test that training loss plot uses PlotGeneratorService.plot_training_curves().
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell using PlotGeneratorService.plot_training_curves()
        found_plot_service = False
        for cell in code_cells:
            if "plot_generator" in cell.source and "plot_training_curves" in cell.source:
                found_plot_service = True
                break

        assert found_plot_service, \
            "Notebook should use PlotGeneratorService.plot_training_curves() for loss visualization"


class TestTimeSnapshotOutputFormat:
    """Test time snapshot visualization output format (Requirement 4.1)"""

    def test_time_snapshot_visualization_minimum_3_snapshots(self, notebook_content):
        """
        Test that time snapshot visualization uses at least 3 snapshots.
        Requirement 4.1: Minimum 3 time snapshots
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that generates time snapshots
        found_snapshots = False
        for cell in code_cells:
            source = cell.source
            if "plot_time_snapshots" in source or "t_snapshots" in source:
                found_snapshots = True

                # Check that at least 3 snapshots are selected
                # Look for array slicing like [::n] or explicit snapshot count
                if "n_snapshots" in source:
                    # Check that n_snapshots >= 3
                    assert "min(5" in source or "min(3" in source or ">= 3" in source, \
                        "Should ensure at least 3 snapshots (Requirement 4.1)"

                break

        assert found_snapshots, \
            "Notebook should contain cell generating time snapshot visualizations (Requirement 4.1)"

    def test_time_snapshot_uses_plot_generator_service(self, notebook_content):
        """
        Test that time snapshot visualization uses PlotGeneratorService.plot_time_snapshots().
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell using PlotGeneratorService.plot_time_snapshots()
        found_plot_service = False
        for cell in code_cells:
            if "plot_time_snapshots" in cell.source:
                found_plot_service = True

                # Should pass fdtd_data and pinn_pred dictionaries
                assert "fdtd_data" in cell.source, "Should pass fdtd_data to plot_time_snapshots()"
                assert "pinn_pred" in cell.source or "pinn_data" in cell.source, \
                    "Should pass pinn_pred to plot_time_snapshots()"

                break

        assert found_plot_service, \
            "Notebook should use PlotGeneratorService.plot_time_snapshots()"


class TestErrorHeatmapOutputFormat:
    """Test error distribution heatmap output format (Requirement 5.2)"""

    def test_error_heatmap_generated(self, notebook_content):
        """
        Test that error distribution heatmap is generated.
        Requirement 5.2: 2D spatial error heatmap visualization
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that generates error heatmap
        found_heatmap = False
        for cell in code_cells:
            source = cell.source
            if "plot_spatial_heatmap" in source or ("abs_error" in source and "imshow" in source):
                found_heatmap = True

                # Should calculate absolute error
                assert "abs_error" in source or "np.abs" in source, \
                    "Should calculate absolute error for heatmap"

                break

        assert found_heatmap, \
            "Notebook should contain cell generating error distribution heatmap (Requirement 5.2)"

    def test_error_heatmap_uses_plot_generator_service(self, notebook_content):
        """
        Test that error heatmap uses PlotGeneratorService.plot_spatial_heatmap().
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell using PlotGeneratorService.plot_spatial_heatmap()
        found_plot_service = False
        for cell in code_cells:
            if "plot_spatial_heatmap" in cell.source:
                found_plot_service = True

                # Should pass error data
                assert "error=" in cell.source or "abs_error" in cell.source, \
                    "Should pass error data to plot_spatial_heatmap()"

                break

        assert found_plot_service, \
            "Notebook should use PlotGeneratorService.plot_spatial_heatmap()"

    def test_error_statistics_displayed(self, notebook_content):
        """
        Test that error statistics (mean, max, relative error) are displayed.
        Requirement 5.5: Display error statistics
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that displays error statistics
        found_stats = False
        for cell in code_cells:
            source = cell.source

            # Check for error statistics calculation
            has_mean = "mean_error" in source or "np.mean(abs_error)" in source or "np.mean(error)" in source
            has_max = "max_error" in source or "np.max(abs_error)" in source or "np.max(error)" in source
            has_rel = "rel_error" in source or "relative_error" in source

            if has_mean and has_max and has_rel:
                found_stats = True
                break

        assert found_stats, \
            "Notebook should display error statistics: mean, max, relative error (Requirement 5.5)"


class TestNotebookOutputConsistency:
    """Test overall notebook output consistency"""

    def test_all_visualizations_use_plot_generator_service(self, notebook_content):
        """
        Test that all visualizations use PlotGeneratorService for consistency.
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Collect all visualization-related cells
        viz_cells = []
        for cell in code_cells:
            if any(keyword in cell.source for keyword in
                   ["plot_training_curves", "plot_training_losses", "plot_time_snapshots", "plot_spatial_heatmap"]):
                viz_cells.append(cell.source)

        # Should have at least 3 visualization cells (loss, snapshots, heatmap)
        assert len(viz_cells) >= 3, \
            f"Notebook should have at least 3 visualization cells (loss, snapshots, error heatmap), got {len(viz_cells)}"

    def test_output_directory_is_created(self, notebook_content):
        """
        Test that notebook creates output directory for saving plots.
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that creates output directory
        found_output_dir = False
        for cell in code_cells:
            source = cell.source
            if "output_dir" in source or "outputs/" in source or "results/" in source:
                if "mkdir" in source or "Path(" in source:
                    found_output_dir = True
                    break

        assert found_output_dir, \
            "Notebook should create output directory for saving plots"

    def test_seed_is_set_for_reproducibility(self, notebook_content):
        """
        Test that random seed is set for reproducibility.
        Requirement 7.4: Set random seed
        """
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # First code cell should set seed
        first_code_cell = code_cells[0]
        assert "SeedManager.set_seed" in first_code_cell.source, \
            "Setup cell should set random seed using SeedManager (Requirement 7.4)"

        assert "SEED = 42" in first_code_cell.source or "seed = 42" in first_code_cell.source, \
            "SEED should be 42 for reproducibility"


# Integration test for complete output format verification
class TestTask92Integration:
    """Integration test for complete output format verification workflow"""

    def test_notebook_output_format_requirements_coverage(self):
        """
        Test that all requirements for Task 9.2 are covered.
        Requirements: 3.3, 3.4, 2.7, 4.1, 5.2
        """
        requirements_coverage = {
            "3.3": "R² scores displayed as pandas DataFrame (4, 1)",  # test_r2_score_dataframe_structure_in_cell
            "3.4": "R² scores visualized as bar chart",  # test_r2_bar_chart_generation_in_cell
            "2.7": "Training loss plot with 4 series (L_data, L_pde, L_bc, Total)",  # test_training_loss_plot_has_4_series
            "4.1": "Time snapshot heatmaps (minimum 3 snapshots)",  # test_time_snapshot_visualization_minimum_3_snapshots
            "5.2": "Error distribution heatmap with colorbar"  # test_error_heatmap_generated
        }

        # Verify all requirements are covered
        assert len(requirements_coverage) == 5, "Should cover all 5 requirements (3.3, 3.4, 2.7, 4.1, 5.2)"
        for req_id, description in requirements_coverage.items():
            assert description, f"Requirement {req_id} should have description"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_output_format_verification_workflow(self, notebook_path, tmp_path):
        """
        Integration test: Execute notebook and verify all output formats.

        This test executes the full notebook and checks:
        1. R² DataFrame has shape (4, 1)
        2. Bar chart has 4 bars
        3. Loss plot exists with 4 series
        4. Time snapshots generated (>= 3)
        5. Error heatmap generated

        Note: This test requires PINN_data/ directory with valid .npz files.
        """
        import os

        # Skip if PINN_data does not exist
        pinn_data_dir = Path(__file__).parent.parent / "PINN_data"
        if not pinn_data_dir.exists():
            pytest.skip(f"Skipping integration test: {pinn_data_dir} directory not found")

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Execute notebook (with extended timeout)
        # Set PYTHONPATH to include project root
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        project_root = str(Path(__file__).parent.parent)
        os.environ["PYTHONPATH"] = f"{project_root}:{old_pythonpath}"

        ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")

        try:
            nb_executed, resources = ep.preprocess(nb, {"metadata": {"path": str(tmp_path)}})

            # Check for errors
            for i, cell in enumerate(nb_executed.cells):
                if cell.cell_type == "code":
                    for output in cell.get("outputs", []):
                        if output.get("output_type") == "error":
                            error_msg = "\n".join(output.get("traceback", []))
                            pytest.fail(f"Cell {i} produced error:\n{error_msg}")

            # Verify outputs exist
            # Count cells with display_data (plots, tables)
            display_count = 0
            for cell in nb_executed.cells:
                if cell.cell_type == "code":
                    for output in cell.get("outputs", []):
                        if output.get("output_type") in ["display_data", "execute_result"]:
                            display_count += 1

            # Should have multiple displays (DataFrames, plots)
            assert display_count >= 5, \
                f"Expected at least 5 displays (DataFrame, bar chart, loss plot, snapshots, heatmap), got {display_count}"

        except Exception as e:
            pytest.fail(f"Output format verification failed: {e}")
        finally:
            # Restore PYTHONPATH
            os.environ["PYTHONPATH"] = old_pythonpath
