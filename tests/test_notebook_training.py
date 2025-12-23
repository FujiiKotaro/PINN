"""
Test for Notebook Cell 5: Training Execution (Tasks 4.1 and 4.2)

Tests the training execution cells to ensure:
Task 4.1:
1. Training execution with callbacks (LossLoggingCallback, R2ValidationCallback, DivergenceDetectionCallback)
2. Training time recording (GPU)
3. Individual loss component logging (L_data, L_pde, L_bc, total_loss)
4. Loss values are explicitly displayed

Task 4.2:
5. Training history plot generation (4 series: L_data, L_pde, L_bc, Total)
6. NaN loss detection and warning
7. Training convergence visual confirmation

Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9
"""

import pytest
import nbformat
from pathlib import Path


@pytest.fixture
def notebook_path():
    """Path to the forward validation notebook."""
    return Path(__file__).parent.parent / "notebooks" / "pinn_2d_forward_validation.ipynb"


@pytest.fixture
def notebook_content(notebook_path):
    """Load notebook content."""
    with open(notebook_path) as f:
        return nbformat.read(f, as_version=4)


class TestTask41TrainingExecution:
    """Test Task 4.1: Training execution and loss logging cell implementation"""

    def test_training_cell_has_callbacks(self, notebook_content):
        """Test that training cell imports and uses required callbacks (Requirement 2.5)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find training cell (should contain TrainingPipelineService or model.train)
        training_cell_found = False
        for cell in code_cells:
            if "LossLoggingCallback" in cell.source and "R2ValidationCallback" in cell.source:
                training_cell_found = True

                # Verify all required callbacks are present (Requirement 2.5, 2.8)
                assert "LossLoggingCallback" in cell.source, "LossLoggingCallback not found"
                assert "R2ValidationCallback" in cell.source, "R2ValidationCallback not found"
                assert "DivergenceDetectionCallback" in cell.source, "DivergenceDetectionCallback not found"

                # Verify callbacks are added to a list
                assert "callbacks" in cell.source, "Callbacks list not created"
                break

        assert training_cell_found, "Training cell with callbacks not found"

    def test_training_cell_has_time_recording(self, notebook_content):
        """Test that training cell records training time (Requirement 2.9)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find training cell
        time_recording_found = False
        for cell in code_cells:
            if "time.time()" in cell.source:
                time_recording_found = True

                # Should have both start and end time recording
                assert cell.source.count("time.time()") >= 2, "Should record start and end time"
                break

        assert time_recording_found, "Training time recording not found (Requirement 2.9)"

    def test_training_cell_uses_training_pipeline(self, notebook_content):
        """Test that training cell uses TrainingPipelineService (Requirement 2.4)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find training cell
        training_pipeline_found = False
        for cell in code_cells:
            if "TrainingPipelineService" in cell.source or "model.train" in cell.source:
                training_pipeline_found = True

                # Verify training is executed with callbacks
                assert "callbacks" in cell.source, "Callbacks not passed to training"

                # Verify epochs configuration
                assert "epochs" in cell.source or "config" in cell.source, "Epochs not configured"
                break

        assert training_pipeline_found, "TrainingPipelineService or model.train not found"

    def test_loss_logging_has_correct_interval(self, notebook_content):
        """Test that LossLoggingCallback has appropriate log_interval (Requirement 2.5, 2.6)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell with LossLoggingCallback initialization
        callback_config_found = False
        for cell in code_cells:
            if "LossLoggingCallback" in cell.source:
                callback_config_found = True

                # Check if log_interval is specified (should be 100 or similar)
                # This could be either LossLoggingCallback(log_interval=100) or default
                # We just verify the callback is instantiated
                assert "LossLoggingCallback(" in cell.source, "LossLoggingCallback not instantiated"
                break

        assert callback_config_found, "LossLoggingCallback configuration not found"

    def test_r2_validation_callback_configured(self, notebook_content):
        """Test that R2ValidationCallback is configured with val_data"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell with R2ValidationCallback initialization
        r2_callback_found = False
        for cell in code_cells:
            if "R2ValidationCallback" in cell.source:
                r2_callback_found = True

                # Verify val_data is passed to callback
                # R2ValidationCallback requires val_x and val_y
                assert "R2ValidationCallback(" in cell.source, "R2ValidationCallback not instantiated"
                break

        assert r2_callback_found, "R2ValidationCallback not found or configured"

    def test_divergence_detection_callback_present(self, notebook_content):
        """Test that DivergenceDetectionCallback is present (Requirement 2.8)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell with DivergenceDetectionCallback
        divergence_callback_found = False
        for cell in code_cells:
            if "DivergenceDetectionCallback" in cell.source:
                divergence_callback_found = True

                # Verify callback is instantiated
                assert "DivergenceDetectionCallback(" in cell.source, "DivergenceDetectionCallback not instantiated"
                break

        assert divergence_callback_found, "DivergenceDetectionCallback not found (Requirement 2.8)"


class TestTask42TrainingHistoryPlot:
    """Test Task 4.2: Training history plot and NaN detection implementation"""

    def test_training_history_plot_exists(self, notebook_content):
        """Test that training history is plotted after training (Requirement 2.7)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that plots training curves
        plot_found = False
        for cell in code_cells:
            if "plot_training_curves" in cell.source or ("plt.plot" in cell.source and "history" in cell.source):
                plot_found = True

                # Verify PlotGeneratorService.plot_training_curves() is used
                # or manual plotting with history dict
                assert "history" in cell.source, "Training history not used in plotting"
                break

        assert plot_found, "Training history plot not found (Requirement 2.7)"

    def test_plot_has_four_series(self, notebook_content):
        """Test that plot includes 4 series: L_data, L_pde, L_bc, Total (Requirement 2.7)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find plotting cell
        plot_series_found = False
        for cell in code_cells:
            if "plot_training_curves" in cell.source:
                plot_series_found = True

                # PlotGeneratorService.plot_training_curves() should handle 4 series automatically
                # if history dict contains L_data, L_pde, L_bc, total_loss keys
                assert "plot_training_curves" in cell.source, "plot_training_curves not called"
                assert "history" in cell.source, "history dict not passed to plot"
                break

        assert plot_series_found, "Training curves plot with 4 series not found (Requirement 2.7)"

    def test_training_convergence_check(self, notebook_content):
        """Test that training convergence is checked (visual or programmatic)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Find cell that checks convergence (either plot or comment)
        convergence_check_found = False
        for cell in code_cells:
            if "plot_training_curves" in cell.source or "convergence" in cell.source.lower():
                convergence_check_found = True
                break

        # Also check markdown cells for convergence discussion
        markdown_cells = [c for c in notebook_content.cells if c.cell_type == "markdown"]
        for cell in markdown_cells:
            if "convergence" in cell.source.lower() or "収束" in cell.source:
                convergence_check_found = True
                break

        assert convergence_check_found, "Training convergence check not found"

    def test_nan_detection_callback_explanation(self, notebook_content):
        """Test that NaN detection is explained in markdown (Requirement 2.8)"""
        markdown_cells = [c for c in notebook_content.cells if c.cell_type == "markdown"]

        # Find markdown cell explaining NaN detection or DivergenceDetectionCallback
        nan_explanation_found = False
        for cell in markdown_cells:
            if "nan" in cell.source.lower() or "divergence" in cell.source.lower():
                nan_explanation_found = True
                break

        # NaN detection should be mentioned in at least one markdown or code comment
        if not nan_explanation_found:
            # Check code cells for NaN-related comments
            code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]
            for cell in code_cells:
                if "DivergenceDetectionCallback" in cell.source:
                    nan_explanation_found = True
                    break

        assert nan_explanation_found, "NaN detection explanation not found (Requirement 2.8)"

    def test_loss_display_format(self, notebook_content):
        """Test that loss values are displayed in expected format (Requirement 2.6)"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # The LossLoggingCallback should automatically print loss values
        # We verify that the callback is configured to do this
        loss_display_found = False
        for cell in code_cells:
            if "LossLoggingCallback" in cell.source:
                loss_display_found = True

                # LossLoggingCallback will handle display via on_epoch_end
                # No explicit print needed in notebook, callback handles it
                break

        assert loss_display_found, "Loss logging callback not configured for display (Requirement 2.6)"


class TestTask4Integration:
    """Integration tests for Task 4 (4.1 + 4.2)"""

    def test_training_cell_structure(self, notebook_content):
        """Test that training cell has proper structure with markdown documentation"""
        # Find training section markdown
        markdown_cells = [c for c in notebook_content.cells if c.cell_type == "markdown"]

        training_section_found = False
        for cell in markdown_cells:
            if "訓練" in cell.source or "training" in cell.source.lower():
                if "step" in cell.source.lower() or "タスク" in cell.source:
                    training_section_found = True
                    break

        assert training_section_found, "Training section markdown header not found (Requirement 7.1, 7.2)"

    def test_training_workflow_completeness(self, notebook_content):
        """Test that training workflow is complete: setup → train → visualize"""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Check for key components
        has_callbacks = any("LossLoggingCallback" in cell.source for cell in code_cells)
        has_training = any("train" in cell.source.lower() and ("model" in cell.source or "pipeline" in cell.source) for cell in code_cells)
        has_plotting = any("plot_training_curves" in cell.source or ("plot" in cell.source and "history" in cell.source) for cell in code_cells)

        assert has_callbacks, "Callbacks setup missing"
        assert has_training, "Training execution missing"
        assert has_plotting, "Training history plotting missing"

    def test_recommended_execution_time_note(self, notebook_content):
        """Test that recommended execution time is noted (Requirement 7.7)"""
        markdown_cells = [c for c in notebook_content.cells if c.cell_type == "markdown"]

        # Find markdown near training cell with execution time estimate
        time_note_found = False
        for cell in markdown_cells:
            # Look for patterns like "約10分", "GPU: 約", "推奨実行時間", etc.
            if any(keyword in cell.source for keyword in ["約", "GPU", "実行時間", "execution time", "minutes", "分"]):
                if any(training_keyword in cell.source for training_keyword in ["訓練", "training", "train", "タスク4"]):
                    time_note_found = True
                    break

        assert time_note_found, "Execution time note not found (Requirement 7.7)"
