"""
Test notebook execution for pinn_2d_forward_validation.ipynb

This test validates that the notebook can execute all cells without errors
and produces expected outputs (Task 9.1).
"""

import json
from pathlib import Path

import nbformat
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


class TestNotebookStructure:
    """Test notebook structure and metadata (TDD: RED phase)."""

    def test_notebook_exists(self, notebook_path):
        """Test that notebook file exists."""
        assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    def test_notebook_has_cells(self, notebook_content):
        """Test that notebook has at least one cell."""
        assert len(notebook_content.cells) > 0, "Notebook has no cells"

    def test_notebook_has_markdown_cells(self, notebook_content):
        """Test that notebook has markdown cells (documentation)."""
        markdown_cells = [c for c in notebook_content.cells if c.cell_type == "markdown"]
        assert len(markdown_cells) > 0, "Notebook has no markdown cells"

    def test_notebook_has_code_cells(self, notebook_content):
        """Test that notebook has code cells."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]
        assert len(code_cells) > 0, "Notebook has no code cells"

    def test_first_cell_is_overview(self, notebook_content):
        """Test that first cell contains overview (Requirement 7.1, 7.2)."""
        first_cell = notebook_content.cells[0]
        assert first_cell.cell_type == "markdown", "First cell should be markdown"

        # Check for required overview content
        overview_text = first_cell.source.lower()
        assert "概要" in first_cell.source, "First cell should contain '概要' section"
        assert "phase 2" in overview_text, "Overview should mention Phase 2"
        assert "pinn_data" in overview_text or "/pinn_data" in overview_text, "Overview should mention /PINN_data directory"
        assert "gpu" in overview_text, "Overview should mention GPU requirement"


class TestNotebookImports:
    """Test that notebook has correct imports (Task 1.2, Requirement 8.1)."""

    def test_setup_cell_has_imports(self, notebook_content):
        """Test that notebook has a setup cell with imports."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]
        assert len(code_cells) > 0, "No code cells found"

        # First code cell should have imports
        first_code_cell = code_cells[0]
        imports = first_code_cell.source

        # Check for required imports (Requirement 8.1)
        required_imports = [
            "from pinn.data.fdtd_loader import FDTDDataLoaderService",
            "from pinn.data.dimensionless_scaler import",
            "from pinn.models.pinn_model_builder_2d import PINNModelBuilder2DService",
            "from pinn.models.pde_definition_2d import PDEDefinition2DService",
            "from pinn.training",
            "from pinn.validation.r2_score import R2ScoreCalculator",
            "from pinn.validation.plot_generator import PlotGeneratorService",
            "from pinn.utils.seed_manager import SeedManager",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import torch",
        ]

        for required_import in required_imports:
            assert required_import in imports, f"Missing import: {required_import}"

    def test_seed_is_set(self, notebook_content):
        """Test that notebook sets random seed (Requirement 7.4)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]
        first_code_cell = code_cells[0]

        assert "SeedManager.set_seed" in first_code_cell.source, "Seed not set in setup cell"
        assert "SEED = 42" in first_code_cell.source, "SEED should be 42"

    def test_gpu_check_present(self, notebook_content):
        """Test that notebook checks GPU availability (Requirement 2.9)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]
        first_code_cell = code_cells[0]

        assert "torch.cuda.is_available()" in first_code_cell.source, "GPU check missing"
        assert "torch.cuda.get_device_name" in first_code_cell.source, "GPU device name check missing"


class TestTask2DataPreparation:
    """Test Task 2: FDTD data loading and preprocessing cells (TDD: RED phase)."""

    def test_data_loading_cell_exists(self, notebook_content):
        """Test that data loading cell exists (Task 2.1)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Should have at least 2 code cells: setup + data loading
        assert len(code_cells) >= 2, "Data loading cell not found"

    def test_data_loading_cell_has_fdtd_loader(self, notebook_content):
        """Test that data loading cell uses FDTDDataLoaderService (Requirement 1.1, 1.2)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        # Check if any code cell contains FDTD data loading
        found_loader = False
        for cell in code_cells[1:]:  # Skip setup cell
            if "FDTDDataLoaderService" in cell.source and "load_multiple_files" in cell.source:
                found_loader = True
                break

        assert found_loader, "FDTDDataLoaderService.load_multiple_files() not found in notebook"

    def test_data_loading_checks_directory(self, notebook_content):
        """Test that data loading cell checks /PINN_data directory (Requirement 1.7)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        found_check = False
        for cell in code_cells[1:]:
            if "/PINN_data" in cell.source or "PINN_data" in cell.source:
                if "exists()" in cell.source or "FileNotFoundError" in cell.source:
                    found_check = True
                    break

        assert found_check, "Data directory existence check not found (Requirement 1.7)"

    def test_dimensionless_scaling_exists(self, notebook_content):
        """Test that dimensionless scaling is applied (Requirement 1.3)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        found_scaler = False
        for cell in code_cells[1:]:
            if "DimensionlessScalerService" in cell.source or "CharacteristicScales" in cell.source:
                found_scaler = True
                break

        assert found_scaler, "DimensionlessScalerService not found in notebook"

    def test_variable_ranges_displayed(self, notebook_content):
        """Test that dimensionless variable ranges are displayed (Requirement 1.4)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        found_ranges = False
        for cell in code_cells[1:]:
            # Check for range display (x̃, ỹ, t̃, etc.)
            if "np.min" in cell.source and "np.max" in cell.source:
                found_ranges = True
                break

        assert found_ranges, "Variable range display not found (Requirement 1.4)"

    def test_train_val_split_exists(self, notebook_content):
        """Test that train/val split is performed (Requirement 1.5)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        found_split = False
        for cell in code_cells[1:]:
            if "train_val_split" in cell.source or ("train_data" in cell.source and "val_data" in cell.source):
                found_split = True
                break

        assert found_split, "Train/val split not found (Requirement 1.5)"

    def test_data_distribution_visualization_exists(self, notebook_content):
        """Test that data distribution visualization exists (Requirement 1.6)."""
        code_cells = [c for c in notebook_content.cells if c.cell_type == "code"]

        found_viz = False
        for cell in code_cells[1:]:
            if "scatter" in cell.source or "plt." in cell.source:
                found_viz = True
                break

        assert found_viz, "Data distribution visualization not found (Requirement 1.6)"


class TestNotebookExecution:
    """Test notebook execution (Task 9.1: Requirement 7.5)."""

    @pytest.mark.slow
    def test_setup_cell_executes(self, notebook_path, tmp_path):
        """Test that setup cell can execute without errors."""
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Execute only first code cell (setup)
        ep = ExecutePreprocessor(timeout=60, kernel_name="python3")

        # Create a notebook with only first 2 cells (markdown overview + setup code)
        test_nb = nbformat.v4.new_notebook()
        test_nb.cells = nb.cells[:2]

        try:
            ep.preprocess(test_nb, {"metadata": {"path": str(tmp_path)}})
        except Exception as e:
            pytest.fail(f"Setup cell execution failed: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_notebook_executes_without_errors(self, notebook_path, tmp_path):
        """
        Test that entire notebook executes without errors (Task 9.1).

        This is a full integration test that executes all cells sequentially.
        Note: This test requires PINN_data/ with valid .npz files and GPU.
        """
        import os

        # Check if PINN_data directory exists
        pinn_data_dir = Path(__file__).parent.parent / "PINN_data"
        if not pinn_data_dir.exists():
            pytest.skip(f"Skipping integration test: {pinn_data_dir} not found")

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Set PYTHONPATH to include project root
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        project_root = str(Path(__file__).parent.parent)
        os.environ["PYTHONPATH"] = f"{project_root}:{old_pythonpath}"

        # ExecutePreprocessor with extended timeout for training (30 minutes)
        ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")

        # Execute notebook from notebooks directory so Path.cwd() is correct
        notebooks_dir = notebook_path.parent

        try:
            # Execute all cells (use notebooks directory as working directory)
            nb_executed, resources = ep.preprocess(nb, {"metadata": {"path": str(notebooks_dir)}})

            # Verify execution completed
            assert nb_executed is not None, "Notebook execution returned None"

            # Check for errors in cell outputs
            for i, cell in enumerate(nb_executed.cells):
                if cell.cell_type == "code":
                    for output in cell.get("outputs", []):
                        if output.get("output_type") == "error":
                            error_msg = "\n".join(output.get("traceback", []))
                            pytest.fail(f"Cell {i} produced error:\n{error_msg}")

        except Exception as e:
            pytest.fail(f"Notebook execution failed: {e}")
        finally:
            # Restore PYTHONPATH
            os.environ["PYTHONPATH"] = old_pythonpath

    @pytest.mark.integration
    @pytest.mark.slow
    def test_notebook_final_kernel_state(self, notebook_path, tmp_path):
        """
        Test that final kernel state contains expected variables (Task 9.1).

        Validates that key variables exist after notebook execution:
        - r2_scores: R² scores dictionary
        - trained_model: Trained PINN model
        - dataset: Full dataset
        - train_data, val_data: Train/val split data
        """
        import os

        # Check if PINN_data directory exists
        pinn_data_dir = Path(__file__).parent.parent / "PINN_data"
        if not pinn_data_dir.exists():
            pytest.skip(f"Skipping integration test: {pinn_data_dir} not found")

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Set PYTHONPATH to include project root
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        project_root = str(Path(__file__).parent.parent)
        os.environ["PYTHONPATH"] = f"{project_root}:{old_pythonpath}"

        ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")

        # Execute notebook from notebooks directory so Path.cwd() is correct
        notebooks_dir = notebook_path.parent

        try:
            nb_executed, resources = ep.preprocess(nb, {"metadata": {"path": str(notebooks_dir)}})

            # Add a verification cell to check kernel state
            verification_code = """
# Verify key variables exist
required_vars = ['r2_scores', 'trained_model', 'dataset', 'train_data', 'val_data',
                 'fdtd_data_2d', 'pinn_pred_2d', 'mean_error', 'rel_error']

missing_vars = []
for var_name in required_vars:
    if var_name not in dir():
        missing_vars.append(var_name)

if missing_vars:
    raise AssertionError(f"Missing variables in kernel state: {missing_vars}")

# Verify r2_scores has correct structure
assert isinstance(r2_scores, dict), "r2_scores should be dict"
assert len(r2_scores) == 4, f"r2_scores should have 4 fields, got {len(r2_scores)}"
assert all(field in r2_scores for field in ['T1', 'T3', 'Ux', 'Uy']), "r2_scores missing fields"

print("✓ All required variables present in kernel state")
print(f"✓ R² scores: {r2_scores}")
"""

            # Execute verification code in same kernel
            verification_cell = nbformat.v4.new_code_cell(source=verification_code)
            nb_executed.cells.append(verification_cell)

            ep_verify = ExecutePreprocessor(timeout=60, kernel_name="python3")
            nb_final, _ = ep_verify.preprocess(nb_executed, {"metadata": {"path": str(notebooks_dir)}})

            # Check verification cell output
            last_cell = nb_final.cells[-1]
            assert last_cell.cell_type == "code"

            # Verify no errors in verification cell
            for output in last_cell.get("outputs", []):
                if output.get("output_type") == "error":
                    error_msg = "\n".join(output.get("traceback", []))
                    pytest.fail(f"Kernel state verification failed:\n{error_msg}")

        except Exception as e:
            pytest.fail(f"Kernel state verification failed: {e}")
        finally:
            # Restore PYTHONPATH
            os.environ["PYTHONPATH"] = old_pythonpath
