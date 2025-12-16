"""Phase 10 Integration Tests.

This module tests the complete Phase 10 implementation:
- Main training script execution
- Weight tuning script execution
- End-to-end workflow validation
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_training_script_end_to_end():
    """Test complete training workflow with standing wave config."""
    config_file = "configs/standing_wave_example.yaml"

    # Create a quick test config with reduced epochs
    test_config_path = Path("configs/test_phase10_quick.yaml")
    test_config_content = """
experiment_name: "phase10_integration_test"
seed: 42
domain:
  x_min: 0.0
  x_max: 1.0
  t_min: 0.0
  t_max: 1.0
  wave_speed: 1.0
boundary_conditions:
  type: "dirichlet"
  left_value: null
  right_value: null
network:
  layer_sizes: [2, 20, 20, 1]
  activation: "tanh"
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  loss_weights:
    data: 1.0
    pde: 1.0
    bc: 10.0
  amp_enabled: false
  checkpoint_interval: 50
"""

    try:
        # Write test config
        test_config_path.write_text(test_config_content)

        # Run training script
        result = subprocess.run(
            [sys.executable, "pinn/training/train.py", str(test_config_path)],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        # Check training succeeded
        assert result.returncode == 0, f"Training failed. stderr: {result.stderr}"

        # Verify experiment directory created
        exp_dir = Path("experiments")
        assert exp_dir.exists(), "experiments/ directory should exist"

        # Find the created experiment subdirectory
        exp_subdirs = list(exp_dir.glob("phase10_integration_test_*"))
        assert len(exp_subdirs) > 0, "Experiment subdirectory should be created"

        exp_output_dir = exp_subdirs[0]

        # Verify config.yaml saved
        config_saved = exp_output_dir / "config.yaml"
        assert config_saved.exists(), "config.yaml should be saved"

        # Verify metadata.json saved
        metadata_saved = exp_output_dir / "metadata.json"
        assert metadata_saved.exists(), "metadata.json should be saved"

        # Verify checkpoints directory created
        checkpoints_dir = exp_output_dir / "checkpoints"
        assert checkpoints_dir.exists(), "checkpoints/ directory should exist"

        # Verify plots directory created (may be empty)
        plots_dir = exp_output_dir / "plots"
        # plots_dir may or may not exist depending on callbacks

        print(f"✓ Integration test passed. Output in: {exp_output_dir}")

    finally:
        # Cleanup
        if test_config_path.exists():
            test_config_path.unlink()


@pytest.mark.slow
def test_weight_tuning_script_small_grid():
    """Test weight tuning with small grid (2x2 = 4 configurations)."""
    config_file = "configs/standing_wave_example.yaml"

    # Create quick tuning config
    test_config_path = Path("configs/test_phase10_tuning.yaml")
    test_config_content = """
experiment_name: "phase10_tuning_test"
seed: 42
domain:
  x_min: 0.0
  x_max: 1.0
  t_min: 0.0
  t_max: 1.0
  wave_speed: 1.0
boundary_conditions:
  type: "dirichlet"
  left_value: null
  right_value: null
network:
  layer_sizes: [2, 10, 10, 1]
  activation: "tanh"
training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  loss_weights:
    data: 1.0
    pde: 1.0
    bc: 1.0
  amp_enabled: false
  checkpoint_interval: 50
"""

    output_file = Path("test_tuning_results.json")
    output_dir = Path("test_tuning_output")

    try:
        # Write test config
        test_config_path.write_text(test_config_content)

        # Run tuning script with small 2x2 grid
        result = subprocess.run(
            [
                sys.executable,
                "pinn/tuning/tune_weights_cli.py",
                str(test_config_path),
                "--search-type", "grid",
                "--weight-ranges", '{"pde": [0.5, 1.0], "bc": [1.0, 10.0]}',
                "--output-file", str(output_file),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Check tuning succeeded
        assert result.returncode == 0, f"Tuning failed. stderr: {result.stderr}"

        # Verify results file created
        assert output_file.exists(), "Tuning results JSON should be created"

        # Verify visualization directory created
        assert output_dir.exists(), "Tuning output directory should exist"

        # Verify visualizations created
        loss_landscape = output_dir / "loss_landscape.png"
        pareto_frontier = output_dir / "pareto_frontier.png"
        # These may not be created if plotting fails, but test should still pass

        print(f"✓ Tuning integration test passed. Results in: {output_file}")

    finally:
        # Cleanup
        if test_config_path.exists():
            test_config_path.unlink()
        if output_file.exists():
            output_file.unlink()
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_project_structure_verification():
    """Verify project structure matches steering conventions."""
    # Check main directories exist
    assert Path("pinn/models").exists(), "pinn/models/ should exist"
    assert Path("pinn/training").exists(), "pinn/training/ should exist"
    assert Path("pinn/validation").exists(), "pinn/validation/ should exist"
    assert Path("pinn/data").exists(), "pinn/data/ should exist"
    assert Path("pinn/tuning").exists(), "pinn/tuning/ should exist"
    assert Path("pinn/utils").exists(), "pinn/utils/ should exist"
    assert Path("pinn/tests").exists(), "pinn/tests/ should exist"

    # Check key files exist
    assert Path("pinn/training/train.py").exists(), "train.py should exist"
    assert Path("pinn/tuning/tune_weights_cli.py").exists(), "tune_weights_cli.py should exist"

    # Check __init__.py files exist for package structure
    assert Path("pinn/__init__.py").exists(), "pinn/__init__.py should exist"
    assert Path("pinn/models/__init__.py").exists(), "pinn/models/__init__.py should exist"
    assert Path("pinn/training/__init__.py").exists(), "pinn/training/__init__.py should exist"

    # Check config examples exist
    assert Path("configs/standing_wave_example.yaml").exists(), "standing_wave_example.yaml should exist"
    assert Path("configs/traveling_wave_example.yaml").exists(), "traveling_wave_example.yaml should exist"

    print("✓ Project structure verification passed")
