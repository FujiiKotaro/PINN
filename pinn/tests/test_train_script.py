"""Tests for main training script.

This module tests the main training script entry point for PINN training.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_train_script_exists():
    """Test that train.py script exists."""
    train_script = Path("pinn/training/train.py")
    assert train_script.exists(), "train.py script should exist"


def test_train_script_has_main_function():
    """Test that train.py has a main function."""
    # Import the train module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train", "pinn/training/train.py"
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    # Check for main function
    assert hasattr(train_module, "main"), "train.py should have a main() function"


def test_train_script_requires_config_argument():
    """Test that train.py requires a config file argument."""
    result = subprocess.run(
        [sys.executable, "pinn/training/train.py"],
        capture_output=True,
        text=True
    )

    # Should exit with non-zero code when no config provided
    assert result.returncode != 0, "Should fail when no config file provided"


def test_train_script_handles_nonexistent_config():
    """Test that train.py handles non-existent config file gracefully."""
    result = subprocess.run(
        [sys.executable, "pinn/training/train.py", "nonexistent_config.yaml"],
        capture_output=True,
        text=True
    )

    # Should exit with non-zero code
    assert result.returncode != 0, "Should fail with non-existent config"
    # Should have error message mentioning file not found
    assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()


@pytest.mark.slow
def test_train_script_with_valid_config():
    """Test that train.py runs successfully with valid config."""
    # Create a minimal test config
    test_config = Path("configs/test_quick_train.yaml")
    test_config_content = """
experiment_name: "test_quick_train"
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
  epochs: 10
  learning_rate: 0.001
  optimizer: "adam"
  loss_weights:
    data: 1.0
    pde: 1.0
    bc: 10.0
  amp_enabled: false
  checkpoint_interval: 10
"""

    try:
        # Write test config
        test_config.write_text(test_config_content)

        # Run training script
        result = subprocess.run(
            [sys.executable, "pinn/training/train.py", str(test_config)],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )

        # Should succeed
        assert result.returncode == 0, f"Training should succeed. stderr: {result.stderr}"

        # Should have output directory created
        output_dir = Path("experiments")
        assert output_dir.exists(), "experiments/ directory should be created"

    finally:
        # Cleanup
        if test_config.exists():
            test_config.unlink()
