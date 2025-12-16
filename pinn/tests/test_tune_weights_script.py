"""Tests for weight tuning script.

This module tests the weight tuning CLI script entry point.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_tune_weights_script_exists():
    """Test that tune_weights.py script exists."""
    script = Path("pinn/tuning/tune_weights_cli.py")
    assert script.exists(), "tune_weights_cli.py script should exist"


def test_tune_weights_script_has_main_function():
    """Test that tune_weights_cli.py has a main function."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tune_weights_cli", "pinn/tuning/tune_weights_cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "main"), "tune_weights_cli.py should have main() function"


def test_tune_weights_script_requires_arguments():
    """Test that tune_weights_cli.py requires config and tuning arguments."""
    result = subprocess.run(
        [sys.executable, "pinn/tuning/tune_weights_cli.py"],
        capture_output=True,
        text=True
    )

    # Should exit with non-zero code when no config provided
    assert result.returncode != 0, "Should fail when no arguments provided"


def test_tune_weights_script_handles_nonexistent_config():
    """Test that script handles non-existent config file gracefully."""
    result = subprocess.run(
        [
            sys.executable,
            "pinn/tuning/tune_weights_cli.py",
            "nonexistent_config.yaml",
            "--search-type", "grid",
            "--weight-ranges", '{"data": [1.0], "pde": [1.0], "bc": [1.0]}'
        ],
        capture_output=True,
        text=True
    )

    # Should exit with non-zero code
    assert result.returncode != 0, "Should fail with non-existent config"
