"""Tests for project directory structure.

This module tests that the required directory structure exists
for the PINN 1D Foundation implementation.
"""

from pathlib import Path

import pytest


class TestProjectStructure:
    """Test suite for validating project directory structure."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Return the project root directory."""
        # Navigate up from pinn/tests/ to project root
        return Path(__file__).parent.parent.parent

    def test_pinn_directory_exists(self, project_root: Path) -> None:
        """Test that /pinn/ directory exists."""
        pinn_dir = project_root / "pinn"
        assert pinn_dir.exists(), "/pinn/ directory should exist"
        assert pinn_dir.is_dir(), "/pinn/ should be a directory"

    def test_pinn_subdirectories_exist(self, project_root: Path) -> None:
        """Test that all required /pinn/ subdirectories exist."""
        pinn_dir = project_root / "pinn"
        required_subdirs = [
            "models",
            "training",
            "validation",
            "data",
            "tuning",
            "utils",
            "tests"
        ]

        for subdir in required_subdirs:
            subdir_path = pinn_dir / subdir
            assert subdir_path.exists(), f"/pinn/{subdir}/ should exist"
            assert subdir_path.is_dir(), f"/pinn/{subdir}/ should be a directory"

    def test_configs_directory_exists(self, project_root: Path) -> None:
        """Test that /configs/ directory exists for YAML configurations."""
        configs_dir = project_root / "configs"
        assert configs_dir.exists(), "/configs/ directory should exist"
        assert configs_dir.is_dir(), "/configs/ should be a directory"

    def test_experiments_directory_exists(self, project_root: Path) -> None:
        """Test that /experiments/ directory exists for output."""
        experiments_dir = project_root / "experiments"
        assert experiments_dir.exists(), "/experiments/ directory should exist"
        assert experiments_dir.is_dir(), "/experiments/ should be a directory"

    def test_pinn_package_structure(self, project_root: Path) -> None:
        """Test that __init__.py files exist for Python package structure."""
        pinn_dir = project_root / "pinn"

        # Check main pinn package
        init_file = pinn_dir / "__init__.py"
        assert init_file.exists(), "/pinn/__init__.py should exist"
        assert init_file.is_file(), "/pinn/__init__.py should be a file"

        # Check subdirectory packages
        package_subdirs = ["models", "training", "validation", "data", "tuning", "utils"]
        for subdir in package_subdirs:
            init_file = pinn_dir / subdir / "__init__.py"
            assert init_file.exists(), f"/pinn/{subdir}/__init__.py should exist"
            assert init_file.is_file(), f"/pinn/{subdir}/__init__.py should be a file"

    def test_experiments_gitkeep_exists(self, project_root: Path) -> None:
        """Test that /experiments/ has .gitkeep to track empty directory."""
        experiments_dir = project_root / "experiments"
        gitkeep = experiments_dir / ".gitkeep"
        assert gitkeep.exists(), "/experiments/.gitkeep should exist"
