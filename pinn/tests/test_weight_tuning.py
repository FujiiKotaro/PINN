"""Unit tests for WeightTuningFramework."""
import itertools
from pathlib import Path

from pinn.tuning.weight_tuning import (
    TuningConfig,
    TuningResult,
    WeightTuningFrameworkService,
)


class TestGridSearch:
    """Test grid search functionality."""

    def test_grid_search_generates_all_combinations(self) -> None:
        """Grid search should generate all combinations of weight ranges."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0],
            "pde": [0.5, 1.0],
            "bc": [0.1, 1.0],
        }

        # Act
        combinations = framework._grid_search(weight_ranges)

        # Assert
        expected_count = 2 * 2 * 2  # 8 combinations
        assert len(combinations) == expected_count

        # Verify all combinations are unique
        assert len(set(tuple(sorted(c.items())) for c in combinations)) == expected_count

    def test_grid_search_with_single_value(self) -> None:
        """Grid search with single value per parameter should return one combination."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [1.0],
            "pde": [1.0],
            "bc": [1.0],
        }

        # Act
        combinations = framework._grid_search(weight_ranges)

        # Assert
        assert len(combinations) == 1
        assert combinations[0] == {"data": 1.0, "pde": 1.0, "bc": 1.0}

    def test_grid_search_preserves_keys(self) -> None:
        """Grid search should preserve all weight keys in output."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0],
            "pde": [0.5],
            "bc": [0.1, 10.0],
        }

        # Act
        combinations = framework._grid_search(weight_ranges)

        # Assert
        for combo in combinations:
            assert set(combo.keys()) == {"data", "pde", "bc"}

    def test_grid_search_with_three_values(self) -> None:
        """Grid search with 3x3x3 grid should generate 27 combinations."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0, 10.0],
            "pde": [0.1, 1.0, 10.0],
            "bc": [0.1, 1.0, 10.0],
        }

        # Act
        combinations = framework._grid_search(weight_ranges)

        # Assert
        assert len(combinations) == 27

        # Verify expected combinations are present
        expected_combos = list(itertools.product([0.1, 1.0, 10.0], repeat=3))
        for data_w, pde_w, bc_w in expected_combos:
            expected = {"data": data_w, "pde": pde_w, "bc": bc_w}
            assert expected in combinations


class TestRandomSearch:
    """Test random search functionality."""

    def test_random_search_generates_requested_samples(self) -> None:
        """Random search should generate exactly n_samples combinations."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0, 10.0],
            "pde": [0.5, 1.0],
            "bc": [0.1, 1.0],
        }
        n_samples = 50

        # Act
        combinations = framework._random_search(weight_ranges, n_samples)

        # Assert
        assert len(combinations) == n_samples

    def test_random_search_uses_seed_for_reproducibility(self) -> None:
        """Random search with same seed should produce identical results."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0, 10.0],
            "pde": [0.5, 1.0, 5.0],
            "bc": [0.1, 1.0],
        }
        n_samples = 20

        # Act
        combinations1 = framework._random_search(weight_ranges, n_samples, seed=42)
        combinations2 = framework._random_search(weight_ranges, n_samples, seed=42)

        # Assert
        assert combinations1 == combinations2

    def test_random_search_samples_from_ranges(self) -> None:
        """Random search should only sample values from specified ranges."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0],
            "pde": [0.5],
            "bc": [10.0],
        }
        n_samples = 100

        # Act
        combinations = framework._random_search(weight_ranges, n_samples)

        # Assert
        for combo in combinations:
            assert combo["data"] in [0.1, 1.0]
            assert combo["pde"] == 0.5
            assert combo["bc"] == 10.0

    def test_random_search_preserves_keys(self) -> None:
        """Random search should preserve all weight keys in output."""
        # Arrange
        framework = WeightTuningFrameworkService()
        weight_ranges = {
            "data": [0.1, 1.0],
            "pde": [0.5, 1.0],
            "bc": [0.1],
        }
        n_samples = 30

        # Act
        combinations = framework._random_search(weight_ranges, n_samples)

        # Assert
        for combo in combinations:
            assert set(combo.keys()) == {"data", "pde", "bc"}


class TestTuningConfig:
    """Test TuningConfig dataclass."""

    def test_tuning_config_creation_grid(self) -> None:
        """TuningConfig should be created with grid search type."""
        # Arrange & Act
        config = TuningConfig(
            search_type="grid",
            weight_ranges={"data": [1.0], "pde": [1.0], "bc": [1.0]},
        )

        # Assert
        assert config.search_type == "grid"
        assert config.weight_ranges == {"data": [1.0], "pde": [1.0], "bc": [1.0]}
        assert config.n_samples == 100  # Default value
        assert config.output_path == Path("tuning_results.json")

    def test_tuning_config_creation_random(self) -> None:
        """TuningConfig should be created with random search type."""
        # Arrange & Act
        config = TuningConfig(
            search_type="random",
            weight_ranges={"data": [0.1, 1.0], "pde": [0.5, 1.0], "bc": [0.1, 1.0]},
            n_samples=50,
            output_path=Path("custom_results.json"),
        )

        # Assert
        assert config.search_type == "random"
        assert config.n_samples == 50
        assert config.output_path == Path("custom_results.json")


class TestTuningResult:
    """Test TuningResult dataclass."""

    def test_tuning_result_creation(self) -> None:
        """TuningResult should store all tuning metrics."""
        # Arrange & Act
        result = TuningResult(
            w_data=1.0,
            w_pde=0.5,
            w_bc=0.1,
            validation_error=0.032,
            training_time=123.45,
        )

        # Assert
        assert result.w_data == 1.0
        assert result.w_pde == 0.5
        assert result.w_bc == 0.1
        assert result.validation_error == 0.032
        assert result.training_time == 123.45


class TestRunTuning:
    """Test run_tuning execution loop."""

    def test_run_tuning_grid_search(self, tmp_path: Path) -> None:
        """run_tuning with grid search should train all configurations."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        base_config = ExperimentConfig(
            experiment_name="test_tuning",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(epochs=10),
        )

        tuning_config = TuningConfig(
            search_type="grid",
            weight_ranges={"data": [0.1, 1.0], "pde": [0.5, 1.0], "bc": [0.1]},
            output_path=tmp_path / "results.json",
        )

        # Mock training function
        call_count = 0

        def mock_train_fn(config: ExperimentConfig) -> tuple[float, float]:
            nonlocal call_count
            call_count += 1
            # Return validation error based on weights (lower w_data = better)
            w_data = config.training.loss_weights.get("data", 1.0)
            return (w_data * 0.01, 10.0)  # (validation_error, training_time)

        framework = WeightTuningFrameworkService()

        # Act
        best_result, all_results = framework.run_tuning(base_config, tuning_config, mock_train_fn)

        # Assert
        expected_configs = 2 * 2 * 1  # 4 grid combinations
        assert call_count == expected_configs
        assert len(all_results) == expected_configs

        # Best should be lowest w_data (0.1)
        assert best_result.w_data == 0.1
        assert best_result.validation_error == 0.001  # 0.1 * 0.01

        # Verify results file was created
        assert (tmp_path / "results.json").exists()

    def test_run_tuning_random_search(self, tmp_path: Path) -> None:
        """run_tuning with random search should sample n configurations."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        base_config = ExperimentConfig(
            experiment_name="test_random",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(epochs=10),
        )

        tuning_config = TuningConfig(
            search_type="random",
            weight_ranges={"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [0.1, 1.0]},
            n_samples=10,
            output_path=tmp_path / "random_results.csv",
        )

        # Mock training function
        call_count = 0

        def mock_train_fn(config: ExperimentConfig) -> tuple[float, float]:
            nonlocal call_count
            call_count += 1
            return (0.05, 15.0)

        framework = WeightTuningFrameworkService()

        # Act
        best_result, all_results = framework.run_tuning(base_config, tuning_config, mock_train_fn)

        # Assert
        assert call_count == 10
        assert len(all_results) == 10

        # Verify CSV file was created
        assert (tmp_path / "random_results.csv").exists()

    def test_run_tuning_invalid_search_type(self, tmp_path: Path) -> None:
        """run_tuning should raise ValueError for invalid search type."""
        # Arrange
        from pinn.utils.config_loader import (
            BoundaryConditionConfig,
            DomainConfig,
            ExperimentConfig,
            NetworkConfig,
            TrainingConfig,
        )

        base_config = ExperimentConfig(
            experiment_name="test_invalid",
            seed=42,
            domain=DomainConfig(),
            boundary_conditions=BoundaryConditionConfig(type="dirichlet"),
            network=NetworkConfig(),
            training=TrainingConfig(epochs=10),
        )

        # Create invalid tuning config by manually setting invalid type
        tuning_config = TuningConfig(
            search_type="grid",  # Set to grid first
            weight_ranges={"data": [1.0], "pde": [1.0], "bc": [1.0]},
            output_path=tmp_path / "invalid.json",
        )
        tuning_config.search_type = "invalid"  # type: ignore

        def mock_train_fn(config: ExperimentConfig) -> tuple[float, float]:
            return (0.05, 10.0)

        framework = WeightTuningFrameworkService()

        # Act & Assert
        try:
            framework.run_tuning(base_config, tuning_config, mock_train_fn)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid search_type" in str(e)


class TestVisualization:
    """Test visualization functionality."""

    def test_visualize_results_creates_plots(self, tmp_path: Path) -> None:
        """visualize_results should create loss landscape and Pareto frontier plots."""
        # Arrange
        framework = WeightTuningFrameworkService()

        # Create sample results with grid search pattern
        results = [
            TuningResult(w_data=0.1, w_pde=0.5, w_bc=0.1, validation_error=0.05, training_time=10.0),
            TuningResult(w_data=1.0, w_pde=0.5, w_bc=0.1, validation_error=0.03, training_time=12.0),
            TuningResult(w_data=0.1, w_pde=1.0, w_bc=0.1, validation_error=0.04, training_time=11.0),
            TuningResult(w_data=1.0, w_pde=1.0, w_bc=0.1, validation_error=0.02, training_time=13.0),
        ]

        output_dir = tmp_path / "plots"

        # Act
        framework.visualize_results(results, output_dir)

        # Assert
        assert output_dir.exists()
        assert (output_dir / "loss_landscape.png").exists()
        assert (output_dir / "pareto_frontier.png").exists()

    def test_visualize_results_with_multiple_bc_values(self, tmp_path: Path) -> None:
        """visualize_results should handle multiple w_bc values by filtering."""
        # Arrange
        framework = WeightTuningFrameworkService()

        # Create results with different w_bc values
        results = [
            TuningResult(w_data=0.1, w_pde=0.5, w_bc=0.1, validation_error=0.05, training_time=10.0),
            TuningResult(w_data=1.0, w_pde=0.5, w_bc=0.1, validation_error=0.03, training_time=12.0),
            TuningResult(w_data=0.1, w_pde=0.5, w_bc=1.0, validation_error=0.06, training_time=11.0),
            TuningResult(w_data=1.0, w_pde=0.5, w_bc=1.0, validation_error=0.04, training_time=13.0),
        ]

        output_dir = tmp_path / "plots_multi_bc"

        # Act
        framework.visualize_results(results, output_dir)

        # Assert - should complete without error
        assert (output_dir / "loss_landscape.png").exists()
        assert (output_dir / "pareto_frontier.png").exists()
