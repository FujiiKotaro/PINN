"""Weight tuning framework for loss function hyperparameter optimization.

This module provides automated grid search and random search capabilities
for finding optimal loss weight combinations (w_data, w_pde, w_bc).
"""
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pinn.utils.config_loader import ExperimentConfig


@dataclass
class TuningConfig:
    """Configuration for weight tuning framework.

    Attributes:
        search_type: Type of search ("grid" or "random")
        weight_ranges: Dictionary mapping weight names to list of values to try.
            Example: {"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [0.1, 1.0]}
        n_samples: Number of random samples to generate (only used for random search)
        output_path: Path to save tuning results (JSON or CSV format)
    """

    search_type: Literal["grid", "random"]
    weight_ranges: dict[str, list[float]]
    n_samples: int = 100
    output_path: Path = Path("tuning_results.json")


@dataclass
class TuningResult:
    """Result from a single tuning configuration.

    Attributes:
        w_data: Weight for data fitting loss component
        w_pde: Weight for PDE residual loss component
        w_bc: Weight for boundary condition loss component
        validation_error: Validation error metric (L2 or relative error)
        training_time: Training time in seconds
    """

    w_data: float
    w_pde: float
    w_bc: float
    validation_error: float
    training_time: float


class WeightTuningFrameworkService:
    """Automated grid and random search for loss weight tuning.

    This service implements systematic hyperparameter search over loss function
    weight combinations to find optimal balance between data fitting and
    physics constraints.
    """

    def _grid_search(self, weight_ranges: dict[str, list[float]]) -> list[dict[str, float]]:
        """Generate all combinations of weights for grid search.

        Args:
            weight_ranges: Dictionary mapping weight names to lists of values.
                Example: {"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [0.1, 1.0]}

        Returns:
            List of weight dictionaries, one for each grid point.
            For example, with the above input, returns 3×2×2=12 combinations.

        Example:
            >>> framework = WeightTuningFrameworkService()
            >>> ranges = {"data": [0.1, 1.0], "pde": [0.5, 1.0], "bc": [0.1, 1.0]}
            >>> combos = framework._grid_search(ranges)
            >>> len(combos)
            8
        """
        keys = list(weight_ranges.keys())
        values = [weight_ranges[key] for key in keys]

        # Generate Cartesian product of all weight combinations
        combinations = []
        for combo in itertools.product(*values):
            weight_dict = dict(zip(keys, combo))
            combinations.append(weight_dict)

        return combinations

    def _random_search(
        self,
        weight_ranges: dict[str, list[float]],
        n_samples: int,
        seed: int | None = None,
    ) -> list[dict[str, float]]:
        """Generate random samples of weight combinations.

        Args:
            weight_ranges: Dictionary mapping weight names to lists of values.
            n_samples: Number of random samples to generate.
            seed: Random seed for reproducibility (optional).

        Returns:
            List of n_samples weight dictionaries with randomly sampled values.

        Example:
            >>> framework = WeightTuningFrameworkService()
            >>> ranges = {"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [0.1, 1.0]}
            >>> combos = framework._random_search(ranges, n_samples=50, seed=42)
            >>> len(combos)
            50
        """
        if seed is not None:
            random.seed(seed)

        combinations = []
        for _ in range(n_samples):
            weight_dict = {key: random.choice(values) for key, values in weight_ranges.items()}
            combinations.append(weight_dict)

        return combinations

    def run_tuning(
        self,
        base_config: ExperimentConfig,
        tuning_config: TuningConfig,
        train_fn: Any,  # Callable that trains model and returns validation error
    ) -> tuple[TuningResult, list[TuningResult]]:
        """Execute grid or random search over loss weight space.

        Args:
            base_config: Base experiment configuration (domain, network, etc.).
                Loss weights will be overridden by tuning combinations.
            tuning_config: Tuning-specific configuration (search type, ranges).
            train_fn: Callable that takes ExperimentConfig and returns
                (validation_error, training_time) tuple.

        Returns:
            Tuple of (best_result, all_results) where:
                - best_result: Configuration with lowest validation error
                - all_results: List of all tuning results

        Raises:
            ValueError: If tuning_config.search_type is invalid.

        Example:
            >>> def train_mock(config):
            ...     # Mock training function
            ...     return (0.05, 100.0)  # (validation_error, training_time)
            >>> framework = WeightTuningFrameworkService()
            >>> base_cfg = ExperimentConfig(...)
            >>> tuning_cfg = TuningConfig(search_type="grid", weight_ranges={...})
            >>> best, all_results = framework.run_tuning(base_cfg, tuning_cfg, train_mock)
        """
        # Generate weight combinations based on search type
        if tuning_config.search_type == "grid":
            weight_combinations = self._grid_search(tuning_config.weight_ranges)
        elif tuning_config.search_type == "random":
            weight_combinations = self._random_search(
                tuning_config.weight_ranges,
                tuning_config.n_samples,
            )
        else:
            raise ValueError(
                f"Invalid search_type: {tuning_config.search_type}. Must be 'grid' or 'random'."
            )

        print(f"Starting {tuning_config.search_type} search with {len(weight_combinations)} configurations...")

        # Train model with each weight combination
        all_results: list[TuningResult] = []
        for i, weights in enumerate(weight_combinations, start=1):
            print(f"Training configuration {i}/{len(weight_combinations)}: {weights}")

            # Update config with new weights
            config = base_config.model_copy(deep=True)
            config.training.loss_weights = weights

            # Train and evaluate
            start_time = time.time()
            validation_error, _ = train_fn(config)
            training_time = time.time() - start_time

            # Store result
            result = TuningResult(
                w_data=weights.get("data", 1.0),
                w_pde=weights.get("pde", 1.0),
                w_bc=weights.get("bc", 1.0),
                validation_error=validation_error,
                training_time=training_time,
            )
            all_results.append(result)

            print(f"  Validation error: {validation_error:.6f}, Time: {training_time:.2f}s")

        # Identify best configuration
        best_result = min(all_results, key=lambda r: r.validation_error)
        print("\nBest configuration found:")
        print(f"  w_data={best_result.w_data}, w_pde={best_result.w_pde}, w_bc={best_result.w_bc}")
        print(f"  Validation error: {best_result.validation_error:.6f}")

        # Save results
        self._save_results(all_results, tuning_config.output_path)

        return best_result, all_results

    def _save_results(self, results: list[TuningResult], output_path: Path) -> None:
        """Save tuning results to JSON or CSV file.

        Args:
            results: List of tuning results to save.
            output_path: Path to output file. Format determined by extension
                (.json or .csv).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            # Save as JSON
            results_dict = [asdict(r) for r in results]
            with open(output_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to {output_path}")

        elif output_path.suffix == ".csv":
            # Save as CSV using pandas
            df = pd.DataFrame([asdict(r) for r in results])
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")

        else:
            # Default to JSON if extension not recognized
            results_dict = [asdict(r) for r in results]
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to {json_path} (defaulted to JSON format)")

    def visualize_results(
        self,
        results: list[TuningResult],
        output_dir: Path,
    ) -> None:
        """Generate loss landscape and Pareto frontier plots.

        Creates two visualizations:
        1. Loss landscape heatmap (2D projection of weight space vs. validation error)
        2. Pareto frontier plot (validation error vs. training time trade-off)

        Args:
            results: List of tuning results to visualize.
            output_dir: Directory to save plot files.

        Saves:
            - loss_landscape.png: Heatmap of validation error across weight space
            - pareto_frontier.png: Scatter plot of error vs. time trade-off
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame([asdict(r) for r in results])

        # Plot 1: Loss landscape (w_data vs w_pde, colored by validation error)
        self._plot_loss_landscape(df, output_dir / "loss_landscape.png")

        # Plot 2: Pareto frontier (validation error vs. training time)
        self._plot_pareto_frontier(df, output_dir / "pareto_frontier.png")

        print(f"Visualizations saved to {output_dir}")

    def _plot_loss_landscape(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot loss landscape heatmap."""
        plt.figure(figsize=(10, 8))

        # For 2D heatmap, fix w_bc at most common value and plot w_data vs w_pde
        most_common_bc = df["w_bc"].mode()[0]
        df_filtered = df[df["w_bc"] == most_common_bc]

        # Pivot table for heatmap
        pivot_table = df_filtered.pivot_table(
            values="validation_error",
            index="w_pde",
            columns="w_data",
            aggfunc="mean",  # Average if duplicates
        )

        # Create heatmap
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".4f",
            cmap="viridis",
            cbar_kws={"label": "Validation Error"},
        )

        plt.title(f"Loss Landscape (w_bc={most_common_bc})")
        plt.xlabel("w_data")
        plt.ylabel("w_pde")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _plot_pareto_frontier(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot Pareto frontier (validation error vs. training time)."""
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(
            df["training_time"],
            df["validation_error"],
            c=df["w_data"],
            s=100,
            alpha=0.6,
            cmap="coolwarm",
        )

        plt.colorbar(label="w_data")
        plt.xlabel("Training Time (s)")
        plt.ylabel("Validation Error")
        plt.title("Pareto Frontier: Error vs. Training Time Trade-off")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
