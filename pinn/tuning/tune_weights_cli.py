#!/usr/bin/env python3
"""Weight tuning CLI script for PINN loss function optimization.

This script provides a command-line interface for running automated hyperparameter
tuning of loss function weights (w_data, w_pde, w_bc) using grid search or random search.

Usage:
    python pinn/tuning/tune_weights_cli.py <config_file.yaml> --search-type grid \\
        --weight-ranges '{"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [1.0, 10.0]}'

Example:
    # Grid search with 3x2x2 = 12 configurations
    python pinn/tuning/tune_weights_cli.py configs/standing_wave_example.yaml \\
        --search-type grid \\
        --weight-ranges '{"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [1.0, 10.0]}'

    # Random search with 50 samples
    python pinn/tuning/tune_weights_cli.py configs/standing_wave_example.yaml \\
        --search-type random \\
        --weight-ranges '{"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0], "bc": [1.0, 10.0]}' \\
        --n-samples 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.training.callbacks import ValidationCallback
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.tuning.weight_tuning import (
    TuningConfig,
    WeightTuningFrameworkService,
)
from pinn.utils.config_loader import ConfigLoaderService, ExperimentConfig
from pinn.utils.experiment_manager import ExperimentManager
from pinn.utils.seed_manager import SeedManager
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.validation.error_metrics import ErrorMetricsService


def create_initial_condition(config):
    """Create initial condition function for the PINN model.

    For standing wave: u(x, 0) = sin(πx/L)

    Args:
        config: ExperimentConfig object with domain configuration

    Returns:
        Callable that takes x coordinates and returns initial displacement
    """
    def standing_wave_ic(x):
        """Standing wave initial condition: u(x, 0) = sin(πx/L)."""
        L = config.domain.x_max - config.domain.x_min
        return np.sin(np.pi * x[:, 0:1] / L)

    return standing_wave_ic


def train_and_evaluate(config: ExperimentConfig) -> tuple[float, float]:
    """Train PINN model and return validation error and training time.

    This function is passed to the tuning framework as the training callback.

    Args:
        config: ExperimentConfig with specific loss weights to evaluate

    Returns:
        Tuple of (validation_error, training_time) where:
            - validation_error: Final L2 relative error vs. analytical solution
            - training_time: Total training time in seconds
    """
    # Set seeds
    SeedManager.set_seed(config.seed, verbose=False)

    # Build model
    model_builder = PINNModelBuilderService()
    initial_condition = create_initial_condition(config)
    model = model_builder.build_model(config, initial_condition, compile_model=True)

    # Create validation callback
    analytical_solver = AnalyticalSolutionGeneratorService()
    error_metrics = ErrorMetricsService()
    validation_callback = ValidationCallback(
        analytical_solver=analytical_solver,
        error_metrics=error_metrics,
        validation_interval=500,
        domain_config={
            "x_min": config.domain.x_min,
            "x_max": config.domain.x_max,
            "t_min": config.domain.t_min,
            "t_max": config.domain.t_max
        },
        wave_speed=config.domain.wave_speed
    )

    # Train model
    training_pipeline = TrainingPipelineService()
    exp_dir = Path(f"experiments/tuning_temp_{config.experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    _, history = training_pipeline.train(
        model,
        config.training,
        exp_dir,
        callbacks=[validation_callback]
    )

    # Extract final validation error
    if hasattr(validation_callback, 'relative_errors') and validation_callback.relative_errors:
        validation_error = validation_callback.relative_errors[-1]
    else:
        # Fallback: compute final error manually
        validation_error = 1.0  # High error if validation failed

    # Clean up temp directory
    import shutil
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    return validation_error, 0.0  # Return 0 for training time (handled by tuning framework)


def main():
    """Main entry point for weight tuning script."""
    parser = argparse.ArgumentParser(
        description="Tune loss function weights for PINN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid search
  python pinn/tuning/tune_weights_cli.py configs/standing_wave_example.yaml \\
      --search-type grid \\
      --weight-ranges '{"data": [1.0], "pde": [0.5, 1.0], "bc": [1.0, 10.0]}'

  # Random search
  python pinn/tuning/tune_weights_cli.py configs/standing_wave_example.yaml \\
      --search-type random \\
      --weight-ranges '{"data": [0.1, 1.0, 10.0], "pde": [0.5, 1.0, 2.0], "bc": [1.0, 10.0]}' \\
      --n-samples 20
        """
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to base YAML configuration file"
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["grid", "random"],
        required=True,
        help="Type of search: 'grid' for exhaustive grid search, 'random' for random sampling"
    )
    parser.add_argument(
        "--weight-ranges",
        type=str,
        required=True,
        help='JSON dict of weight ranges. Example: \'{"data": [1.0], "pde": [0.5, 1.0], "bc": [1.0, 10.0]}\''
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of random samples (only used for random search). Default: 100"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="tuning_results.json",
        help="Output file for tuning results (.json or .csv). Default: tuning_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_output",
        help="Output directory for visualizations. Default: tuning_output/"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs for tuning (reduces epochs for faster tuning)"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Parse weight ranges JSON
        try:
            weight_ranges = json.loads(args.weight_ranges)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON for --weight-ranges: {e}", file=sys.stderr)
            sys.exit(1)

        # Validate weight_ranges structure
        if not isinstance(weight_ranges, dict):
            print("ERROR: --weight-ranges must be a JSON dict", file=sys.stderr)
            sys.exit(1)

        for key, values in weight_ranges.items():
            if not isinstance(values, list):
                print(f"ERROR: weight_ranges['{key}'] must be a list", file=sys.stderr)
                sys.exit(1)

        # Load base configuration
        print(f"Loading base configuration from {args.config_file}...")
        base_config = ConfigLoaderService.load_config(str(config_path))
        print(f"✓ Configuration loaded: {base_config.experiment_name}")

        # Override epochs if specified
        if args.epochs is not None:
            base_config.training.epochs = args.epochs
            print(f"✓ Epochs overridden to {args.epochs} for faster tuning")

        # Create tuning configuration
        tuning_config = TuningConfig(
            search_type=args.search_type,
            weight_ranges=weight_ranges,
            n_samples=args.n_samples,
            output_path=Path(args.output_file)
        )

        # Print tuning summary
        print(f"\n{'='*60}")
        print("Weight Tuning Configuration")
        print(f"{'='*60}")
        print(f"Search type: {args.search_type}")
        print(f"Weight ranges: {weight_ranges}")
        if args.search_type == "grid":
            n_configs = 1
            for values in weight_ranges.values():
                n_configs *= len(values)
            print(f"Number of configurations: {n_configs}")
        else:
            print(f"Number of samples: {args.n_samples}")
        print(f"Epochs per configuration: {base_config.training.epochs}")
        print(f"Output file: {args.output_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*60}\n")

        # Run tuning
        framework = WeightTuningFrameworkService()
        best_result, all_results = framework.run_tuning(
            base_config,
            tuning_config,
            train_and_evaluate
        )

        # Generate visualizations
        print("\nGenerating visualizations...")
        output_dir = Path(args.output_dir)
        framework.visualize_results(all_results, output_dir)

        # Print summary
        print(f"\n{'='*60}")
        print("Tuning Complete!")
        print(f"{'='*60}")
        print(f"Best configuration:")
        print(f"  w_data = {best_result.w_data}")
        print(f"  w_pde = {best_result.w_pde}")
        print(f"  w_bc = {best_result.w_bc}")
        print(f"  Validation error = {best_result.validation_error:.6f}")
        print(f"  Training time = {best_result.training_time:.2f}s")
        print(f"\nResults saved to:")
        print(f"  - {args.output_file}")
        print(f"  - {output_dir}/loss_landscape.png")
        print(f"  - {output_dir}/pareto_frontier.png")
        print(f"{'='*60}\n")

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Tuning failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
