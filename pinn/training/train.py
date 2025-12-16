#!/usr/bin/env python3
"""Main training script for 1D PINN wave equation models.

This script provides a command-line interface for training PINN models using
DeepXDE. It handles:
- Configuration loading and validation
- Model building and compilation
- Training execution with monitoring
- Results saving and visualization

Usage:
    python pinn/training/train.py <config_file.yaml>

Example:
    python pinn/training/train.py configs/standing_wave_example.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.training.callbacks import (
    CheckpointCallback,
    LossLoggingCallback,
    ValidationCallback,
)
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.utils.config_loader import ConfigLoaderService
from pinn.utils.experiment_manager import ExperimentManager
from pinn.utils.metadata_logger import MetadataLogger
from pinn.utils.seed_manager import SeedManager
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.validation.error_metrics import ErrorMetricsService
from pinn.validation.plot_generator import PlotGeneratorService


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


def main():
    """Main entry point for training script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train 1D PINN for wave equation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pinn/training/train.py configs/standing_wave_example.yaml
  python pinn/training/train.py configs/traveling_wave_example.yaml
        """,
    )
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--output-dir", type=str, default="experiments", help="Base directory for experiment outputs (default: experiments/)"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Load configuration
        print(f"Loading configuration from {args.config_file}...")
        config = ConfigLoaderService.load_config(str(config_path))
        print(f"✓ Configuration loaded: {config.experiment_name}")

        # Set random seeds for reproducibility
        print(f"Setting random seed: {config.seed}")
        SeedManager.set_seed(config.seed, verbose=True)

        # Create experiment directory
        print("Creating experiment directory...")
        experiment_manager = ExperimentManager(base_dir=args.output_dir)
        exp_dir = experiment_manager.create_experiment_directory(config.experiment_name)
        print(f"✓ Experiment directory: {exp_dir}")

        # Save configuration to experiment directory
        config_save_path = exp_dir / "config.yaml"
        ConfigLoaderService.save_config(config, str(config_save_path))
        print(f"✓ Configuration saved to: {config_save_path}")

        # Log metadata (software versions, seed, etc.)
        print("Logging metadata...")
        metadata_logger = MetadataLogger()
        metadata_path = exp_dir / "metadata.json"
        metadata = metadata_logger.capture_full_metadata(config)
        metadata_logger.save_metadata(metadata, metadata_path)
        print(f"✓ Metadata saved to: {metadata_path}")

        # Build PINN model
        print("\nBuilding PINN model...")
        model_builder = PINNModelBuilderService()
        initial_condition = create_initial_condition(config)
        model = model_builder.build_model(config, initial_condition, compile_model=True)
        print("✓ PINN model built and compiled")
        print(
            f"  - Domain: x ∈ [{config.domain.x_min}, {config.domain.x_max}], "
            f"t ∈ [{config.domain.t_min}, {config.domain.t_max}]"
        )
        print(f"  - Wave speed: c = {config.domain.wave_speed}")
        print(f"  - Network architecture: {config.network.layer_sizes}")
        print(f"  - Optimizer: {config.training.optimizer}")
        print(
            f"  - Loss weights: data={config.training.loss_weights['data']}, "
            f"pde={config.training.loss_weights['pde']}, "
            f"bc={config.training.loss_weights['bc']}"
        )

        # Create callbacks
        print("\nSetting up training callbacks...")
        callbacks = []

        # Loss logging callback
        loss_callback = LossLoggingCallback(log_interval=100)
        callbacks.append(loss_callback)
        print("  ✓ Loss logging callback (every 100 epochs)")

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            output_dir=exp_dir / "checkpoints", save_interval=config.training.checkpoint_interval
        )
        callbacks.append(checkpoint_callback)
        print(f"  ✓ Checkpoint callback (every {config.training.checkpoint_interval} epochs)")

        # Validation callback (for analytical solution comparison)
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
                "t_max": config.domain.t_max,
            },
            wave_speed=config.domain.wave_speed,
        )
        callbacks.append(validation_callback)
        print("  ✓ Validation callback (every 500 epochs)")

        # Train model
        print(f"\n{'=' * 60}")
        print(f"Starting training for {config.training.epochs} epochs...")
        print(f"{'=' * 60}\n")

        training_pipeline = TrainingPipelineService()
        trained_model, history = training_pipeline.train(model, config.training, exp_dir, callbacks=callbacks)

        print(f"\n{'=' * 60}")
        print("Training completed!")
        print(f"{'=' * 60}\n")

        # Generate training curves
        print("Generating training visualizations...")
        plot_generator = PlotGeneratorService()

        # Plot loss curves
        if history:
            loss_plot_path = exp_dir / "plots" / "training_curves.png"
            plot_generator.plot_training_curves(history, str(loss_plot_path))
            print(f"  ✓ Training curves saved to: {loss_plot_path}")

        # Final validation
        if hasattr(validation_callback, "errors") and validation_callback.errors:
            final_error = validation_callback.errors[-1]
            print(f"\nFinal validation L2 error: {final_error:.6f}")

            if final_error < 0.05:
                print("✓ Model meets <5% error threshold!")
            else:
                print("⚠ Model exceeds 5% error threshold - consider tuning hyperparameters")

        # Summary
        print(f"\n{'=' * 60}")
        print("Training Summary")
        print(f"{'=' * 60}")
        print(f"Experiment: {config.experiment_name}")
        print(f"Output directory: {exp_dir}")
        print(f"Epochs trained: {config.training.epochs}")
        print(f"Configuration saved: {config_save_path}")
        print(f"Metadata saved: {metadata_path}")
        print(f"{'=' * 60}\n")

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Training failed with exception: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
