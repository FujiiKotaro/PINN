"""
Automated experiment runner for PINN wave equation experiments.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
import time
from datetime import datetime

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.training.callbacks import LossLoggingCallback, ValidationCallback
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.validation.error_metrics import ErrorMetricsService
from pinn.validation.plot_generator import PlotGeneratorService
from pinn.utils.config_loader import ConfigLoaderService
from pinn.utils.seed_manager import SeedManager
from pinn.utils.experiment_manager import ExperimentManager

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def create_initial_condition(config):
    """Create initial condition function based on analytical solution type."""
    L = config.domain.x_max - config.domain.x_min

    if config.analytical_solution is not None:
        solution_type = config.analytical_solution.solution_type
        n_mode = config.analytical_solution.mode
        amplitude = config.analytical_solution.initial_amplitude
    else:
        if config.boundary_conditions.type == "dirichlet":
            solution_type = "standing_wave"
        elif config.boundary_conditions.type == "neumann":
            solution_type = "standing_wave_neumann"
        else:
            solution_type = "standing_wave"
        n_mode = 1
        amplitude = 1.0

    if solution_type == "standing_wave":
        def initial_condition(x):
            return amplitude * np.sin(n_mode * np.pi * x[:, 0:1] / L)
        description = f"sin({n_mode}πx/L)"

    elif solution_type == "standing_wave_neumann":
        def initial_condition(x):
            return amplitude * np.cos(n_mode * np.pi * x[:, 0:1] / L)
        description = f"cos({n_mode}πx/L)"

    elif solution_type == "traveling_wave":
        x0 = L / 2
        sigma = L / 10

        def initial_condition(x):
            if x.ndim == 1:
                x_reshaped = x.reshape(-1, 1)
            else:
                x_reshaped = x if x.shape[1] >= 1 else x.reshape(-1, 1)
            return amplitude * np.exp(-((x_reshaped[:, 0:1] - x0) ** 2) / (2 * sigma**2))
        description = f"Gaussian pulse at x={x0}"

    else:
        raise ValueError(f"Unknown solution_type: {solution_type}")

    return initial_condition, description, n_mode, solution_type


def run_experiment(config_path, epochs=None):
    """Run a single PINN experiment with the given configuration."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {config_path.name}")
    print(f"{'='*80}\n")

    # Load configuration
    config_loader = ConfigLoaderService()
    config = config_loader.load_config(config_path)

    # Override epochs if specified
    if epochs is not None:
        config.training.epochs = epochs

    # Set random seed
    SeedManager.set_seed(config.seed)

    # Create initial condition
    initial_condition, ic_description, n_mode, solution_type = create_initial_condition(config)
    L = config.domain.x_max - config.domain.x_min

    print(f"Experiment: {config.experiment_name}")
    print(f"Solution type: {solution_type}")
    print(f"Boundary conditions: {config.boundary_conditions.type}")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Network: {config.network.layer_sizes}")
    print(f"Loss weights: {config.training.loss_weights}\n")

    # Build PINN model
    model_builder = PINNModelBuilderService()
    model = model_builder.build_model(
        config=config,
        initial_condition_func=initial_condition,
        compile_model=True
    )

    # Create experiment directory
    exp_manager = ExperimentManager(base_dir=project_root / "experiments")
    exp_dir = exp_manager.create_experiment_directory(config.experiment_name)

    # Create callbacks
    analytical_solver = AnalyticalSolutionGeneratorService()
    error_metrics = ErrorMetricsService()

    if solution_type == "traveling_wave":
        bc_type_for_validation = "traveling_wave"
    else:
        bc_type_for_validation = config.boundary_conditions.type

    enable_validation = True
    if config.analytical_solution and hasattr(config.analytical_solution, "enable_validation"):
        enable_validation = config.analytical_solution.enable_validation

    loss_callback = LossLoggingCallback(log_interval=100)
    validation_callback = ValidationCallback(
        analytical_solver=analytical_solver,
        error_metrics=error_metrics,
        validation_interval=500,
        domain_config=config.domain,
        wave_speed=config.domain.wave_speed,
        n_mode=n_mode,
        bc_type=bc_type_for_validation,
        initial_condition_func=initial_condition if solution_type == "traveling_wave" else None,
        enable_validation=enable_validation,
    )

    callbacks = [loss_callback, validation_callback]

    # Train model
    training_pipeline = TrainingPipelineService()
    start_time = time.time()

    trained_model, training_history = training_pipeline.train(
        model=model,
        config=config.training,
        output_dir=exp_dir,
        callbacks=callbacks
    )

    training_time = time.time() - start_time

    # Generate predictions
    nx = 100
    nt = 5
    x_test = np.linspace(config.domain.x_min, config.domain.x_max, nx)
    t_test = np.linspace(config.domain.t_min, config.domain.t_max, nt)

    X, T = np.meshgrid(x_test, t_test, indexing="ij")
    X_flat = X.flatten()[:, None]
    T_flat = T.flatten()[:, None]
    XT = np.hstack([X_flat, T_flat])
    u_pinn_flat = trained_model.predict(XT)
    u_pinn = u_pinn_flat.reshape(nx, nt)

    # Compute error metrics if validation is enabled
    results = {
        "config_name": config_path.name,
        "experiment_name": config.experiment_name,
        "solution_type": solution_type,
        "boundary_condition": config.boundary_conditions.type,
        "epochs": config.training.epochs,
        "training_time": training_time,
        "network_architecture": config.network.layer_sizes,
        "loss_weights": config.training.loss_weights,
        "enable_validation": enable_validation,
    }

    if enable_validation:
        # Generate analytical solution
        if solution_type == "standing_wave":
            u_analytical = analytical_solver.standing_wave(
                x=x_test, t=t_test, L=L, c=config.domain.wave_speed, n=n_mode
            )
        elif solution_type == "standing_wave_neumann":
            u_analytical = analytical_solver.standing_wave_neumann(
                x=x_test, t=t_test, L=L, c=config.domain.wave_speed, n=n_mode
            )
        elif solution_type == "traveling_wave":
            u_analytical = analytical_solver.traveling_wave(
                x=x_test, t=t_test, c=config.domain.wave_speed, initial_condition=initial_condition
            )

        # Compute errors
        l2_error = error_metrics.l2_error(u_pinn, u_analytical)
        relative_error = error_metrics.relative_error(u_pinn, u_analytical)
        max_error = error_metrics.max_absolute_error(u_pinn, u_analytical)

        results.update({
            "l2_error": float(l2_error),
            "relative_error": float(relative_error),
            "max_error": float(max_error),
            "validation_passed": relative_error < 0.05,
        })

        print(f"\nValidation Results:")
        print(f"  L2 Error: {l2_error:.6f}")
        print(f"  Relative Error: {relative_error:.6f} ({relative_error * 100:.2f}%)")
        print(f"  Max Error: {max_error:.6f}")
        print(f"  Validation: {'PASSED' if relative_error < 0.05 else 'FAILED'}")

        # Plot comparison
        plot_generator = PlotGeneratorService()
        time_snapshots = [0.0, 0.25, 0.5, 0.75, 1.0]
        u_pinn_dict = {}
        u_analytical_dict = {}

        for t_val in time_snapshots:
            t_idx = np.argmin(np.abs(t_test - t_val))
            u_pinn_dict[t_val] = u_pinn[:, t_idx]
            u_analytical_dict[t_val] = u_analytical[:, t_idx]

        fig = plot_generator.plot_solution_comparison(
            x=x_test,
            time_snapshots=time_snapshots,
            u_pinn=u_pinn_dict,
            u_analytical=u_analytical_dict,
            save_path=exp_dir / "solution_comparison.png",
        )
        plt.close(fig)

    # Plot training curves
    if training_history:
        plot_generator = PlotGeneratorService()
        fig = plot_generator.plot_training_curves(
            training_history,
            save_path=exp_dir / "training_curves.png"
        )
        plt.close(fig)

        # Extract final losses
        if training_history.get("loss_components"):
            final_losses = training_history["loss_components"][-1]
            results["final_total_loss"] = float(final_losses.get("total", 0))
            results["final_pde_loss"] = float(final_losses.get("pde", 0))
            results["final_bc_loss"] = float(final_losses.get("bc", 0))
            results["final_ic_loss"] = float(final_losses.get("ic", 0))

    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Results saved to: {exp_dir}")

    return results, exp_dir


def run_all_experiments():
    """Run experiments with different configurations."""

    configs_dir = project_root / "configs"

    # Experiment configurations to test
    experiments = [
        {
            "config": configs_dir / "standing_wave_example.yaml",
            "name": "Standing Wave (Dirichlet BC)",
            "epochs": 8000
        },
        {
            "config": configs_dir / "neumann_bc_example.yaml",
            "name": "Neumann BC",
            "epochs": 8000
        },
        {
            "config": configs_dir / "traveling_wave_example.yaml",
            "name": "Traveling Wave",
            "epochs": 8000
        },
    ]

    # Also test different epoch counts
    epoch_experiments = [
        {
            "config": configs_dir / "standing_wave_example.yaml",
            "name": "Standing Wave - 4000 epochs",
            "epochs": 4000
        },
        {
            "config": configs_dir / "standing_wave_example.yaml",
            "name": "Standing Wave - 12000 epochs",
            "epochs": 12000
        },
    ]

    all_results = []

    print(f"\n{'#'*80}")
    print("PINN EXPERIMENT SUITE")
    print(f"{'#'*80}\n")
    print(f"Total experiments: {len(experiments) + len(epoch_experiments)}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Run main experiments
    for exp in experiments:
        try:
            results, exp_dir = run_experiment(exp["config"], exp["epochs"])
            results["experiment_group"] = "configuration_comparison"
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Run epoch experiments
    for exp in epoch_experiments:
        try:
            results, exp_dir = run_experiment(exp["config"], exp["epochs"])
            results["experiment_group"] = "epoch_comparison"
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Save all results
    results_file = project_root / "experiments" / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#'*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'#'*80}\n")
    print(f"Total experiments run: {len(all_results)}")
    print(f"Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80 + "\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['experiment_name']}")
        print(f"   Config: {result['config_name']}")
        print(f"   Epochs: {result['epochs']}, Time: {result['training_time']:.1f}s")
        if result.get('enable_validation'):
            print(f"   Relative Error: {result.get('relative_error', 'N/A'):.4f}")
            print(f"   Validation: {'PASSED' if result.get('validation_passed') else 'FAILED'}")
        else:
            print(f"   Validation: DISABLED")
        print()
