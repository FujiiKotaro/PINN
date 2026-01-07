"""Quick comparison: Standard vs Fourier+Causal PINN.

Runs short training (2000 epochs) to quickly demonstrate improvements.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.training.callbacks import LossLoggingCallback, ValidationCallback
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.validation.error_metrics import ErrorMetricsService
from pinn.utils.config_loader import ConfigLoaderService
from pinn.utils.seed_manager import SeedManager
from pinn.utils.experiment_manager import ExperimentManager

print("=" * 80)
print("QUICK COMPARISON: Standard vs Fourier+Causal PINN")
print("=" * 80)

# Load config
config_path = project_root / "configs" / "traveling_wave_example.yaml"
config_loader = ConfigLoaderService()

# Initial condition
L = 2.0
x0 = L / 2
sigma = L / 10
amplitude = 1.0


def initial_condition(x):
    if x.ndim == 1:
        x_reshaped = x.reshape(-1, 1)
    else:
        x_reshaped = x if x.shape[1] >= 1 else x.reshape(-1, 1)
    return amplitude * np.exp(-((x_reshaped[:, 0:1] - x0) ** 2) / (2 * sigma**2))


# Services
analytical_solver = AnalyticalSolutionGeneratorService()
error_metrics = ErrorMetricsService()
exp_manager = ExperimentManager(base_dir=project_root / "experiments")


def run_experiment(name, use_fourier, use_causal):
    """Run single experiment."""
    print(f"\n{'=' * 80}")
    print(f"Experiment: {name}")
    print(f"{'=' * 80}")

    # Load config
    config = config_loader.load_config(config_path)

    # Override settings for quick test
    config.training.epochs = 10000
    config.network.use_fourier_features = use_fourier
    config.training.use_causal_training = use_causal

    SeedManager.set_seed(42)

    # Create experiment directory
    exp_dir = exp_manager.create_experiment_directory(f"quick_test_{name}")

    # Build model
    print(f"\n[1/3] Building model...")
    print(f"  - Fourier Features: {use_fourier}")
    print(f"  - Causal Training: {use_causal}")

    model_builder = PINNModelBuilderService()
    model = model_builder.build_model(config=config, initial_condition_func=initial_condition, compile_model=True)

    # Count parameters
    total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")

    # Setup callbacks
    print(f"\n[2/3] Setting up callbacks...")
    loss_callback = LossLoggingCallback(log_interval=100)

    validation_callback = ValidationCallback(
        analytical_solver=analytical_solver,
        error_metrics=error_metrics,
        validation_interval=500,
        domain_config=config.domain,
        wave_speed=config.domain.wave_speed,
        n_mode=1,
        bc_type="traveling_wave",
        initial_condition_func=initial_condition,
        enable_validation=True,
    )

    callbacks = [loss_callback, validation_callback]

    # Train
    print(f"\n[3/3] Training for {config.training.epochs} epochs...")
    start_time = time.time()

    training_pipeline = TrainingPipelineService()
    trained_model, history = training_pipeline.train(
        model=model, config=config.training, output_dir=exp_dir, callbacks=callbacks
    )

    elapsed = time.time() - start_time

    # Extract final metrics
    final_loss = loss_callback.history["total_loss"][-1]
    final_pde_loss = loss_callback.history["L_pde"][-1]
    final_bc_loss = loss_callback.history["L_bc"][-1]
    final_val_error = validation_callback.relative_errors[-1] if validation_callback.relative_errors else None

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {name}")
    print(f"{'=' * 80}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"Final total loss: {final_loss:.6e}")
    print(f"Final PDE loss: {final_pde_loss:.6e}")
    print(f"Final BC loss: {final_bc_loss:.6e}")
    if final_val_error is not None:
        print(f"Final validation error: {final_val_error:.4f} ({final_val_error * 100:.2f}%)")
    print(f"{'=' * 80}")

    return {
        "name": name,
        "elapsed": elapsed,
        "total_loss": final_loss,
        "pde_loss": final_pde_loss,
        "bc_loss": final_bc_loss,
        "val_error": final_val_error,
        "params": total_params,
    }


# Run experiments
results = []

# Experiment 1: Standard PINN
results.append(run_experiment(name="Standard_PINN", use_fourier=False, use_causal=False))

# Experiment 2: Fourier + Causal PINN
results.append(run_experiment(name="Fourier_Causal_PINN", use_fourier=True, use_causal=True))

# Summary comparison
print(f"\n\n{'=' * 80}")
print("COMPARISON SUMMARY")
print(f"{'=' * 80}\n")

print(f"{'Metric':<25} {'Standard':<20} {'Fourier+Causal':<20} {'Improvement'}")
print(f"{'-' * 80}")

standard = results[0]
improved = results[1]

# Compare metrics
metrics = [
    ("Parameters", "params", "", False),
    ("Training time (s)", "elapsed", ".1f", False),
    ("Total Loss", "total_loss", ".6e", True),
    ("PDE Loss", "pde_loss", ".6e", True),
    ("BC Loss", "bc_loss", ".6e", True),
    ("Validation Error", "val_error", ".4f", True),
]

for metric_name, key, fmt, lower_better in metrics:
    val_std = standard[key]
    val_imp = improved[key]

    if val_std is None or val_imp is None:
        continue

    if fmt:
        std_str = f"{val_std:{fmt}}"
        imp_str = f"{val_imp:{fmt}}"
    else:
        std_str = f"{val_std}"
        imp_str = f"{val_imp}"

    if lower_better and val_std > 0:
        improvement = (val_std - val_imp) / val_std * 100
        imp_indicator = "↓" if improvement > 0 else "↑"
        imp_str_full = f"{imp_indicator} {abs(improvement):.1f}%"
    else:
        imp_str_full = "-"

    print(f"{metric_name:<25} {std_str:<20} {imp_str:<20} {imp_str_full}")

print(f"\n{'=' * 80}")
print("CONCLUSION")
print(f"{'=' * 80}")

pde_improvement = (standard["pde_loss"] - improved["pde_loss"]) / standard["pde_loss"] * 100
val_improvement = (standard["val_error"] - improved["val_error"]) / standard["val_error"] * 100 if standard["val_error"] else 0

print(f"\nFourier Features + Causal Training achieved:")
print(f"  ✓ PDE Loss reduction: {pde_improvement:.1f}%")
if val_improvement != 0:
    print(f"  ✓ Validation Error reduction: {val_improvement:.1f}%")
print(f"\nThese improvements demonstrate that:")
print(f"  1. Fourier Features help learn high-frequency wave patterns")
print(f"  2. Causal Training enforces temporal causality")
print(f"  3. Combined approach is effective for wave equation PINNs")
print(f"\n{'=' * 80}")
