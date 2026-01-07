"""
Traveling Wave PINN Analysis with Reflected d'Alembert Solution

このスクリプトは反射を含むd'Alembert解を使用して、
有界領域でのtraveling waveを正しく解析します。
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

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
plt.rcParams["figure.figsize"] = (14, 10)

print("="*80)
print("TRAVELING WAVE PINN - 反射を考慮したd'Alembert解析")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Load configuration
config_path = project_root / "configs" / "traveling_wave_example.yaml"
config_loader = ConfigLoaderService()
config = config_loader.load_config(config_path)

# Override settings
config.training.epochs = 8000

# Set seed
SeedManager.set_seed(config.seed)

print("Configuration:")
print(f"  Experiment: {config.experiment_name}")
print(f"  Domain: x ∈ [{config.domain.x_min}, {config.domain.x_max}]")
print(f"          t ∈ [{config.domain.t_min}, {config.domain.t_max}]")
print(f"  Wave speed: c = {config.domain.wave_speed}")
print(f"  Boundary condition: {config.boundary_conditions.type}")
print(f"  Network: {config.network.layer_sizes}")
print()

# Define initial condition (Gaussian pulse)
L = config.domain.x_max - config.domain.x_min
x0 = L / 2
sigma = L / 10
amplitude = config.analytical_solution.initial_amplitude

def initial_condition(x):
    """Gaussian pulse initial condition."""
    if x.ndim == 1:
        x_reshaped = x.reshape(-1, 1)
    else:
        x_reshaped = x if x.shape[1] >= 1 else x.reshape(-1, 1)
    return amplitude * np.exp(-((x_reshaped[:, 0:1] - x0) ** 2) / (2 * sigma**2))

print(f"Initial condition: Gaussian pulse")
print(f"  Center: x0 = {x0}")
print(f"  Width: σ = {sigma}")
print(f"  Amplitude: A = {amplitude}")
print()

# Calculate reflection time
c = config.domain.wave_speed
distance_to_boundary = min(x0 - config.domain.x_min, config.domain.x_max - x0)
t_reflection = distance_to_boundary / c

print(f"Wave propagation analysis:")
print(f"  Wave speed: c = {c}")
print(f"  Distance to nearest boundary: {distance_to_boundary}")
print(f"  Time to reach boundary: t_reflection ≈ {t_reflection:.3f}")
print(f"\n✓ d'Alembert解に反射を追加 → 全時刻で比較可能！")
print()

# Build PINN model
print("Building PINN model...")
model_builder = PINNModelBuilderService()
model = model_builder.build_model(
    config=config,
    initial_condition_func=initial_condition,
    compile_model=True
)

total_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
print()

# Create experiment directory
exp_manager = ExperimentManager(base_dir=project_root / "experiments")
exp_dir = exp_manager.create_experiment_directory("traveling_wave_reflected")

print(f"Experiment directory: {exp_dir}")
print()

# Setup callbacks (validation disabled during training for now)
print("Setting up callbacks...")

analytical_solver = AnalyticalSolutionGeneratorService()
error_metrics = ErrorMetricsService()

loss_callback = LossLoggingCallback(log_interval=100)
callbacks = [loss_callback]

print(f"Callbacks created: {len(callbacks)}")
print()

# Train
print("="*80)
print("STARTING TRAINING")
print("="*80)

training_pipeline = TrainingPipelineService()
start_time = time.time()

trained_model, training_history = training_pipeline.train(
    model=model,
    config=config.training,
    output_dir=exp_dir,
    callbacks=callbacks
)

training_time = time.time() - start_time

print()
print("="*80)
print("TRAINING COMPLETED")
print("="*80)
print(f"Training time: {training_time:.2f} seconds")
print()

# Generate predictions
print("Generating predictions for analysis...")

nx = 200
nt = 20
x_test = np.linspace(config.domain.x_min, config.domain.x_max, nx)
t_test = np.linspace(config.domain.t_min, config.domain.t_max, nt)

X, T = np.meshgrid(x_test, t_test, indexing="ij")
X_flat = X.flatten()[:, None]
T_flat = T.flatten()[:, None]
XT = np.hstack([X_flat, T_flat])

# PINN predictions
u_pinn_flat = trained_model.predict(XT)
u_pinn = u_pinn_flat.reshape(nx, nt)

# Analytical solution WITHOUT reflections (for comparison)
u_analytical_no_reflection = analytical_solver.traveling_wave(
    x=x_test,
    t=t_test,
    c=config.domain.wave_speed,
    initial_condition=initial_condition
)

# Analytical solution WITH reflections (NEW!)
print("Computing d'Alembert solution with reflections...")
u_analytical_with_reflection = analytical_solver.traveling_wave_with_reflections(
    x=x_test,
    t=t_test,
    c=config.domain.wave_speed,
    initial_condition=initial_condition,
    L=L,
    n_reflections=10  # Include up to 10 reflections
)

print(f"Generated predictions on {nx} × {nt} = {nx*nt} points")
print()

# Compute time-dependent errors (with reflections)
print("Computing time-dependent errors (with reflections)...")

errors_per_time = []
relative_errors_per_time = []
max_errors_per_time = []

for i in range(nt):
    u_pinn_t = u_pinn[:, i].reshape(-1, 1)
    u_analytical_t = u_analytical_with_reflection[:, i].reshape(-1, 1)

    l2_err = error_metrics.l2_error(u_pinn_t, u_analytical_t)
    rel_err = error_metrics.relative_error(u_pinn_t, u_analytical_t)
    max_err = error_metrics.max_absolute_error(u_pinn_t, u_analytical_t)

    errors_per_time.append(l2_err)
    relative_errors_per_time.append(rel_err)
    max_errors_per_time.append(max_err)

# Overall metrics
overall_l2 = error_metrics.l2_error(u_pinn, u_analytical_with_reflection)
overall_rel = error_metrics.relative_error(u_pinn, u_analytical_with_reflection)
overall_max = error_metrics.max_absolute_error(u_pinn, u_analytical_with_reflection)

# Print results
print()
print("="*80)
print("TRAVELING WAVE PINN ANALYSIS - 反射を考慮した結果")
print("="*80)

print("\n【Overall Error Metrics (反射を含むd'Alembert解との比較)】")
print(f"  L2 Error:        {overall_l2:.6f}")
print(f"  Relative Error:  {overall_rel:.6f} ({overall_rel*100:.2f}%)")
print(f"  Max Error:       {overall_max:.6f}")

# Time-based statistics
mean_l2 = np.mean(errors_per_time)
mean_rel = np.mean(relative_errors_per_time)
max_err_overall = np.max(max_errors_per_time)

print(f"\n【時間平均誤差】")
print(f"  Mean L2 error:       {mean_l2:.6f}")
print(f"  Mean relative error: {mean_rel:.6f} ({mean_rel*100:.2f}%)")
print(f"  Max error (any time): {max_err_overall:.6f}")

if mean_rel < 0.05:
    print(f"\n  ✓ 5%目標を達成！反射を考慮すれば正しく評価できます")
else:
    print(f"\n  ⚠ 平均相対誤差 {mean_rel*100:.2f}% > 5%目標")
    print(f"     初期条件の損失ウェイト調整が必要")

print("\n" + "="*80)

# Generate visualizations
print("\nGenerating visualizations...")

# 1. Error vs time plot (with reflections)
try:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(t_test, errors_per_time, 'b-o', linewidth=2, markersize=4)
    axes[0].axvline(t_reflection, color='r', linestyle='--', label=f'Reflection time ≈ {t_reflection:.3f}')
    axes[0].set_xlabel('Time (t)', fontsize=12)
    axes[0].set_ylabel('L2 Error', fontsize=12)
    axes[0].set_title('L2 Error vs Time (反射を含むd\'Alembert解)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_test, relative_errors_per_time, 'g-o', linewidth=2, markersize=4)
    axes[1].axvline(t_reflection, color='r', linestyle='--', label=f'Reflection time ≈ {t_reflection:.3f}')
    axes[1].axhline(0.05, color='orange', linestyle=':', label='5% threshold')
    axes[1].set_xlabel('Time (t)', fontsize=12)
    axes[1].set_ylabel('Relative Error', fontsize=12)
    axes[1].set_title('Relative Error vs Time (反射を含むd\'Alembert解)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t_test, max_errors_per_time, 'm-o', linewidth=2, markersize=4)
    axes[2].axvline(t_reflection, color='r', linestyle='--', label=f'Reflection time ≈ {t_reflection:.3f}')
    axes[2].set_xlabel('Time (t)', fontsize=12)
    axes[2].set_ylabel('Max Absolute Error', fontsize=12)
    axes[2].set_title('Maximum Absolute Error vs Time (反射を含むd\'Alembert解)', fontsize=14)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(exp_dir / "error_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Error vs time plot saved")
except Exception as e:
    print(f"  ✗ Error saving error vs time plot: {e}")

# 2. Solution evolution comparison
try:
    n_snapshots = 8
    time_indices = np.linspace(0, nt-1, n_snapshots, dtype=int)
    time_snapshots = t_test[time_indices]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, (t_idx, t_val) in enumerate(zip(time_indices, time_snapshots)):
        ax = axes[i]

        ax.plot(x_test, u_analytical_with_reflection[:, t_idx], 'b-', linewidth=2.5,
                label="d'Alembert (反射込)", alpha=0.7)
        ax.plot(x_test, u_pinn[:, t_idx], 'r--', linewidth=2, label='PINN', alpha=0.7)
        ax.plot(x_test, u_analytical_no_reflection[:, t_idx], 'g:', linewidth=1.5,
                label="d'Alembert (反射無)", alpha=0.5)

        if t_val >= t_reflection:
            ax.set_facecolor('#f0fff0')
            title_suffix = " (反射後)"
        else:
            title_suffix = ""

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u(x, t)', fontsize=10)
        ax.set_title(f't = {t_val:.3f}{title_suffix}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.2, 1.2])

    plt.tight_layout()
    plt.savefig(exp_dir / "solution_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Solution evolution plot saved")
except Exception as e:
    print(f"  ✗ Error saving solution evolution: {e}")

# 3. Spatiotemporal heatmap comparison
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # PINN
    im1 = axes[0, 0].imshow(u_pinn.T, aspect='auto', origin='lower', cmap='RdBu_r',
                             extent=[config.domain.x_min, config.domain.x_max,
                                    config.domain.t_min, config.domain.t_max],
                             vmin=-0.5, vmax=1.0)
    axes[0, 0].axhline(t_reflection, color='yellow', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('t', fontsize=12)
    axes[0, 0].set_title('PINN Solution', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    # d'Alembert with reflections
    im2 = axes[0, 1].imshow(u_analytical_with_reflection.T, aspect='auto', origin='lower', cmap='RdBu_r',
                             extent=[config.domain.x_min, config.domain.x_max,
                                    config.domain.t_min, config.domain.t_max],
                             vmin=-0.5, vmax=1.0)
    axes[0, 1].axhline(t_reflection, color='yellow', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_ylabel('t', fontsize=12)
    axes[0, 1].set_title("d'Alembert Solution (反射込)", fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1])

    # d'Alembert without reflections
    im3 = axes[1, 0].imshow(u_analytical_no_reflection.T, aspect='auto', origin='lower', cmap='RdBu_r',
                             extent=[config.domain.x_min, config.domain.x_max,
                                    config.domain.t_min, config.domain.t_max],
                             vmin=-0.5, vmax=1.0)
    axes[1, 0].axhline(t_reflection, color='yellow', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].set_ylabel('t', fontsize=12)
    axes[1, 0].set_title("d'Alembert Solution (反射無・従来)", fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0])

    # Error (with reflections)
    error_map = np.abs(u_pinn - u_analytical_with_reflection)
    im4 = axes[1, 1].imshow(error_map.T, aspect='auto', origin='lower', cmap='hot',
                             extent=[config.domain.x_min, config.domain.x_max,
                                    config.domain.t_min, config.domain.t_max])
    axes[1, 1].axhline(t_reflection, color='cyan', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('t', fontsize=12)
    axes[1, 1].set_title("Absolute Error (反射込d'Alembert)", fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(exp_dir / "spatiotemporal_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Spatiotemporal heatmap saved")
except Exception as e:
    print(f"  ✗ Error saving spatiotemporal heatmap: {e}")

print()

# Summary
print("="*80)
print("SUMMARY - 反射を考慮したd'Alembert解析")
print("="*80)

print("\n【改善点】")
print("  ✓ d'Alembert解に境界反射を追加（鏡像法）")
print("  ✓ 反射後も解析解と正しく比較可能")
print("  ✓ 全時刻で一貫した誤差評価")

print(f"\n【結果】")
print(f"  平均相対誤差: {mean_rel*100:.2f}%")
if mean_rel < 0.05:
    print(f"  🎉 5%目標達成！")
else:
    print(f"  ⚠ まだ5%目標未達成")
    print(f"     → 初期条件の損失ウェイト調整が推奨される")

print(f"\n【従来法との比較】")
print(f"  反射無しd'Alembert: 反射後は誤差が異常に増加")
print(f"  反射込みd'Alembert: 全時刻で適切な誤差評価 ✓")

print(f"\nResults saved to: {exp_dir}")
print("="*80)
