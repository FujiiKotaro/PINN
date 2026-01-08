"""Check if PINN learns correct wave amplitude (should be 0.5 for split waves)."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pinn.models.pinn_model_builder import PINNModelBuilderService
from pinn.training.training_pipeline import TrainingPipelineService
from pinn.training.callbacks import LossLoggingCallback
from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService
from pinn.utils.config_loader import ConfigLoaderService
from pinn.utils.seed_manager import SeedManager
from pinn.utils.experiment_manager import ExperimentManager

# Parameters
L = 2.0
x0 = L / 2
sigma = L / 10
amplitude = 1.0
c = 1.5


def initial_condition(x):
    """Gaussian pulse initial condition."""
    if x.ndim == 1:
        x_reshaped = x.reshape(-1, 1)
    else:
        x_reshaped = x if x.shape[1] >= 1 else x.reshape(-1, 1)
    return amplitude * np.exp(-((x_reshaped[:, 0:1] - x0) ** 2) / (2 * sigma**2))


print("=" * 80)
print("CHECKING PINN AMPLITUDE PREDICTION")
print("=" * 80)

# Load config
config_path = project_root / "configs" / "traveling_wave_example.yaml"
config_loader = ConfigLoaderService()
config = config_loader.load_config(config_path)

# Train for 5000 epochs to see converged behavior
config.training.epochs = 5000
config.network.use_fourier_features = False
config.training.use_causal_training = False

SeedManager.set_seed(42)

# Build model
print("\n[1/2] Building and training PINN (5000 epochs)...")
model_builder = PINNModelBuilderService()
model = model_builder.build_model(
    config=config,
    initial_condition_func=initial_condition,
    compile_model=True
)

# Setup experiment
exp_manager = ExperimentManager(base_dir=project_root / "experiments")
exp_dir = exp_manager.create_experiment_directory("amplitude_check")

# Train
loss_callback = LossLoggingCallback(log_interval=100)
training_pipeline = TrainingPipelineService()

trained_model, _ = training_pipeline.train(
    model=model,
    config=config.training,
    output_dir=exp_dir,
    callbacks=[loss_callback]
)

# Check predictions at t=0.3 (before reflection, wave should have split)
print("\n[2/2] Analyzing predictions at t=0.3...")
t_check = 0.3
x_test = np.linspace(0, L, 200)
X, T = np.meshgrid(x_test, [t_check], indexing="ij")
XT = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])

# PINN prediction
u_pinn = trained_model.predict(XT).reshape(-1)

# Analytical solution
analytical_solver = AnalyticalSolutionGeneratorService()
u_analytical = analytical_solver.traveling_wave_with_reflections(
    x=x_test,
    t=np.array([t_check]),
    c=c,
    initial_condition=initial_condition,
    L=L,
    n_reflections=10,
)[:, 0]

# Amplitude analysis
pinn_max = np.max(np.abs(u_pinn))
analytical_max = np.max(np.abs(u_analytical))

print("\n" + "=" * 80)
print("AMPLITUDE ANALYSIS AT t=0.3 (Before Reflection)")
print("=" * 80)
print(f"\nInitial pulse amplitude: {amplitude}")
print(f"Expected split wave amplitude: {amplitude/2:.4f}\n")
print(f"PINN max amplitude:       {pinn_max:.4f}")
print(f"Analytical max amplitude: {analytical_max:.4f}\n")

amplitude_ratio = pinn_max / analytical_max
print(f"PINN / Analytical ratio: {amplitude_ratio:.4f}")

# Diagnosis
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if pinn_max > 0.8:
    print("\n⚠⚠⚠ PROBLEM CONFIRMED!")
    print(f"  PINN amplitude ({pinn_max:.4f}) is TOO HIGH")
    print(f"  Should be around {amplitude/2:.4f}")
    print("\n  This confirms Gemini's diagnosis:")
    print("  - PINN is NOT learning the wave split correctly")
    print("  - PINN predicts amplitude ≈ 1.0 (unsplit wave)")
    print("  - Analytical solution has amplitude ≈ 0.5 (split wave)")
    print("  - This mismatch causes ~50-60% relative error")
elif pinn_max < 0.3:
    print("\n⚠ PINN amplitude is too LOW")
    print("  Underfitting or learning issues")
elif abs(pinn_max - analytical_max) < 0.1:
    print("\n✓ PINN amplitude is CORRECT")
    print("  PINN correctly learns the wave split")
else:
    print(f"\n△ Partial learning, needs more training")

# Check initial velocity loss
if loss_callback.history["L_ic_velocity"]:
    final_vel_loss = loss_callback.history["L_ic_velocity"][-1]
    print(f"\nFinal L_ic_velocity loss: {final_vel_loss:.6e}")
    if final_vel_loss > 0.1:
        print("  ⚠ HIGH velocity loss - PINN struggles with u_t(x,0)=0 condition")
        print("  This is WHY the wave doesn't split correctly!")
    else:
        print("  ✓ Velocity loss is low - initial condition well satisfied")

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_test, u_analytical, "b-", linewidth=2, label="Analytical (split wave, amp≈0.5)", alpha=0.7)
ax.plot(x_test, u_pinn, "r--", linewidth=2, label=f"PINN (amp={pinn_max:.4f})", alpha=0.7)
ax.axhline(amplitude / 2, color="g", linestyle=":", label=f"Expected amplitude ({amplitude/2:.1f})", alpha=0.5)
ax.axhline(amplitude, color="orange", linestyle=":", label=f"Original amplitude ({amplitude:.1f})", alpha=0.5)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("u(x, t)", fontsize=12)
ax.set_title(f"Wave Amplitude at t={t_check} (Before Reflection)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(project_root / "pinn_amplitude_check.png", dpi=150)
print(f"\nPlot saved to: pinn_amplitude_check.png")

print("\n" + "=" * 80)
