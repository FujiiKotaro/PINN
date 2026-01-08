"""Diagnose PINN's learned initial velocity condition."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import deepxde as dde

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from pinn.models.boundary_conditions import BoundaryConditionsService
from pinn.models.causal_pde_definition import CausalPDEDefinitionService
from pinn.models.fourier_network import FourierFeatureNetwork
from pinn.utils.config_loader import ConfigLoaderService

# Load config and model
config_path = project_root / "configs" / "traveling_wave_example.yaml"
config_loader = ConfigLoaderService()
config = config_loader.load_config(config_path)

# Reconstruct model
geom = dde.geometry.Interval(config.domain.x_min, config.domain.x_max)
timedomain = dde.geometry.TimeDomain(config.domain.t_min, config.domain.t_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Load latest experiment
exp_dir = project_root / "experiments" / "right_wave_2026-01-08_10-11-18"
model_path = exp_dir / "model.pt"

# Parameters
L = config.domain.x_max - config.domain.x_min
x0 = L / 2
sigma = L / 10
amplitude = 1.0
c = config.domain.wave_speed


def initial_displacement(x):
    """Initial displacement: Gaussian pulse."""
    if x.ndim == 1:
        x_val = x.reshape(-1, 1)
    else:
        x_val = x[:, 0:1] if x.shape[1] >= 1 else x
    return amplitude * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))


def initial_velocity_analytical(x):
    """Initial velocity for RIGHT-traveling wave."""
    if x.ndim == 1:
        x_val = x.reshape(-1, 1)
    else:
        x_val = x[:, 0:1] if x.shape[1] >= 1 else x
    return c * amplitude * (x_val - x0) / (sigma**2) * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))


# Create minimal PDE for reconstruction
pde_func = CausalPDEDefinitionService.create_adaptive_causal_pde_function(
    c=config.domain.wave_speed, t_max=config.domain.t_max, beta=config.training.causal_beta
)

def zero_dirichlet(x):
    return np.zeros((len(x), 1))

def on_boundary(x, on_boundary):
    return on_boundary

bc = BoundaryConditionsService.create_dirichlet_bc(geomtime, zero_dirichlet, on_boundary)
ic_displacement = BoundaryConditionsService.create_initial_condition(geomtime, initial_displacement)
ic_velocity = BoundaryConditionsService.create_initial_velocity(geomtime, initial_velocity_analytical)

data = dde.data.TimePDE(
    geomtime,
    pde_func,
    [bc, ic_displacement, ic_velocity],
    num_domain=100,
    num_boundary=20,
    num_initial=100,
)

# Reconstruct network
hidden_layers = config.network.layer_sizes[1:-1]
net = FourierFeatureNetwork(
    input_dim=2,
    output_dim=1,
    hidden_layers=hidden_layers,
    num_fourier_features=config.network.num_fourier_features,
    fourier_scale=config.network.fourier_scale,
    activation=config.network.activation,
)

model = dde.Model(data, net)
model.compile("adam", lr=config.training.learning_rate)

# Load trained weights
print(f"Loading model from: {model_path}")
model.restore(str(model_path))

# Compute PINN's velocity at t=0
nx = 200
x_test = np.linspace(config.domain.x_min, config.domain.x_max, nx)
t_zero = np.zeros_like(x_test)
xt_zero = np.column_stack([x_test, t_zero])

# Convert to torch tensor for gradient computation
xt_tensor = torch.tensor(xt_zero, dtype=torch.float32, requires_grad=True)

# Get network output
u_output = net(xt_tensor)

# Compute du/dt
du_dt = torch.autograd.grad(
    outputs=u_output,
    inputs=xt_tensor,
    grad_outputs=torch.ones_like(u_output),
    create_graph=True,
)[0][:, 1:2]  # Take time derivative (index 1)

# Convert to numpy
u_t_pinn = du_dt.detach().numpy()

# Analytical initial velocity
u_t_analytical = initial_velocity_analytical(x_test)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Initial displacement (for reference)
u_displacement = initial_displacement(x_test)
axes[0].plot(x_test, u_displacement, "b-", linewidth=2)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x, 0)")
axes[0].set_title("Initial Displacement (Reference)")
axes[0].grid(True)

# Initial velocity comparison
axes[1].plot(x_test, u_t_analytical, "b-", linewidth=2, label="Analytical $u_t(x,0)$", alpha=0.7)
axes[1].plot(x_test, u_t_pinn, "r--", linewidth=2, label="PINN $u_t(x,0)$", alpha=0.7)
axes[1].axhline(0, color="k", linestyle="--", alpha=0.3)
axes[1].set_xlabel("x")
axes[1].set_ylabel("$u_t(x, 0)$")
axes[1].set_title("Initial Velocity: Analytical vs PINN")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("pinn_initial_velocity_check.png", dpi=150)
print("Saved: pinn_initial_velocity_check.png")

# Compute error metrics
l2_error = np.linalg.norm(u_t_pinn.flatten() - u_t_analytical.flatten()) / np.linalg.norm(u_t_analytical.flatten())
max_error = np.max(np.abs(u_t_pinn.flatten() - u_t_analytical.flatten()))

print("\n" + "=" * 80)
print("Initial Velocity Diagnosis")
print("=" * 80)
print(f"Analytical u_t(x,0):")
print(f"  Max:  {np.max(u_t_analytical):.6f}")
print(f"  Min:  {np.min(u_t_analytical):.6f}")
print(f"  Mean: {np.mean(u_t_analytical):.6f}")
print(f"\nPINN u_t(x,0):")
print(f"  Max:  {np.max(u_t_pinn):.6f}")
print(f"  Min:  {np.min(u_t_pinn):.6f}")
print(f"  Mean: {np.mean(u_t_pinn):.6f}")
print(f"\nError:")
print(f"  L2 relative error: {l2_error:.6f} ({l2_error * 100:.2f}%)")
print(f"  Max absolute error: {max_error:.6f}")
print("=" * 80)

if l2_error > 0.5:
    print("\nCONCLUSION: PINN did NOT learn the initial velocity condition correctly!")
    print("This explains the wave splitting observed in the solution.")
elif l2_error > 0.1:
    print("\nCONCLUSION: PINN partially learned the velocity condition, but with significant error.")
else:
    print("\nCONCLUSION: PINN learned the velocity condition well.")
    print("The error must come from somewhere else (BC, PDE, or long-time propagation).")

plt.show()
