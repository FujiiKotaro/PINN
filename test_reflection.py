"""Test the reflection implementation for traveling waves."""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PINN.pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService

# Setup
L = 2.0
c = 1.5
x0 = L / 2
sigma = L / 10
amplitude = 1.0


def initial_condition(x):
    """Gaussian pulse initial condition."""
    if x.ndim == 1:
        x_reshaped = x.reshape(-1, 1)
    else:
        x_reshaped = x if x.shape[1] >= 1 else x.reshape(-1, 1)
    return amplitude * np.exp(-((x_reshaped[:, 0:1] - x0) ** 2) / (2 * sigma**2))


# Generate analytical solution
nx = 200
nt = 8
x_test = np.linspace(0, L, nx)
t_test = np.linspace(0, 2.0, nt)

solver = AnalyticalSolutionGeneratorService()
u_analytical = solver.traveling_wave_with_reflections(
    x=x_test,
    t=t_test,
    c=c,
    initial_condition=initial_condition,
    L=L,
    n_reflections=10
)

# Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(nt):
    ax = axes[i]
    ax.plot(x_test, u_analytical[:, i], 'b-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title(f't = {t_test[i]:.3f}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.2])
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='boundaries')
    ax.axvline(L, color='r', linestyle='--', alpha=0.5)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('test_reflection_output.png', dpi=150, bbox_inches='tight')
print("Test complete. Saved to test_reflection_output.png")

# Check that boundary conditions are satisfied
print("\nBoundary condition check (du/dx should be ~0 at boundaries):")
for i in range(nt):
    # Approximate derivative at boundaries
    dx = x_test[1] - x_test[0]
    du_dx_left = (u_analytical[1, i] - u_analytical[0, i]) / dx
    du_dx_right = (u_analytical[-1, i] - u_analytical[-2, i]) / dx
    print(f"t={t_test[i]:.3f}: du/dx(0)={du_dx_left:.6f}, du/dx(L)={du_dx_right:.6f}")
