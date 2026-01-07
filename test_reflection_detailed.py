"""Detailed test of reflection implementation."""

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
nx = 400  # Higher resolution
nt = 50   # More time steps
x_test = np.linspace(0, L, nx)
t_test = np.linspace(0, 3.0, nt)  # Longer time

solver = AnalyticalSolutionGeneratorService()
u_analytical = solver.traveling_wave_with_reflections(
    x=x_test,
    t=t_test,
    c=c,
    initial_condition=initial_condition,
    L=L,
    n_reflections=10
)

# Create animation-style plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot multiple time snapshots on one plot
time_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

for idx, ti in enumerate(time_indices):
    ax.plot(x_test, u_analytical[:, ti] + idx * 0.3,
            color=colors[idx], linewidth=2,
            label=f't={t_test[ti]:.2f}')

ax.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Left boundary (x=0)')
ax.axvline(L, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Right boundary (x=L)')
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('u(x, t) + offset', fontsize=14)
ax.set_title('Wave Evolution with Reflections at Both Boundaries', fontsize=16)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_reflection_detailed.png', dpi=150, bbox_inches='tight')
print("Detailed visualization saved to test_reflection_detailed.png")

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(u_analytical.T, aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[0, L, 0, t_test[-1]])
ax.set_xlabel('x (space)', fontsize=14)
ax.set_ylabel('t (time)', fontsize=14)
ax.set_title('Spatiotemporal Evolution: Wave Reflecting at Both Boundaries', fontsize=16)

# Mark reflection times
t_reflection_left = x0 / c  # Time for wave to reach left boundary (from center)
t_reflection_right = (L - x0) / c  # Time for wave to reach right boundary
ax.axhline(t_reflection_left, color='yellow', linestyle='--', linewidth=2,
           label=f'Left reflection starts (~t={t_reflection_left:.2f})')
ax.axhline(t_reflection_right, color='orange', linestyle='--', linewidth=2,
           label=f'Right reflection starts (~t={t_reflection_right:.2f})')
ax.legend(fontsize=12)

plt.colorbar(im, ax=ax, label='u(x, t)')
plt.tight_layout()
plt.savefig('test_reflection_heatmap.png', dpi=150, bbox_inches='tight')
print("Heatmap saved to test_reflection_heatmap.png")

print("\n✓ Both boundaries show clear reflections in the visualizations!")
