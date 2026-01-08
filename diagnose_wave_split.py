"""Diagnose wave splitting issue in traveling wave PINN.

This script checks if the PINN correctly learns wave splitting behavior
when initial velocity is zero.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pinn.validation.analytical_solutions import AnalyticalSolutionGeneratorService

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
print("DIAGNOSING WAVE SPLITTING BEHAVIOR")
print("=" * 80)

# Generate analytical solution
analytical_solver = AnalyticalSolutionGeneratorService()

# Test at different times
x_test = np.linspace(0, L, 200)
times = [0.0, 0.2, 0.4, 0.6, 0.8]

fig, axes = plt.subplots(len(times), 1, figsize=(12, 12))

print("\n【Analytical Solution Analysis】")
print(f"Initial amplitude: {amplitude}")
print(f"Initial velocity: 0 (wave should split)\n")

for i, t in enumerate(times):
    ax = axes[i]

    # Analytical solution
    u_analytical = analytical_solver.traveling_wave_with_reflections(
        x=x_test,
        t=np.array([t]),
        c=c,
        initial_condition=initial_condition,
        L=L,
        n_reflections=10,
    )

    # Check maximum amplitude
    max_amp = np.max(u_analytical)

    print(f"t = {t:.1f}s:")
    print(f"  Max amplitude: {max_amp:.4f}")

    # Expected behavior
    if t == 0.0:
        expected = amplitude
        print(f"  Expected: {expected:.4f} (initial pulse)")
    else:
        expected = amplitude / 2.0  # Split wave should have half amplitude
        print(f"  Expected: ~{expected:.4f} (split waves)")
        if abs(max_amp - expected) < 0.1:
            print(f"  ✓ Correctly shows wave splitting")
        else:
            print(f"  ⚠ Amplitude mismatch!")

    # Plot
    ax.plot(x_test, u_analytical[:, 0], "b-", linewidth=2, label="Analytical")
    ax.axhline(amplitude / 2, color="r", linestyle="--", alpha=0.5, label="Expected amplitude (0.5)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"t = {t:.1f}s, Max amplitude = {max_amp:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig(project_root / "wave_split_diagnosis.png", dpi=150)
print(f"\nPlot saved to: wave_split_diagnosis.png")
plt.show()

print("\n" + "=" * 80)
print("THEORETICAL EXPLANATION")
print("=" * 80)
print("\nD'Alembert solution with ZERO initial velocity:")
print("  u(x,t) = 1/2 * [f(x-ct) + f(x+ct)]")
print("\nThis means:")
print("  - Initial pulse f(x) splits into TWO waves")
print("  - Left-traveling wave: 1/2 * f(x+ct)")
print("  - Right-traveling wave: 1/2 * f(x-ct)")
print("  - Each wave has HALF the original amplitude")
print("\n✓ If analytical solution shows amplitude ≈ 0.5, it's CORRECT")
print("✗ If PINN shows amplitude ≈ 1.0, it's NOT learning the split correctly")
print("=" * 80)

# Check d'Alembert solution analytically
print("\n【Manual D'Alembert Verification】")
t_test = 0.3
x_test_point = np.array([1.0])

# Direct d'Alembert calculation
u_left = initial_condition(x_test_point + c * t_test)  # f(x + ct)
u_right = initial_condition(x_test_point - c * t_test)  # f(x - ct)
u_dalembert = 0.5 * (u_left + u_right)

print(f"\nAt x={x_test_point[0]}, t={t_test}:")
print(f"  f(x + ct) = {u_left[0,0]:.6f}")
print(f"  f(x - ct) = {u_right[0,0]:.6f}")
print(f"  u(x,t) = 1/2 * [f(x+ct) + f(x-ct)] = {u_dalembert[0,0]:.6f}")

# Compare with analytical solver
u_solver = analytical_solver.traveling_wave_with_reflections(
    x=x_test_point,
    t=np.array([t_test]),
    c=c,
    initial_condition=initial_condition,
    L=L,
    n_reflections=10,
)

print(f"  Analytical solver result: {u_solver[0,0]:.6f}")
print(f"  Match: {np.allclose(u_dalembert, u_solver, atol=1e-4)}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
