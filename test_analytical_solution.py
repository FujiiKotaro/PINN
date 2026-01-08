"""Quick test of the analytical solution with Neumann BC reflections."""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0
x0 = L / 4
sigma = L / 10
amplitude = 1.0
c = 1.5


def initial_displacement(x):
    """Gaussian pulse initial condition."""
    if x.ndim == 1:
        x_val = x.reshape(-1, 1)
    else:
        x_val = x[:, 0:1] if x.shape[1] >= 1 else x
    return amplitude * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))


def analytical_solution_right_wave_with_reflections(x, t, c, L, initial_condition):
    """Right-traveling wave with Neumann BC reflections.

    Solution: u(x,t) = f_extended(x - ct)
    where f_extended uses even symmetry (Neumann BC) with period 2L.
    """
    x = np.atleast_1d(x).flatten()
    t = np.atleast_1d(t).flatten()

    X, T = np.meshgrid(x, t, indexing="ij")
    X_flat = X.flatten()
    T_flat = T.flatten()

    def extended_ic(x_eval):
        """Extend IC with even symmetry for Neumann BC."""
        x_eval = np.atleast_1d(x_eval).flatten()
        result = np.zeros_like(x_eval)

        for i, xe in enumerate(x_eval):
            # Shift to positive range
            xe_shifted = xe
            while xe_shifted < 0:
                xe_shifted += 2 * L

            # Map to fundamental period [0, 2L]
            xe_mod = xe_shifted % (2 * L)

            # Even extension: mirror at x=L
            if xe_mod <= L:
                x_ic = xe_mod
            else:  # L < xe_mod < 2L
                x_ic = 2*L - xe_mod

            # Clamp and evaluate
            x_ic = np.clip(x_ic, 0, L)

            try:
                val = initial_condition(np.array([[x_ic]]))
                result[i] = val[0, 0]
            except:
                result[i] = 0.0

        return result.reshape(-1, 1)

    # Right-traveling wave: u(x,t) = f(x - ct)
    pos = X_flat - c * T_flat
    u = extended_ic(pos)

    return u.reshape(X.shape)


print("=" * 80)
print("Testing Analytical Solution with Neumann BC Reflections")
print("=" * 80)

# Test at several time points
x_test = np.linspace(0, L, 200)
test_times = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, t_val in enumerate(test_times):
    u = analytical_solution_right_wave_with_reflections(
        x=x_test, t=np.array([t_val]), c=c, L=L, initial_condition=initial_displacement
    )

    ax = axes[idx]
    ax.plot(x_test, u[:, 0], "b-", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"t = {t_val:.2f}s")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    ax.axvline(0, color="k", linestyle="--", alpha=0.3, label="Boundaries")
    ax.axvline(L, color="k", linestyle="--", alpha=0.3)

    # Check amplitude
    max_amp = np.max(np.abs(u))
    ax.text(0.05, 0.95, f"Max: {max_amp:.3f}", transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    print(f"t = {t_val:.2f}s: max amplitude = {max_amp:.4f}")

plt.tight_layout()
plt.savefig("analytical_solution_test.png", dpi=150)
print(f"\nPlot saved to: analytical_solution_test.png")

print("\n" + "=" * 80)
print("Expected Behavior:")
print("=" * 80)
print("- t=0.0: Initial Gaussian pulse at x=0.5 with amplitude 1.0")
print("- Wave travels RIGHT at speed c=1.5")
print("- At x=2.0: Neumann BC reflection (no phase flip)")
print("- Amplitude should stay around 1.0 (single-direction wave)")
print("=" * 80)
