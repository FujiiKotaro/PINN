"""Test corrected analytical solution with Neumann BC reflections."""

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
    """Right-traveling wave with Neumann BC reflections - CORRECTED."""
    x = np.atleast_1d(x).flatten()
    t = np.atleast_1d(t).flatten()

    X, T = np.meshgrid(x, t, indexing="ij")
    X_flat = X.flatten()
    T_flat = T.flatten()

    def extended_ic(x_eval):
        """Extend IC with even symmetry for Neumann BC - CORRECTED."""
        x_eval = np.atleast_1d(x_eval).flatten()
        result = np.zeros_like(x_eval)

        for i, xe in enumerate(x_eval):
            # Map to fundamental period [0, 2L] using modulo
            # np.mod handles negative values correctly
            xe_mod = np.mod(xe, 2 * L)

            # Even extension: mirror at x=L
            if 0 <= xe_mod <= L:
                x_ic = xe_mod
            else:  # L < xe_mod < 2L
                x_ic = 2*L - xe_mod

            # Clamp to [0, L] and evaluate
            x_ic = np.clip(x_ic, 0, L)

            try:
                val = initial_condition(np.array([[x_ic]]))
                result[i] = val[0, 0]
            except:
                result[i] = 0.0

        return result.reshape(-1, 1)

    # Right-traveling wave: u(x,t) = f(x - ct) with extended f
    pos = X_flat - c * T_flat
    u = extended_ic(pos)

    return u.reshape(X.shape)


print("=" * 80)
print("Testing CORRECTED Analytical Solution")
print("=" * 80)

# Test at several time points
x_test = np.linspace(0, L, 200)
test_times = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
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
    ax.axvline(0, color="r", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(L, color="r", linestyle="--", alpha=0.3, linewidth=1)

    # Check amplitude
    max_amp = np.max(np.abs(u))
    ax.text(0.05, 0.95, f"Max: {max_amp:.3f}", transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Track wave position
    if max_amp > 0.3:
        peak_idx = np.argmax(np.abs(u[:, 0]))
        peak_x = x_test[peak_idx]
        print(f"t = {t_val:.2f}s: max = {max_amp:.4f} at x = {peak_x:.3f}")
    else:
        print(f"t = {t_val:.2f}s: max = {max_amp:.4f} (wave position unclear)")

plt.tight_layout()
plt.savefig("analytical_solution_test_v2.png", dpi=150)
print(f"\nPlot saved to: analytical_solution_test_v2.png")

print("\n" + "=" * 80)
print("Expected Behavior (Right-Traveling Wave):")
print("=" * 80)
print("- t=0.0: Initial pulse at x=0.5, amplitude=1.0")
print(f"- Wave travels RIGHT at speed c={c}")
print(f"- At t={L/c:.2f}s: Wave reaches x=L (right boundary)")
print(f"- Neumann BC: Reflects without phase flip (even extension)")
print(f"- At t={2*L/c:.2f}s: Returns to x=0 (after reflection)")
print("- Amplitude maintained at ~1.0 throughout")
print("=" * 80)

# Verify reflection behavior
print("\n" + "=" * 80)
print("Reflection Check:")
print("=" * 80)

# Time when wave should be at boundaries
t_at_right_boundary = (L - x0) / c
t_after_reflection = t_at_right_boundary + 0.5 / c

print(f"\nWave should reach RIGHT boundary (x={L}) at t ≈ {t_at_right_boundary:.3f}s")
print(f"After reflection, at t ≈ {t_after_reflection:.3f}s, should be traveling LEFT")

u_before = analytical_solution_right_wave_with_reflections(
    x=x_test, t=np.array([t_at_right_boundary - 0.1]), c=c, L=L, initial_condition=initial_displacement
)
u_at = analytical_solution_right_wave_with_reflections(
    x=x_test, t=np.array([t_at_right_boundary]), c=c, L=L, initial_condition=initial_displacement
)
u_after = analytical_solution_right_wave_with_reflections(
    x=x_test, t=np.array([t_after_reflection]), c=c, L=L, initial_condition=initial_displacement
)

peak_before_idx = np.argmax(np.abs(u_before[:, 0]))
peak_at_idx = np.argmax(np.abs(u_at[:, 0]))
peak_after_idx = np.argmax(np.abs(u_after[:, 0]))

print(f"\nBefore reflection: peak at x = {x_test[peak_before_idx]:.3f}")
print(f"At boundary: peak at x = {x_test[peak_at_idx]:.3f}")
print(f"After reflection: peak at x = {x_test[peak_after_idx]:.3f}")

if x_test[peak_after_idx] < x_test[peak_at_idx]:
    print("\n✓ CORRECT: Wave is traveling LEFT after reflection")
else:
    print("\n✗ ERROR: Wave should be traveling LEFT after reflection")

print("=" * 80)
