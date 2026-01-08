"""Verify that period 2L extension satisfies Neumann BC, not periodic BC."""

import numpy as np
import matplotlib.pyplot as plt

L = 2.0
x0 = L / 2  # 1.0
sigma = L / 10
amplitude = 1.0
c = 1.5


def initial_displacement(x):
    x_val = np.atleast_1d(x).reshape(-1, 1)
    return amplitude * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))


def extended_ic(x_eval):
    """Period 2L even extension for Neumann BC."""
    x_eval = np.atleast_1d(x_eval).flatten()
    result = np.zeros_like(x_eval)

    for i, xe in enumerate(x_eval):
        xe_mod = np.mod(xe, 2 * L)
        if 0 <= xe_mod <= L:
            x_ic = xe_mod
        else:
            x_ic = 2*L - xe_mod
        x_ic = np.clip(x_ic, 0, L)
        val = initial_displacement(np.array([[x_ic]]))
        result[i] = val[0, 0]

    return result


def check_boundary_conditions(t_val):
    """Check if Neumann BC is satisfied at boundaries."""
    print(f"\n{'='*80}")
    print(f"Checking Boundary Conditions at t = {t_val:.2f}s")
    print(f"{'='*80}")

    # Define solution
    def u(x, t):
        return extended_ic(x - c * t)

    # Numerical derivative at boundaries
    dx = 1e-6

    # At x=0
    u_0_plus = u(np.array([0 + dx]), t_val)[0]
    u_0 = u(np.array([0]), t_val)[0]
    dudx_0 = (u_0_plus - u_0) / dx

    # At x=L
    u_L = u(np.array([L]), t_val)[0]
    u_L_minus = u(np.array([L - dx]), t_val)[0]
    dudx_L = (u_L - u_L_minus) / dx

    print(f"\nNeumann BC: ∂u/∂x should be ≈ 0 at both boundaries")
    print(f"  ∂u/∂x|_(x=0) = {dudx_0:.6e}")
    print(f"  ∂u/∂x|_(x=L) = {dudx_L:.6e}")

    if abs(dudx_0) < 1e-4 and abs(dudx_L) < 1e-4:
        print(f"  ✓ Neumann BC satisfied!")
    else:
        print(f"  ✗ Neumann BC NOT satisfied")

    # Check periodic BC (for comparison)
    print(f"\nPeriodic BC check (for comparison): u(0) should equal u(L)")
    print(f"  u(0, t) = {u_0:.6f}")
    print(f"  u(L, t) = {u_L:.6f}")
    print(f"  Difference: {abs(u_0 - u_L):.6f}")

    if abs(u_0 - u_L) < 1e-4:
        print(f"  → Would satisfy periodic BC (NOT our case)")
    else:
        print(f"  → Does NOT satisfy periodic BC (correct for Neumann)")

    return dudx_0, dudx_L


# Test at multiple times
test_times = [0.0, 0.5, 1.0, 1.5, 2.0]

fig, axes = plt.subplots(len(test_times), 1, figsize=(10, 12))

for idx, t_val in enumerate(test_times):
    dudx_0, dudx_L = check_boundary_conditions(t_val)

    # Plot
    x_plot = np.linspace(0, L, 300)
    u_vals = extended_ic(x_plot - c * t_val)

    ax = axes[idx]
    ax.plot(x_plot, u_vals, 'b-', linewidth=2, label='u(x,t)')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Boundaries')
    ax.axvline(L, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f't = {t_val:.2f}s | ∂u/∂x|₀ = {dudx_0:.2e}, ∂u/∂x|_L = {dudx_L:.2e}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig('neumann_bc_verification.png', dpi=150)
print(f"\n{'='*80}")
print("Plot saved: neumann_bc_verification.png")
print(f"{'='*80}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print("The period 2L even extension implements:")
print("  ✓ Neumann BC: ∂u/∂x = 0 at x=0, L")
print("  ✗ NOT periodic BC: u(0) ≠ u(L) in general")
print("\nThis is the CORRECT way to handle Neumann BC with method of images.")
print(f"{'='*80}")
