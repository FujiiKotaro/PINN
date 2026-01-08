"""Analytical solution for 1D wave equation with Dirichlet BC.

Physics:
- Wave equation: u_tt = c^2 * u_xx
- Dirichlet BC: u(0,t) = u(L,t) = 0
- Initial conditions: u(x,0) = f(x), u_t(x,0) = g(x)

Solution: d'Alembert formula with method of images (odd extension)
"""

import numpy as np
import matplotlib.pyplot as plt


def analytical_solution_1d_wave_dirichlet(x, t, c, L, initial_displacement, initial_velocity):
    """Analytical solution for 1D wave with Dirichlet BC.

    Uses d'Alembert formula: u(x,t) = 0.5 * [f(x-ct) + f(x+ct)] + integral term
    with odd extension for Dirichlet BC.

    Args:
        x: Spatial points in [0, L]
        t: Time points
        c: Wave speed
        L: Domain length
        initial_displacement: f(x) function
        initial_velocity: g(x) function

    Returns:
        u(x,t) with shape (len(x), len(t))
    """
    x = np.atleast_1d(x).flatten()
    t = np.atleast_1d(t).flatten()

    X, T = np.meshgrid(x, t, indexing="ij")
    X_flat = X.flatten()
    T_flat = T.flatten()

    def extend_odd(func, x_eval):
        """Extend function with odd symmetry for Dirichlet BC.

        Odd extension: f(-x) = -f(x) at x=0, f(2L-x) = -f(x) at x=L
        Period = 2L
        """
        x_eval = np.atleast_1d(x_eval).flatten()
        result = np.zeros_like(x_eval)

        for i, xe in enumerate(x_eval):
            # Map to period [0, 2L)
            xe_mod = np.mod(xe, 2 * L)

            # Odd extension at x=L
            if 0 <= xe_mod <= L:
                x_ic = xe_mod
                sign = 1.0
            else:  # L < xe_mod < 2L
                x_ic = 2*L - xe_mod
                sign = -1.0  # Phase flip for odd extension

            # Evaluate function
            x_ic = np.clip(x_ic, 0, L)
            try:
                val = func(np.array([[x_ic]]))
                if val.ndim > 1:
                    result[i] = sign * val[0, 0]
                else:
                    result[i] = sign * val[0]
            except:
                result[i] = 0.0

        return result

    # d'Alembert solution: u(x,t) = 0.5 * [f(x-ct) + f(x+ct)] + integral of g

    # Part 1: 0.5 * [f(x-ct) + f(x+ct)]
    pos_left = X_flat - c * T_flat
    pos_right = X_flat + c * T_flat

    f_left = extend_odd(initial_displacement, pos_left)
    f_right = extend_odd(initial_displacement, pos_right)

    u_displacement = 0.5 * (f_left + f_right)

    # Part 2: (1/2c) * integral from x-ct to x+ct of g(s) ds
    # For initial velocity g(x), we need to integrate
    # We'll approximate this numerically

    u_velocity = np.zeros_like(X_flat)

    for i in range(len(X_flat)):
        x_val = X_flat[i]
        t_val = T_flat[i]

        # Integration bounds
        s_min = x_val - c * t_val
        s_max = x_val + c * t_val

        # Numerical integration (simple trapezoid rule)
        n_points = 50
        s_points = np.linspace(s_min, s_max, n_points)
        g_values = extend_odd(initial_velocity, s_points)

        # Trapezoid rule
        integral = np.trapz(g_values, s_points)
        u_velocity[i] = integral / (2 * c)

    # Total solution
    u_total = u_displacement + u_velocity

    return u_total.reshape(X.shape)


# Test with our problem
if __name__ == "__main__":
    L = 2.0
    x0 = L / 2
    sigma = L / 10
    amplitude = 1.0
    c = 1.5

    def initial_displacement(x):
        x_val = np.atleast_1d(x).reshape(-1, 1)
        return amplitude * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))

    def initial_velocity(x):
        """Right-traveling wave: u_t(x,0) = -c * f'(x)"""
        x_val = np.atleast_1d(x).reshape(-1, 1)
        return c * amplitude * (x_val - x0) / (sigma**2) * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))

    print("=" * 80)
    print("Testing 1D Wave Analytical Solution with Dirichlet BC")
    print("=" * 80)

    x_test = np.linspace(0, L, 200)
    test_times = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, t_val in enumerate(test_times):
        u = analytical_solution_1d_wave_dirichlet(
            x=x_test,
            t=np.array([t_val]),
            c=c,
            L=L,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity
        )

        ax = axes[idx]
        ax.plot(x_test, u[:, 0], 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(L, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f't = {t_val:.2f}s')
        ax.set_ylim([-1.2, 1.2])
        ax.grid(True, alpha=0.3)

        # Check boundary conditions
        u_0 = u[0, 0]
        u_L = u[-1, 0]
        max_amp = np.max(np.abs(u[:, 0]))

        ax.text(0.05, 0.95,
                f'u(0)={u_0:.2e}\\nu(L)={u_L:.2e}\\nmax={max_amp:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

        print(f"\nt = {t_val:.2f}s:")
        print(f"  u(0,t) = {u_0:.6e} (should be ≈ 0)")
        print(f"  u(L,t) = {u_L:.6e} (should be ≈ 0)")
        print(f"  Max amplitude: {max_amp:.4f}")

    plt.tight_layout()
    plt.savefig('analytical_dirichlet_wave.png', dpi=150)
    print(f"\nPlot saved: analytical_dirichlet_wave.png")

    print("\n" + "=" * 80)
    print("Verification Complete")
    print("=" * 80)
    print("Analytical solution uses:")
    print("  - d'Alembert formula with odd extension")
    print("  - Dirichlet BC: u(0,t) = u(L,t) = 0")
    print("  - Custom initial velocity: u_t(x,0) = -c*f'(x)")
    print("=" * 80)
