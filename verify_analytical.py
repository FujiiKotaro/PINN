"""Verify analytical solution implementation step by step."""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0
x0 = L / 4  # 0.5
sigma = L / 10  # 0.2
amplitude = 1.0
c = 1.5


def initial_displacement(x):
    """Gaussian pulse."""
    x_val = np.atleast_1d(x).reshape(-1, 1)
    return amplitude * np.exp(-((x_val - x0) ** 2) / (2 * sigma**2))


def test_extended_ic():
    """Test the extended initial condition."""
    print("=" * 80)
    print("Testing Extended Initial Condition (Period 2L)")
    print("=" * 80)

    x_extended = np.linspace(-2*L, 2*L, 800)

    def extended_ic(x_eval):
        """Period 2L even extension."""
        x_eval = np.atleast_1d(x_eval).flatten()
        result = np.zeros_like(x_eval)

        for i, xe in enumerate(x_eval):
            # Map to [0, 2L) using modulo
            xe_mod = np.mod(xe, 2 * L)

            # Even extension at x=L
            if 0 <= xe_mod <= L:
                x_ic = xe_mod
            else:  # L < xe_mod < 2L
                x_ic = 2*L - xe_mod

            x_ic = np.clip(x_ic, 0, L)
            val = initial_displacement(np.array([[x_ic]]))
            result[i] = val[0, 0]

        return result

    u_extended = extended_ic(x_extended)

    plt.figure(figsize=(14, 5))
    plt.plot(x_extended, u_extended, 'b-', linewidth=2)
    plt.axvline(0, color='r', linestyle='--', alpha=0.5, label='x=0')
    plt.axvline(L, color='r', linestyle='--', alpha=0.5, label='x=L')
    plt.axvline(-L, color='orange', linestyle=':', alpha=0.5)
    plt.axvline(2*L, color='orange', linestyle=':', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('f_extended(x)')
    plt.title('Extended Initial Condition (Period 2L = 4.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('extended_ic_test.png', dpi=150)
    plt.show()

    # Check key properties
    print("\nKey positions:")
    print(f"  f(0.5) = {extended_ic(np.array([0.5]))[0]:.4f} (original peak)")
    print(f"  f(1.5) = {extended_ic(np.array([1.5]))[0]:.4f} (mirror at x=L)")
    print(f"  f(-0.5) = {extended_ic(np.array([-0.5]))[0]:.4f} (mirror at x=0)")
    print(f"  f(2.5) = {extended_ic(np.array([2.5]))[0]:.4f} (next period)")
    print(f"  f(4.5) = {extended_ic(np.array([4.5]))[0]:.4f} (period 4.0)")

    # Check symmetry
    print("\nSymmetry check:")
    print(f"  f(-0.5) == f(0.5)? {np.isclose(extended_ic(np.array([-0.5])), extended_ic(np.array([0.5])))[0]}")
    print(f"  f(1.5) == f(2.5)? {np.isclose(extended_ic(np.array([1.5])), extended_ic(np.array([2.5])))[0]}")

test_extended_ic()


def test_wave_evolution():
    """Test wave evolution over time."""
    print("\n" + "=" * 80)
    print("Testing Wave Evolution u(x,t) = f(x - ct)")
    print("=" * 80)

    x_test = np.linspace(0, L, 200)
    test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.67]  # 2.67 = one period

    def extended_ic(x_eval):
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, t_val in enumerate(test_times):
        u = extended_ic(x_test - c * t_val)

        ax = axes[idx]
        ax.plot(x_test, u, 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f't = {t_val:.2f}s (wave moved {c*t_val:.2f})')
        ax.set_ylim([-0.1, 1.1])
        ax.axvline(0, color='r', linestyle='--', alpha=0.3)
        ax.axvline(L, color='r', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

        max_val = np.max(np.abs(u))
        ax.text(0.05, 0.95, f'Max: {max_val:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        print(f"\nt = {t_val:.2f}s (moved {c*t_val:.2f}):")
        print(f"  Max amplitude: {max_val:.4f}")
        if max_val > 0.3:
            peak_idx = np.argmax(np.abs(u))
            print(f"  Peak position: x = {x_test[peak_idx]:.3f}")

    plt.tight_layout()
    plt.savefig('wave_evolution_test.png', dpi=150)
    plt.show()

    print("\n" + "=" * 80)
    print("Expected behavior:")
    print(f"  Initial peak at x = {x0}")
    print(f"  t = {(L-x0)/c:.2f}s: reaches right boundary")
    print(f"  t = {2*L/c:.2f}s: returns to left boundary")
    print(f"  t = {2*L/c:.2f}s: full cycle, should return to original")
    print("=" * 80)

test_wave_evolution()
