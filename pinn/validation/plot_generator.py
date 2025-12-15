"""Plot Generator for PINN Validation Visualization.

Generates publication-quality plots for:
- Training curves (loss vs epochs)
- Solution comparisons (PINN vs analytical)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class PlotGeneratorService:
    """Generate visualization plots for PINN training and validation."""

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Path] = None,
        log_scale: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot training loss curves over epochs.

        Args:
            history: Dictionary containing loss history with keys:
                - "total_loss": Total loss values per epoch
                - "L_data": Data fitting loss (optional)
                - "L_pde": PDE residual loss (optional)
                - "L_bc": Boundary condition loss (optional)
            save_path: Path to save plot (if provided)
            log_scale: Use logarithmic scale for y-axis

        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract epochs
        total_loss = history.get("total_loss", [])
        epochs = np.arange(len(total_loss))

        # Plot each loss component
        if "total_loss" in history:
            ax.plot(epochs, history["total_loss"],
                   label="Total Loss", linewidth=2, color="black")

        if "L_data" in history:
            ax.plot(epochs, history["L_data"],
                   label="L_data (Data Fitting)", linewidth=1.5,
                   linestyle="--", color="blue")

        if "L_pde" in history:
            ax.plot(epochs, history["L_pde"],
                   label="L_pde (PDE Residual)", linewidth=1.5,
                   linestyle="--", color="red")

        if "L_bc" in history:
            ax.plot(epochs, history["L_bc"],
                   label="L_bc (Boundary Conditions)", linewidth=1.5,
                   linestyle="--", color="green")

        # Configure plot
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale('log')

        fig.tight_layout()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_solution_comparison(
        self,
        x: np.ndarray,
        time_snapshots: List[float],
        u_pinn: Dict[float, np.ndarray],
        u_analytical: Dict[float, np.ndarray],
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot PINN predictions vs analytical solutions at multiple time snapshots.

        Args:
            x: Spatial coordinates
            time_snapshots: List of time values to plot
            u_pinn: Dictionary mapping time -> PINN solution array
            u_analytical: Dictionary mapping time -> analytical solution array
            save_path: Path to save plot (if provided)

        Returns:
            Tuple of (figure, list of axes)
        """
        n_snapshots = len(time_snapshots)

        # Determine subplot layout (prefer wide layout)
        if n_snapshots <= 2:
            nrows, ncols = 1, n_snapshots
        elif n_snapshots <= 4:
            nrows, ncols = 2, 2
        elif n_snapshots <= 6:
            nrows, ncols = 2, 3
        else:
            nrows = int(np.ceil(n_snapshots / 3))
            ncols = 3

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

        # Flatten axes for easy iteration
        if n_snapshots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else axes

        # Plot each time snapshot
        for idx, t in enumerate(time_snapshots):
            ax = axes[idx]

            # Plot analytical solution
            ax.plot(x, u_analytical[t],
                   label="Analytical", linewidth=2,
                   color="blue", linestyle="-")

            # Plot PINN prediction
            ax.plot(x, u_pinn[t],
                   label="PINN", linewidth=2,
                   color="red", linestyle="--", alpha=0.8)

            # Configure subplot
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.set_title(f"Solution at t = {t:.3f}")
            ax.legend(loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_snapshots, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, axes[:n_snapshots]

    def plot_error_evolution(
        self,
        epochs: np.ndarray,
        errors: Dict[str, List[float]],
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot validation error evolution over training.

        Args:
            epochs: Array of epoch numbers where validation was computed
            errors: Dictionary containing error metrics:
                - "L2_error": L2 error values
                - "relative_error": Relative error values (optional)
                - "max_error": Maximum absolute error (optional)
            save_path: Path to save plot (if provided)

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each error metric
        if "L2_error" in errors:
            ax.plot(epochs, errors["L2_error"],
                   label="L2 Error", linewidth=2,
                   marker='o', markersize=4, color="blue")

        if "relative_error" in errors:
            ax.plot(epochs, errors["relative_error"],
                   label="Relative Error", linewidth=2,
                   marker='s', markersize=4, color="red")

        if "max_error" in errors:
            ax.plot(epochs, errors["max_error"],
                   label="Max Absolute Error", linewidth=2,
                   marker='^', markersize=4, color="green")

        # Add threshold line (5% from requirements)
        if "relative_error" in errors:
            ax.axhline(y=0.05, color='black', linestyle='--',
                      linewidth=1, label="5% Threshold", alpha=0.5)

        # Configure plot
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title("Validation Error Evolution")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax

    def plot_loss_landscape(
        self,
        tuning_results: List[Dict],
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot loss landscape from weight tuning results.

        Args:
            tuning_results: List of dicts containing:
                - "w_data": Data loss weight
                - "w_pde": PDE loss weight
                - "w_bc": BC loss weight
                - "validation_error": Resulting error
            save_path: Path to save plot (if provided)

        Returns:
            Tuple of (figure, axes)
        """
        fig = plt.figure(figsize=(12, 5))

        # Extract data
        w_data = [r["w_data"] for r in tuning_results]
        w_pde = [r["w_pde"] for r in tuning_results]
        validation_errors = [r["validation_error"] for r in tuning_results]

        # Plot 1: w_data vs w_pde colored by error
        ax1 = fig.add_subplot(121)
        scatter1 = ax1.scatter(w_data, w_pde, c=validation_errors,
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel("w_data")
        ax1.set_ylabel("w_pde")
        ax1.set_title("Loss Landscape: Data vs PDE Weights")
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        plt.colorbar(scatter1, ax=ax1, label="Validation Error")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Validation error vs weight combinations
        ax2 = fig.add_subplot(122)
        sorted_results = sorted(tuning_results, key=lambda r: r["validation_error"])
        errors_sorted = [r["validation_error"] for r in sorted_results]
        ax2.plot(np.arange(len(errors_sorted)), errors_sorted,
                linewidth=2, color='blue', marker='o', markersize=4)
        ax2.set_xlabel("Configuration Index (sorted by error)")
        ax2.set_ylabel("Validation Error")
        ax2.set_title("Ranked Tuning Configurations")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, (ax1, ax2)
