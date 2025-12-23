"""Plot Generator for PINN Validation Visualization.

Generates publication-quality plots for:
- Training curves (loss vs epochs)
- Solution comparisons (PINN vs analytical)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class PlotGeneratorService:
    """Generate visualization plots for PINN training and validation."""

    def plot_training_curves(
        self,
        history: dict[str, list[float]],
        save_path: Path | None = None,
        log_scale: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
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
        time_snapshots: list[float],
        u_pinn: dict[float, np.ndarray],
        u_analytical: dict[float, np.ndarray],
        save_path: Path | None = None
    ) -> tuple[plt.Figure, list[plt.Axes]]:
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
        errors: dict[str, list[float]],
        save_path: Path | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
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
        tuning_results: list[dict],
        save_path: Path | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
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

    def plot_time_snapshots(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t_list: list[float],
        fdtd_data: dict[float, np.ndarray],
        pinn_pred: dict[float, np.ndarray],
        output_field: str,
        save_path: Path | None = None
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plot 2D spatial distribution snapshots at multiple time points (Task 4.3).

        Generates side-by-side comparison of FDTD ground truth and PINN predictions
        for multiple time snapshots. Each time snapshot shows the 2D spatial distribution
        as a heatmap.

        Args:
            x: 1D array of x-coordinates (nx,) in meters
            y: 1D array of y-coordinates (ny,) in meters
            t_list: List of time values to visualize (minimum 3, Requirement 7.4)
            fdtd_data: Dict mapping time -> 2D array (ny, nx) FDTD ground truth
            pinn_pred: Dict mapping time -> 2D array (ny, nx) PINN predictions
            output_field: Field name ('T1', 'T3', 'Ux', 'Uy')
            save_path: Path to save plot (if provided)

        Returns:
            Tuple of (figure, list of axes)

        Raises:
            ValueError: If fewer than 3 time snapshots provided
            ValueError: If data shapes are inconsistent

        Preconditions:
            - len(t_list) >= 3 (Requirement 7.4)
            - All fdtd_data[t] and pinn_pred[t] have shape (ny, nx)
            - x.shape[0] == nx, y.shape[0] == ny

        Example:
            >>> plot_gen = PlotGeneratorService()
            >>> x = np.linspace(0, 0.04, 40)
            >>> y = np.linspace(0, 0.02, 20)
            >>> t_list = [4.0e-6, 5.0e-6, 6.0e-6]
            >>> fdtd_data = {t: np.random.randn(20, 40) for t in t_list}
            >>> pinn_pred = {t: np.random.randn(20, 40) for t in t_list}
            >>> fig, axes = plot_gen.plot_time_snapshots(
            ...     x, y, t_list, fdtd_data, pinn_pred, 'Ux'
            ... )
        """
        # Validate minimum snapshots (Requirement 7.4)
        if len(t_list) < 3:
            raise ValueError(
                f"At least 3 time snapshots required (got {len(t_list)}). "
                "Requirement 7.4 specifies minimum 3 snapshots."
            )

        # Validate shape consistency
        ny, nx = len(y), len(x)
        for t in t_list:
            if fdtd_data[t].shape != (ny, nx):
                raise ValueError(
                    f"FDTD data shape mismatch at t={t}: "
                    f"expected ({ny}, {nx}), got {fdtd_data[t].shape}"
                )
            if pinn_pred[t].shape != (ny, nx):
                raise ValueError(
                    f"PINN prediction shape mismatch at t={t}: "
                    f"expected ({ny}, {nx}), got {pinn_pred[t].shape}"
                )

        n_snapshots = len(t_list)

        # Create figure with 2 columns (FDTD, PINN) and n_snapshots rows
        fig, axes = plt.subplots(
            nrows=n_snapshots,
            ncols=2,
            figsize=(10, 3.5 * n_snapshots)
        )

        # Handle single snapshot case (axes not 2D array)
        if n_snapshots == 1:
            axes = axes.reshape(1, 2)

        # Convert x, y from meters to millimeters for better readability
        x_mm = x * 1000  # m to mm
        y_mm = y * 1000  # m to mm

        # Plot each time snapshot
        for idx, t in enumerate(t_list):
            # FDTD plot (left column)
            ax_fdtd = axes[idx, 0]
            im_fdtd = ax_fdtd.imshow(
                fdtd_data[t],
                extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
                origin='lower',
                aspect='auto',
                cmap='RdBu_r',
                interpolation='bilinear'
            )
            ax_fdtd.set_xlabel("x (mm)")
            ax_fdtd.set_ylabel("y (mm)")
            ax_fdtd.set_title(f"FDTD: {output_field} at t = {t*1e6:.2f} µs")
            plt.colorbar(im_fdtd, ax=ax_fdtd, label=output_field)

            # PINN plot (right column)
            ax_pinn = axes[idx, 1]
            im_pinn = ax_pinn.imshow(
                pinn_pred[t],
                extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
                origin='lower',
                aspect='auto',
                cmap='RdBu_r',
                interpolation='bilinear'
            )
            ax_pinn.set_xlabel("x (mm)")
            ax_pinn.set_ylabel("y (mm)")
            ax_pinn.set_title(f"PINN: {output_field} at t = {t*1e6:.2f} µs")
            plt.colorbar(im_pinn, ax=ax_pinn, label=output_field)

            # Use same color scale for both plots
            vmin = min(fdtd_data[t].min(), pinn_pred[t].min())
            vmax = max(fdtd_data[t].max(), pinn_pred[t].max())
            im_fdtd.set_clim(vmin, vmax)
            im_pinn.set_clim(vmin, vmax)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # Return flattened axes for easier access
        axes_list = axes.flatten().tolist()
        return fig, axes_list

    def plot_spatial_heatmap(
        self,
        x: np.ndarray,
        y: np.ndarray,
        error: np.ndarray,
        output_field: str,
        save_path: Path | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot 2D spatial error distribution heatmap (Task 4.4).

        Generates heatmap showing absolute error |PINN - FDTD| across the 2D spatial
        domain. Helps identify regions where PINN prediction quality is poor.

        Args:
            x: 1D array of x-coordinates (nx,) in meters
            y: 1D array of y-coordinates (ny,) in meters
            error: 2D array of absolute errors (ny, nx)
            output_field: Field name ('T1', 'T3', 'Ux', 'Uy')
            save_path: Path to save plot (if provided)

        Returns:
            Tuple of (figure, axes)

        Raises:
            ValueError: If error shape doesn't match (ny, nx)

        Preconditions:
            - error.shape == (len(y), len(x))
            - error contains absolute values (>= 0)

        Example:
            >>> plot_gen = PlotGeneratorService()
            >>> x = np.linspace(0, 0.04, 40)
            >>> y = np.linspace(0, 0.02, 20)
            >>> error = np.abs(np.random.randn(20, 40) * 0.01)
            >>> fig, ax = plot_gen.plot_spatial_heatmap(x, y, error, 'Ux')
        """
        ny, nx = len(y), len(x)

        # Validate error shape
        if error.shape != (ny, nx):
            raise ValueError(
                f"Error shape mismatch: expected ({ny}, {nx}), got {error.shape}"
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert to millimeters for better readability
        x_mm = x * 1000
        y_mm = y * 1000

        # Plot heatmap
        im = ax.imshow(
            error,
            extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
            origin='lower',
            aspect='auto',
            cmap='hot',
            interpolation='bilinear'
        )

        # Configure plot
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Spatial Error Distribution: |PINN - FDTD| for {output_field}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=f"Absolute Error ({output_field})")

        # Add statistics text
        error_mean = np.mean(error)
        error_max = np.max(error)
        error_std = np.std(error)

        stats_text = (
            f"Mean: {error_mean:.2e}\n"
            f"Max: {error_max:.2e}\n"
            f"Std: {error_std:.2e}"
        )

        # Place text box in upper right
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig, ax
