"""Custom callbacks for PINN training monitoring."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class LossLoggingCallback:
    """Log individual loss components (L_data, L_pde, L_bc) during training."""

    def __init__(self, log_interval: int = 100):
        """Initialize loss logging callback.

        Args:
            log_interval: Number of epochs between logging events
        """
        self.log_interval = log_interval
        self.history = {
            "L_data": [],
            "L_pde": [],
            "L_bc": [],
            "total_loss": []
        }
        self.model = None

    def set_model(self, model: Any) -> None:
        """Set the model for this callback (required by DeepXDE).

        Args:
            model: DeepXDE Model instance
        """
        self.model = model

    def on_train_begin(self) -> None:
        """Callback executed at the beginning of training."""
        pass

    def on_epoch_begin(self) -> None:
        """Callback executed at the beginning of each epoch."""
        pass

    def on_batch_begin(self) -> None:
        """Callback executed at the beginning of each batch."""
        pass

    def on_batch_end(self) -> None:
        """Callback executed at the end of each batch."""
        pass

    def on_epoch_end(self) -> None:
        """Callback executed at the end of each epoch.

        Logs loss components if current epoch is a multiple of log_interval.
        """
        if self.model.train_state.epoch % self.log_interval == 0:
            # Extract loss components from DeepXDE model state
            losses = self.model.train_state.loss_train

            self.history["L_data"].append(float(losses[0]))
            self.history["L_pde"].append(float(losses[1]))
            self.history["L_bc"].append(float(losses[2]))
            self.history["total_loss"].append(float(np.sum(losses)))

    def on_train_end(self) -> None:
        """Callback executed at the end of training."""
        pass


class CheckpointCallback:
    """Save model checkpoints at configurable intervals."""

    def __init__(self, output_dir: Path, save_interval: int = 1000):
        """Initialize checkpoint callback.

        Args:
            output_dir: Directory where checkpoints will be saved
            save_interval: Number of epochs between checkpoint saves
        """
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.best_loss = float("inf")
        self.model = None

    def set_model(self, model: Any) -> None:
        """Set the model for this callback (required by DeepXDE).

        Args:
            model: DeepXDE Model instance
        """
        self.model = model
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self) -> None:
        """Callback executed at the beginning of training."""
        pass

    def on_epoch_begin(self) -> None:
        """Callback executed at the beginning of each epoch."""
        pass

    def on_batch_begin(self) -> None:
        """Callback executed at the beginning of each batch."""
        pass

    def on_batch_end(self) -> None:
        """Callback executed at the end of each batch."""
        pass

    def on_epoch_end(self) -> None:
        """Callback executed at the end of each epoch.

        Saves checkpoint if current epoch is a multiple of save_interval.
        Tracks best checkpoint based on lowest total loss.
        """
        if self.model.train_state.epoch % self.save_interval == 0:
            # Compute current total loss
            current_loss = float(np.sum(self.model.train_state.loss_train))

            # Update best loss tracking
            if current_loss < self.best_loss:
                self.best_loss = current_loss

            # Save checkpoint
            checkpoint_path = (
                self.output_dir /
                f"checkpoint_epoch_{self.model.train_state.epoch}.pth"
            )
            self.model.save(str(checkpoint_path))

    def on_train_end(self) -> None:
        """Callback executed at the end of training."""
        pass


class ValidationCallback:
    """Compute L2 error vs. analytical solution during training."""

    def __init__(
        self,
        analytical_solver: Any,
        error_metrics: Any,
        validation_interval: int,
        domain_config: Any,
        wave_speed: float,
        error_threshold: float = 0.05,
        n_mode: int = 1,
        bc_type: str = "dirichlet",
        initial_condition_func: Any = None,
        enable_validation: bool = True
    ):
        """Initialize validation callback.

        Args:
            analytical_solver: AnalyticalSolutionGeneratorService instance
            error_metrics: ErrorMetricsService instance
            validation_interval: Number of epochs between validation checks
            domain_config: DomainConfig object or dict with domain bounds (x_min, x_max, t_min, t_max)
            wave_speed: Wave propagation speed c
            error_threshold: Relative error threshold for warnings (default 5%)
            n_mode: Mode number for standing wave (default: fundamental mode)
            bc_type: Boundary condition type ("dirichlet", "neumann", or "traveling_wave")
            initial_condition_func: Initial condition function for traveling wave validation
            enable_validation: Enable validation against analytical solution (default: True)
        """
        self.analytical_solver = analytical_solver
        self.error_metrics = error_metrics
        self.validation_interval = validation_interval
        self.domain_config = domain_config
        self.wave_speed = wave_speed
        self.error_threshold = error_threshold
        self.n_mode = n_mode
        self.bc_type = bc_type
        self.initial_condition_func = initial_condition_func
        self.enable_validation = enable_validation

        self.errors = []
        self.relative_errors = []
        self.model = None
        self.test_points = None

    def set_model(self, model: Any) -> None:
        """Set the model for this callback (required by DeepXDE).

        Args:
            model: DeepXDE Model instance
        """
        self.model = model

        # Generate test points for validation
        # Support both dict and Pydantic model
        if hasattr(self.domain_config, 'x_min'):
            # Pydantic model
            x_min = self.domain_config.x_min
            x_max = self.domain_config.x_max
            t_min = self.domain_config.t_min
            t_max = self.domain_config.t_max
        else:
            # Dictionary
            x_min = self.domain_config['x_min']
            x_max = self.domain_config['x_max']
            t_min = self.domain_config['t_min']
            t_max = self.domain_config['t_max']

        x_test = np.linspace(x_min, x_max, 100)
        t_test = np.linspace(t_min, t_max, 100)
        X_test, T_test = np.meshgrid(x_test, t_test)
        self.test_points = np.column_stack([X_test.flatten(), T_test.flatten()])

    def on_train_begin(self) -> None:
        """Callback executed at the beginning of training."""
        pass

    def on_epoch_begin(self) -> None:
        """Callback executed at the beginning of each epoch."""
        pass

    def on_batch_begin(self) -> None:
        """Callback executed at the beginning of each batch."""
        pass

    def on_batch_end(self) -> None:
        """Callback executed at the end of each batch."""
        pass

    def on_epoch_end(self) -> None:
        """Callback executed at the end of each epoch.

        Computes L2 error vs. analytical solution if current epoch
        is a multiple of validation_interval and validation is enabled.
        """
        if not self.enable_validation:
            return

        if self.model.train_state.epoch % self.validation_interval == 0:
            # Get PINN predictions
            u_pred = self.model.predict(self.test_points)

            # Get analytical solution
            # Support both dict and Pydantic model
            if hasattr(self.domain_config, 'x_min'):
                # Pydantic model
                x_min = self.domain_config.x_min
                x_max = self.domain_config.x_max
            else:
                # Dictionary
                x_min = self.domain_config['x_min']
                x_max = self.domain_config['x_max']

            L = x_max - x_min

            # Select solution type based on boundary condition
            if self.bc_type == "neumann":
                solution_type = "standing_neumann"
                u_exact = self.analytical_solver.evaluate_at_points(
                    self.test_points,
                    L=L,
                    c=self.wave_speed,
                    solution_type=solution_type,
                    n=self.n_mode
                ).reshape(-1, 1)
            elif self.bc_type == "traveling_wave":
                # For traveling wave, compute analytical solution manually
                x_vals = self.test_points[:, 0]
                t_vals = self.test_points[:, 1]
                # Use traveling_wave method which requires meshgrid
                x_unique = np.unique(x_vals)
                t_unique = np.unique(t_vals)
                u_analytical_grid = self.analytical_solver.traveling_wave(
                    x=x_unique,
                    t=t_unique,
                    c=self.wave_speed,
                    initial_condition=self.initial_condition_func
                )
                # Flatten to match test_points order
                u_exact = u_analytical_grid.T.flatten().reshape(-1, 1)
            else:  # dirichlet
                solution_type = "standing"
                u_exact = self.analytical_solver.evaluate_at_points(
                    self.test_points,
                    L=L,
                    c=self.wave_speed,
                    solution_type=solution_type,
                    n=self.n_mode
                ).reshape(-1, 1)

            # Compute errors
            l2_error = self.error_metrics.l2_error(u_pred, u_exact)
            rel_error = self.error_metrics.relative_error(u_pred, u_exact)

            # Store errors
            self.errors.append(l2_error)
            self.relative_errors.append(rel_error)

            # Check threshold and warn if needed
            if rel_error > self.error_threshold:
                print(
                    f"WARNING: High validation error ({rel_error:.4f}) at epoch "
                    f"{self.model.train_state.epoch}, may need hyperparameter tuning"
                )

    def on_train_end(self) -> None:
        """Callback executed at the end of training."""
        pass


class DivergenceDetectionCallback:
    """Detect loss divergence (NaN or excessive values) and halt training."""

    def __init__(
        self,
        output_dir: Path,
        nan_threshold: float = 1e10
    ):
        """Initialize divergence detection callback.

        Args:
            output_dir: Directory to save diagnostic information
            nan_threshold: Maximum allowed loss value before halting
        """
        self.output_dir = output_dir
        self.nan_threshold = nan_threshold
        self.divergence_detected = False
        self.model = None

    def set_model(self, model: Any) -> None:
        """Set the model for this callback (required by DeepXDE).

        Args:
            model: DeepXDE Model instance
        """
        self.model = model

    def on_train_begin(self) -> None:
        """Callback executed at the beginning of training."""
        pass

    def on_epoch_begin(self) -> None:
        """Callback executed at the beginning of each epoch."""
        pass

    def on_batch_begin(self) -> None:
        """Callback executed at the beginning of each batch."""
        pass

    def on_batch_end(self) -> None:
        """Callback executed at the end of each batch."""
        pass

    def on_epoch_end(self) -> None:
        """Callback executed at the end of each epoch.

        Checks for NaN or excessive loss values and halts training if detected.
        Saves diagnostic information for debugging.
        """
        losses = self.model.train_state.loss_train
        total_loss = float(np.sum(losses))

        # Check for NaN or excessive loss
        if np.isnan(total_loss) or total_loss > self.nan_threshold:
            self.divergence_detected = True
            self.model.stop_training = True

            # Save diagnostic information
            diagnostic_info = {
                "epoch": self.model.train_state.epoch,
                "loss_values": {
                    "L_data": float(losses[0]) if not np.isnan(losses[0]) else "NaN",
                    "L_pde": float(losses[1]) if not np.isnan(losses[1]) else "NaN",
                    "L_bc": float(losses[2]) if not np.isnan(losses[2]) else "NaN",
                    "total": total_loss if not np.isnan(total_loss) else "NaN"
                },
                "divergence_detected": True,
                "reason": "NaN detected" if np.isnan(total_loss) else f"Loss exceeded threshold ({self.nan_threshold})"
            }

            diagnostic_path = self.output_dir / "divergence_diagnostic.json"
            with open(diagnostic_path, 'w') as f:
                json.dump(diagnostic_info, f, indent=2)

            print(f"ERROR: Loss divergence detected at epoch {self.model.train_state.epoch}")
            print(f"Diagnostic information saved to: {diagnostic_path}")

    def on_train_end(self) -> None:
        """Callback executed at the end of training."""
        pass
