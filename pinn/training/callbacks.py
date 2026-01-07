"""Custom callbacks for PINN training monitoring."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class LossLoggingCallback:
    """Log individual loss components (L_bc, L_ic_displacement, L_ic_velocity, L_pde) during training."""

    def __init__(self, log_interval: int = 100):
        """Initialize loss logging callback.

        Args:
            log_interval: Number of epochs between logging events
        """
        self.log_interval = log_interval
        self.history = {
            "L_bc": [],
            "L_ic_displacement": [],
            "L_ic_velocity": [],
            "L_pde": [],
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
        Loss order: [bc, ic_displacement, ic_velocity, pde]
        """
        if self.model.train_state.epoch % self.log_interval == 0:
            # Extract loss components from DeepXDE model state
            # Order: [bc, ic_displacement, ic_velocity, pde]
            losses = self.model.train_state.loss_train

            self.history["L_bc"].append(float(losses[0]))
            self.history["L_ic_displacement"].append(float(losses[1]))
            self.history["L_ic_velocity"].append(float(losses[2]))
            self.history["L_pde"].append(float(losses[3]))
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
                # For traveling wave, compute analytical solution with reflections
                x_vals = self.test_points[:, 0]
                t_vals = self.test_points[:, 1]
                # Use traveling_wave_with_reflections method which requires meshgrid
                x_unique = np.unique(x_vals)
                t_unique = np.unique(t_vals)
                u_analytical_grid = self.analytical_solver.traveling_wave_with_reflections(
                    x=x_unique,
                    t=t_unique,
                    c=self.wave_speed,
                    initial_condition=self.initial_condition_func,
                    L=L,
                    n_reflections=10
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


class EarlyStoppingCallback:
    """Early stopping based on validation error or total loss."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        monitor: str = "loss",
        restore_best_weights: bool = True,
        output_dir: Path = None
    ):
        """Initialize early stopping callback.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor ("loss" or "val_error")
            restore_best_weights: Whether to restore weights from best epoch
            output_dir: Directory to save best model checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.output_dir = output_dir

        self.best_value = float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.model = None
        self.best_weights_path = None

    def set_model(self, model: Any) -> None:
        """Set the model for this callback.

        Args:
            model: DeepXDE Model instance
        """
        self.model = model

        if self.output_dir and self.restore_best_weights:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.best_weights_path = self.output_dir / "best_model_early_stopping.pth"

    def on_train_begin(self) -> None:
        """Callback executed at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf')
        self.best_epoch = 0

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

        Monitors the specified metric and stops training if no improvement
        is observed for 'patience' epochs.
        """
        epoch = self.model.train_state.epoch

        # Get current metric value
        if self.monitor == "loss":
            current_value = float(np.sum(self.model.train_state.loss_train))
        else:
            # For validation error, we would need access to validation callback
            # For now, use total loss as fallback
            current_value = float(np.sum(self.model.train_state.loss_train))

        # Check if improvement
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0

            # Save best weights
            if self.restore_best_weights and self.best_weights_path:
                self.model.save(str(self.best_weights_path))

        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best epoch: {self.best_epoch} with {self.monitor} = {self.best_value:.6e}")

                # Restore best weights
                if self.restore_best_weights and self.best_weights_path and self.best_weights_path.exists():
                    print(f"Restoring best weights from epoch {self.best_epoch}")
                    self.model.restore(str(self.best_weights_path))

    def on_train_end(self) -> None:
        """Callback executed at the end of training."""
        if self.stopped_epoch > 0:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch}")
            print(f"Best epoch was {self.best_epoch} with {self.monitor} = {self.best_value:.6e}")


class R2ValidationCallback:
    """Compute R² scores during training for 2D PINN validation (Task 4.2).

    This callback computes R² (coefficient of determination) for each output field
    (T1, T3, Ux, Uy) at regular intervals during training. Warns if R² falls below
    a configurable threshold.
    """

    def __init__(
        self,
        val_x: np.ndarray,
        val_y: np.ndarray,
        r2_threshold: float = 0.9,
        log_interval: int = 1000
    ):
        """Initialize R² validation callback.

        Args:
            val_x: Validation input data (N_val, 5) [x, y, t, pitch_norm, depth_norm]
            val_y: Validation output data (N_val, 4) [T1, T3, Ux, Uy]
            r2_threshold: R² threshold for warnings (default: 0.9)
            log_interval: Number of epochs between R² computation (default: 1000)

        Preconditions:
            - val_x.shape[1] == 5 (5D input)
            - val_y.shape[1] == 4 (4 outputs: T1, T3, Ux, Uy)
            - val_x.shape[0] == val_y.shape[0] (same number of samples)
            - r2_threshold in (0, 1)
            - log_interval > 0

        Example:
            >>> callback = R2ValidationCallback(val_x, val_y, r2_threshold=0.9)
            >>> callback.set_model(model)
            >>> callback.on_epoch_end()  # Computes R² at intervals
        """
        self.val_x = val_x
        self.val_y = val_y
        self.r2_threshold = r2_threshold
        self.log_interval = log_interval

        # Import R2ScoreCalculator
        from pinn.validation.r2_score import R2ScoreCalculator
        self.r2_calculator = R2ScoreCalculator()

        # History tracking
        self.r2_history = []

        # Model reference (set by DeepXDE)
        self.model = None

        # Field names
        self.field_names = ['T1', 'T3', 'Ux', 'Uy']

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

        Computes R² scores if current epoch is a multiple of log_interval.
        Emits warnings for fields with R² < r2_threshold.
        """
        epoch = self.model.train_state.epoch

        # Only compute R² at specified intervals
        if epoch % self.log_interval != 0:
            return

        # Get predictions on validation set
        y_pred = self.model.predict(self.val_x)

        # Compute R² for each output field
        r2_scores = {}
        for i, field_name in enumerate(self.field_names):
            y_true_field = self.val_y[:, i]
            y_pred_field = y_pred[:, i]
            r2 = self.r2_calculator.compute_r2(y_true_field, y_pred_field)
            r2_scores[field_name] = r2

        # Store in history
        self.r2_history.append({
            'epoch': epoch,
            'scores': r2_scores
        })

        # Print R² scores
        print(f"\nEpoch {epoch} | R² Validation Scores:")
        for field_name, r2 in r2_scores.items():
            print(f"  {field_name}: R² = {r2:.4f}")

        # Check threshold and emit warnings
        low_scoring_fields = [
            (field, r2) for field, r2 in r2_scores.items()
            if r2 < self.r2_threshold
        ]

        if low_scoring_fields:
            print(f"\nWARNING: Low R² scores detected (threshold={self.r2_threshold}):")
            for field_name, r2 in low_scoring_fields:
                print(f"  - {field_name}: R² = {r2:.4f}")
            print("\nRecommendation: Consider tuning hyperparameters:")
            print("  - Adjust loss weights (w_data, w_pde)")
            print("  - Modify learning rate")
            print("  - Increase training epochs")
            print("  - Adjust network architecture (layer_sizes)")
        else:
            print(f"✓ All fields meet quality threshold (R² >= {self.r2_threshold})\n")

    def on_train_end(self) -> None:
        """Callback executed at the end of training."""
        pass
